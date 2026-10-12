# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UBTree inference reference (arXiv:2609.39972, equations 4–12).

This is an experimental correctness oracle, not a registered serving method.
Scoring uses PyTorch; bounded search and path traversal deliberately run on the
CPU. No checkpoint format, target attention backend, or KV allocator is assumed.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def precompute_codebook(
    target_embeddings: torch.Tensor,
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply one trained bias-free SiLU MLP to frozen target embeddings.

    Call once for each of the predecessor and successor MLPs after loading
    trained weights. The returned lookup table has shape [vocabulary, rank].
    """
    with torch.no_grad():
        return F.linear(
            F.silu(F.linear(target_embeddings, input_weight)), output_weight
        )


def _check_candidates(candidate_ids: torch.Tensor) -> tuple[int, int]:
    if candidate_ids.ndim != 2 or min(candidate_ids.shape) < 1:
        raise ValueError("candidate_ids must have shape [depth >= 1, top_k >= 1]")
    if candidate_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("candidate_ids must contain integer target vocabulary IDs")
    if (candidate_ids < 0).any():
        raise ValueError("candidate_ids must be nonnegative")
    sorted_ids = candidate_ids.sort(dim=-1).values
    if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any():
        raise ValueError("candidate IDs must be unique within each depth")
    return candidate_ids.shape


def score_ubtree_edges(
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    projected_hidden: torch.Tensor,
    predecessor_codebook: torch.Tensor,
    successor_codebook: torch.Tensor,
    anchor_token_id: int,
    log_unigram_scale: torch.Tensor,
    log_bigram_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score all adjacent candidate pairs for one request in parallel.

    Args:
        candidate_ids: Target vocabulary IDs [depth, K], without the anchor.
        unary_logits: Proposer logits at those IDs [depth, K].
        projected_hidden: Learned projection P(h_i) [depth, rank].
        predecessor_codebook: Precomputed phi [vocabulary, rank].
        successor_codebook: Precomputed psi [vocabulary, rank].
        anchor_token_id: Already sampled target token at the root.
        log_unigram_scale: Learned rho [depth], exponentiated before use.
        log_bigram_scale: Learned kappa [depth], exponentiated before use.

    Returns:
        FP32 root scores [K] and transition scores [depth - 1, K, K].
        Transition axes are predecessor then successor. These are candidate
        ranking scores, not target probabilities or rejection-sampling inputs.

    """
    depth, top_k = _check_candidates(candidate_ids)
    if predecessor_codebook.ndim != 2 or (
        predecessor_codebook.shape != successor_codebook.shape
    ):
        raise ValueError("codebooks must share shape [vocabulary, rank]")
    vocab_size, rank = predecessor_codebook.shape
    if not 0 <= anchor_token_id < vocab_size or (candidate_ids >= vocab_size).any():
        raise ValueError(
            "anchor and candidate IDs must belong to the target vocabulary"
        )
    if unary_logits.shape != (depth, top_k) or projected_hidden.shape != (depth, rank):
        raise ValueError("unary logits or projected hidden states have the wrong shape")
    if log_unigram_scale.shape != (depth,) or log_bigram_scale.shape != (depth,):
        raise ValueError("calibration scales must have one value per draft depth")

    successors = successor_codebook[candidate_ids].float()
    hidden = projected_hidden.float()
    root_delta = (
        predecessor_codebook[anchor_token_id].float() * hidden[0] * successors[0]
    ).sum(dim=-1)
    transitions = torch.einsum(
        "dpr,dkr->dpk",
        predecessor_codebook[candidate_ids[:-1]].float() * hidden[1:, None, :],
        successors[1:],
    )
    alpha = log_unigram_scale.float().exp()
    lam = log_bigram_scale.float().exp()
    root = alpha[0] * unary_logits[0].float() + lam[0] * root_delta
    edges = (
        alpha[1:, None, None] * unary_logits[1:, None, :].float()
        + lam[1:, None, None] * transitions
    )
    return root, edges


@dataclass(frozen=True)
class UBTree:
    """Compact CPU reference tree; index 0 is the anchor, outside the budget.

    Nodes are in topological order. The same token under different parents has
    distinct indices. parent_indices[0] is -1; depths[0] and log_scores[0] are 0.
    """

    token_ids: torch.Tensor
    parent_indices: torch.Tensor
    depths: torch.Tensor
    log_scores: torch.Tensor

    def attention_mask(self) -> torch.Tensor:
        """Return [nodes, nodes] bool mask; True permits self/ancestor attention.

        This covers only tree-local keys. A target backend must also expose the
        verified prefix to every node, and translate this mask to its convention.
        """
        count = self.token_ids.numel()
        mask = torch.eye(count, dtype=torch.bool)
        for node in range(1, count):
            mask[node] |= mask[int(self.parent_indices[node])]
        return mask

    def position_ids(self, anchor_position: int) -> torch.Tensor:
        """Use prefix depth, not the flattened node index, for target positions."""
        if anchor_position < 0:
            raise ValueError("anchor_position must be nonnegative")
        return self.depths + anchor_position


def build_ubtree(
    candidate_ids: torch.Tensor,
    root_scores: torch.Tensor,
    transition_scores: torch.Tensor,
    *,
    anchor_token_id: int,
    node_budget: int,
    search_width: int,
) -> UBTree:
    """Construct the paper's bounded, prefix-closed tree with depth bonus zero.

    Retain at most search_width nodes across each entire depth, then globally
    retain node_budget non-root nodes. Search is bounded, not exhaustive. Ties
    favor ancestors; remaining ties follow parent and input candidate order.
    Inputs may be on a GPU, but materialization synchronizes/copies to the CPU.
    """
    depth, top_k = _check_candidates(candidate_ids)
    if node_budget < 0 or search_width < 1 or anchor_token_id < 0:
        raise ValueError("require node_budget >= 0, search_width >= 1, anchor >= 0")
    if root_scores.shape != (top_k,) or transition_scores.shape != (
        depth - 1,
        top_k,
        top_k,
    ):
        raise ValueError("scores must have shapes [K] and [depth - 1, K, K]")
    for scores in (root_scores, transition_scores):
        if not scores.is_floating_point() or not torch.isfinite(scores).all():
            raise ValueError("candidate scores must be finite floating-point values")

    ids = candidate_ids.detach().cpu().tolist()
    root_log_probs = root_scores.detach().double().cpu().log_softmax(-1).tolist()
    edge_log_probs = transition_scores.detach().double().cpu().log_softmax(-1).tolist()
    # Node records: token, parent, depth, candidate index, accumulated log score.
    nodes = [(anchor_token_id, -1, 0, -1, 0.0)]
    frontier = [0]
    for step in range(depth) if node_budget else ():
        extensions = []
        for parent in frontier:
            previous = nodes[parent]
            log_probs = (
                root_log_probs if step == 0 else edge_log_probs[step - 1][previous[3]]
            )
            for candidate, token_id in enumerate(ids[step]):
                extensions.append(
                    (
                        token_id,
                        parent,
                        step + 1,
                        candidate,
                        previous[4] + log_probs[candidate],
                    )
                )
        extensions.sort(key=lambda node: (-node[4], node[1], node[3]))
        frontier = []
        for node in extensions[:search_width]:
            frontier.append(len(nodes))
            nodes.append(node)

    ranked = sorted(range(1, len(nodes)), key=lambda i: (-nodes[i][4], nodes[i][2], i))
    selected = [0, *sorted(ranked[:node_budget])]
    remap = {old: new for new, old in enumerate(selected)}
    return UBTree(
        token_ids=torch.tensor([nodes[i][0] for i in selected], dtype=torch.long),
        parent_indices=torch.tensor(
            [-1, *[remap[nodes[i][1]] for i in selected[1:]]], dtype=torch.long
        ),
        depths=torch.tensor([nodes[i][2] for i in selected], dtype=torch.long),
        log_scores=torch.tensor([nodes[i][4] for i in selected], dtype=torch.float64),
    )


@dataclass(frozen=True)
class UBTreeAcceptance:
    """Reference commit plan; no mutation of the target KV cache is performed."""

    output_token_ids: torch.Tensor
    # Gather these verification rows in order: anchor, then matched descendants.
    kv_row_indices: torch.Tensor
    # An unmatched target sample has no computed KV row and must be fed next round.
    has_bonus_token: bool


def follow_target_samples(
    tree: UBTree,
    target_next_token_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...] = (),
) -> UBTreeAcceptance:
    """Follow one target-sampled path until a miss, EOS, or the output limit.

    target_next_token_ids[i] must be sampled from the target distribution AFTER
    tree node i under its exact prefix. Only samples on the reached path matter;
    do not choose the longest among independently sampled leaf paths. This
    function supplies traversal/commit semantics, not a target sampler.
    """
    if target_next_token_ids.shape != tree.token_ids.shape or (
        target_next_token_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("provide one integer next-token sample per tree node")
    if max_new_tokens < 0 or (target_next_token_ids < 0).any():
        raise ValueError("max_new_tokens and sampled token IDs must be nonnegative")
    samples = target_next_token_ids.detach().cpu().tolist()
    children = {
        (int(tree.parent_indices[i]), int(tree.token_ids[i])): i
        for i in range(1, tree.token_ids.numel())
    }
    node, bonus = 0, False
    output, kv_rows = [], [0]
    for _ in range(max_new_tokens):
        token = samples[node]
        output.append(token)
        child = children.get((node, token))
        if child is None:
            bonus = True
        else:
            kv_rows.append(child)
            node = child
        if child is None or token in eos_token_ids:
            break
    return UBTreeAcceptance(
        torch.tensor(output, dtype=torch.long),
        torch.tensor(kv_rows, dtype=torch.long),
        bonus,
    )
