# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the pure reference without initializing vLLM's compiled GPU platform.

This keeps the standalone Transformers experiment runnable from a source checkout
on a GPU host without installing a vLLM engine in its reference environment.
"""

import importlib.util
import sys
from pathlib import Path

_path = Path(__file__).resolve().parents[2] / "vllm/v1/spec_decode/ubtree.py"
_spec = importlib.util.spec_from_file_location("_vllm_ubtree_reference", _path)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

UBTree = _module.UBTree
UBTreeAcceptance = _module.UBTreeAcceptance
build_ubtree = _module.build_ubtree
follow_target_samples = _module.follow_target_samples
precompute_codebook = _module.precompute_codebook
score_ubtree_edges = _module.score_ubtree_edges
