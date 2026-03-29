# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored HyperCLOVAX Vision config with transformers v5 compatibility fix.

The original remote code config at
naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B does not handle
empty initialization (text_config=None), which breaks transformers v5's
@strict config validation.

TODO: Remove this file once HyperCLOVAX is upstreamed to transformers.
Tracking PR: https://github.com/huggingface/transformers/pull/44956
"""

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class HCXVisionConfig(PretrainedConfig):
    model_type = "hyperclovax_vlm"
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config_attribute_map = {
        "n_embd": "hidden_size",
        "n_positions": "max_position_embeddings",
        "n_head": "num_attention_heads",
        "n_layer": "num_hidden_layers",
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        use_nth_layer=-2,
        img_start_id=100009,
        decoder_max_length=4096,
        anyres=False,
        unpad=False,
        max_num_grids=-1,
        num_queries_vis_abstractor=-1,
        ignore_index=-100,
        proj_pos_emb=True,
        proj_prenorm=False,
        use_1x1_grid=False,
        **kwargs,
    ):
        for key, val in self.text_config_attribute_map.items():
            if text_config is not None and key in text_config:
                text_config[val] = text_config.pop(key)

        self.text_config = None
        if text_config is not None:
            _text_config = AutoConfig.for_model(text_config["model_type"])
            self.text_config = _text_config.from_dict(text_config)
            self.hidden_size = text_config.get(
                "hidden_size", text_config.get("n_embd"))

        self.vision_config = None
        if vision_config is not None:
            _vision_config = AutoConfig.for_model(
                vision_config["model_type"])
            self.vision_config = _vision_config.from_dict(vision_config)

        self.use_nth_layer = use_nth_layer
        self.decoder_max_length = decoder_max_length
        self.anyres = anyres
        self.unpad = unpad
        self.max_num_grids = max_num_grids
        self.num_queries_vis_abstractor = num_queries_vis_abstractor
        self.img_start_id = img_start_id
        self.ignore_index = ignore_index
        self.proj_pos_emb = proj_pos_emb
        self.proj_prenorm = proj_prenorm
        self.use_1x1_grid = use_1x1_grid
        super().__init__(**kwargs)

    def get_text_config(self, decoder=False):
        if self.text_config is not None:
            return self.text_config
        return self
