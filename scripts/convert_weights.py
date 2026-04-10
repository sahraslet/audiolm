import re

from typing import Any, Optional

import torch

def get_mapped_key(key: str, mapping_dict: dict[str, str]) -> str:
    try:
        # Checks if there is a layer # in the key
        if any(k.isdigit() for k in key.split(".")):
            # Replace layer number with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = mapping_dict[abstract_key]
            new_key = new_key.format(layer_num)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format. "
        ) from e

    return new_key


_FROM_HF = {
     "model.embed_tokens.weight": "model.embed_tokens.weight",
     "model.layers.{}.self_attn.q_proj.weight": "model.layers.{}.self_attn.q_proj.weight",
     "model.layers.{}.self_attn.q_proj.bias": "model.layers.{}.self_attn.q_proj.bias",
     "model.layers.{}.self_attn.k_proj.weight": "model.layers.{}.self_attn.k_proj.weight",
     "model.layers.{}.self_attn.k_proj.bias": "model.layers.{}.self_attn.k_proj.bias",
     "model.layers.{}.self_attn.v_proj.weight": "model.layers.{}.self_attn.v_proj.weight",
     "model.layers.{}.self_attn.v_proj.bias": "model.layers.{}.self_attn.v_proj.bias",
     "model.layers.{}.self_attn.o_proj.weight": "model.layers.{}.self_attn.o_proj.weight",
     "model.layers.{}.mlp.gate_proj.weight": "model.layers.{}.mlp.gate_proj.weight",
     "model.layers.{}.mlp.up_proj.weight": "model.layers.{}.mlp.up_proj.weight",
     "model.layers.{}.mlp.down_proj.weight": "model.layers.{}.mlp.down_proj.weight",
     "model.layers.{}.input_layernorm.weight": "model.layers.{}.input_layernorm.weight",
     "model.layers.{}.post_attention_layernorm.weight": "model.layers.{}.post_attention_layernorm.weight",
     "model.norm.weight": "model.norm.weight",
}