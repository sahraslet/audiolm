from audiolm.layers import QwenMLP, QwenRoPE
from audiolm.qwen import QwenConfig

import torch
import torch.nn as nn
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

cfg = QwenConfig(
    block_size=64,
    d_model=896,
    d_ffn=4864,
    n_layers=24,
    n_heads=14,
    n_kv_heads=2,
    max_positional_embed=32768,
    rmsnorm_eps=1e-06,
    rope_theta=1000000.0,
    dropout=0.0,
    vocab_size=151936,
    activation='silu',
    proj_bias=False,
    gate_bias=False,
)

# mlp = QwenMLP(config=cfg)

# tensor = torch.randn((2, 4, 8))
# print(tensor)
# print(tensor.shape)

# out = mlp(tensor)
# print(out)
# print(out.shape)

# print(mlp)

my_rope = QwenRoPE(cfg)
hf_cfg = Qwen2Config.from_pretrained("Qwen/Qwen2.5-0.5B")
hf_rope = Qwen2RotaryEmbedding(hf_cfg)

x = torch.randn((2, 2, 10, 10))
pos_ids = torch.arange(10).unsqueeze(0)

# print(hf_cfg.head_dim)
m_cos, m_sin = my_rope(x, pos_ids)
h_cos, h_sin = hf_rope(x, pos_ids)

print(m_cos)
print(h_cos)

assert torch.allclose(m_cos, h_cos), "cos are not equivalent"
assert torch.allclose(m_sin, h_sin), "sin are not equivalent"