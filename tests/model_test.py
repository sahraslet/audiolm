from audiolm.qwen import QwenCausalLM
from audiolm.config import QwenConfig

from transformers import Qwen2Config, AutoModelForCausalLM
from safetensors.torch import save_model, load_file
import torch

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
    pad_token_id=151643,
    tie_word_embeddings=True
)

model = QwenCausalLM(cfg)
sd = torch.load("ckpt/qwen.bin")

print("loading sq into model")
missing, unexpected = model.load_state_dict(sd)

print("missing: ", missing)
print("unexpected: ", unexpected)

