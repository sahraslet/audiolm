from audiolm.qwen import QwenCausalLM
from audiolm.config import QwenConfig

from transformers import Qwen2Config, AutoModelForCausalLM, AutoTokenizer
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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = QwenCausalLM(cfg)
sd = torch.load("ckpt/qwen.bin", weights_only=True)
missing, unexpected = model.load_state_dict(sd)

print("loaded weights! starting generation :)")

prefix = "A long time ago"
input_ids = tokenizer(prefix, return_tensors="pt").input_ids
MAX_NEW_TOKENS = 15
TEMPERATURE = 0

y = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(y[0].tolist()))
