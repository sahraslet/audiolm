from huggingface_hub import snapshot_download, hf_hub_download
from transformers import Qwen2Config, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
torch.save(model.state_dict(), "ckpt/qwen.bin")