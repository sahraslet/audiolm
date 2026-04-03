from audiolm.audiolm import AudioLM
from audiolm.config import QwenConfig
from audiolm.trainer import Trainer
from audiolm.functional import audio_lm_loss
from datacollator import AudioLMCollator
import torch.nn as nn


from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


# ===========================================================================
# Args
# ===========================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer_path",      type=str, default="Qwen/Qwen2.5-0.5B")
parser.add_argument("--model_checkpoint",    type=str, default=None)  # optionaler .pth Checkpoint
parser.add_argument("--dataset_path",        type=str, required=True)
parser.add_argument("--checkpoint_dir",      type=str, required=True)
parser.add_argument("--logfile_path",        type=str, required=True)
parser.add_argument("--wandb_project_name",  type=str, required=True)
parser.add_argument("--wandb_entity",        type=str, required=True)
parser.add_argument("--wandb_run_name",      type=str, required=True)
parser.add_argument("--lr",                  type=float, default=5e-5)
parser.add_argument("--device",              type=str, default="cuda")
parser.add_argument("--num_epochs",          type=int, default=1)
parser.add_argument("--eval_every",          type=int, default=5000)
parser.add_argument("--save_every",          type=int, default=5000)
parser.add_argument("--grad_accumulation_steps", type=int, default=1)
parser.add_argument("--batch_size",          type=int, default=2)
parser.add_argument("--push_to_hub",   action="store_true", default=False)
parser.add_argument("--hub_repo_id",   type=str, default=None)

args = parser.parse_args()
import os
os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(os.path.dirname(args.logfile_path), exist_ok=True)

# ===========================================================================
# Tokenizer
# ===========================================================================

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
qwen = AutoModelForCausalLM.from_pretrained(args.tokenizer_path)
qwen_vocab_size = qwen.model.embed_tokens.weight.shape[0]  # 151936
print(f"Tokenizer vocab: {len(tokenizer)} | Embedding size: {qwen_vocab_size}")

special_tokens = ["<|audio|>", "<|audio_bos|>", "<|audio_pad|>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

for t in special_tokens:
    print(f"{t} → {tokenizer.convert_tokens_to_ids(t)}")

# vocab_size nach add_special_tokens — embed_tokens muss darauf resized werden
total_vocab_size = qwen_vocab_size + len(special_tokens)  # 151936 + 3 = 151939
print(f"Total vocab size: {total_vocab_size}")


# ===========================================================================
# Config
# ===========================================================================

cfg = QwenConfig(
    block_size=1024,
    d_model=896,
    d_ffn=4864,
    n_layers=24,
    n_heads=14,
    n_kv_heads=2,
    max_positional_embed=32768,
    rmsnorm_eps=1e-06,
    rope_theta=1000000.0,
    dropout=0.0,
    vocab_size=qwen_vocab_size,       # Qwen embed_tokens size
    text_vocab_size=total_vocab_size,  # für AudioLM assert + Loss
    activation='silu',
    pad_token_id=tokenizer.pad_token_id,
    audio_token_id=tokenizer.convert_tokens_to_ids("<|audio|>"),
    tie_word_embeddings=True,
    audio_vocab_size=1024,
    n_codebooks=8,
)


# ===========================================================================
# Modell
# ===========================================================================

model = AudioLM(cfg)


# 2. Pretrained Qwen Weights laden (nur Backbone, neue Parameter bleiben zufällig)
print("Loading pretrained Qwen weights...")
qwen = AutoModelForCausalLM.from_pretrained(args.tokenizer_path)
missing, unexpected = model.model.load_state_dict(qwen.state_dict(), strict=False)
print(f"Missing keys (neue Parameter, erwartet): {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")
del qwen  # VRAM freigeben

# 5. Embeddings auf neue Größe erweitern
old_emb = model.model.model.embed_tokens
new_emb = nn.Embedding(total_vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
print(f"qwen_embed_size: {qwen_vocab_size}")
print(f"old_emb.weight.shape: {old_emb.weight.shape}")
new_emb.weight.data[:qwen_vocab_size] = old_emb.weight.data
nn.init.normal_(new_emb.weight.data[qwen_vocab_size:], mean=0.0, std=0.02)
model.model.model.embed_tokens = new_emb
model.model.lm_head.weight = new_emb.weight  # weight-tying

# Config updaten
cfg.vocab_size = total_vocab_size
cfg.text_vocab_size = total_vocab_size
#cfg.audio_token_id = tokenizer.convert_tokens_to_ids("<|audio|>")

# 3. Optionaler eigener Checkpoint (Resume Training)
if args.model_checkpoint is not None:
    print(f"Resuming from checkpoint: {args.model_checkpoint}")
    ckpt = torch.load(args.model_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    print("Checkpoint loaded successfully")


# ===========================================================================
# Dataset + DataLoader
# ===========================================================================

print(f"Loading dataset from {args.dataset_path}...")
ds = load_dataset(args.dataset_path)
ds.set_format("torch", columns=["text_ids", "audio_codes", "attention_mask"])

train_ds = ds["train"]
val_ds   = ds["validation"]

collator = AudioLMCollator(text_pad_token_id=tokenizer.pad_token_id, max_length=cfg.block_size)

train_dl = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collator,
    num_workers=0,
    pin_memory=True,
)
valid_dl = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    collate_fn=collator,
    num_workers=0,
    pin_memory=True,
)

print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")


# ===========================================================================
# Loss + Optimizer
# ===========================================================================

loss_fn = partial(
    audio_lm_loss,
    n_codebooks=cfg.n_codebooks,
    audio_vocab_size=cfg.audio_vocab_size,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=0.01,
    betas=(0.9, 0.95),
)


# ===========================================================================
# Trainer
# ===========================================================================

trainer = Trainer(
    config=cfg,
    checkpoint_dir=args.checkpoint_dir,
    log_file=args.logfile_path,
    wandb_project_name=args.wandb_project_name,
    wandb_entity=args.wandb_entity,
    wandb_run_name=args.wandb_run_name,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=args.device,
    push_to_hub=args.push_to_hub,
    hub_repo_id=args.hub_repo_id,
)

# Resume Trainer-State falls Checkpoint vorhanden
if args.model_checkpoint is not None:
    trainer.load_checkpoint(args.model_checkpoint)


# ===========================================================================
# Train
# ===========================================================================

trainer.train(
    train_dataloader=train_dl,
    val_dataloader=valid_dl,
    num_epochs=args.num_epochs,
    eval_every=args.eval_every,
    save_every=args.save_every,
    grad_accumulation_steps=args.grad_accumulation_steps,
)