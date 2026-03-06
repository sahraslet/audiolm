from audiolm.qwen import QwenCausalLM
from audiolm.config import QwenConfig
from audiolm.trainer import Trainer

import argparse
import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()

parser.add_argument("--dataset_path", type=str)
parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument("--logfile_path", type=str)
parser.add_argument("--wandb_project_name", type=str)
parser.add_argument("--wandb_entity", type=str)
parser.add_argument("--wandb_run_name", type=str)
parser.add_argument("--lr", type=float)
parser.add_argument("--device", type=str)
parser.add_argument("--num_epochs", type=int)
parser.add_argument("--eval_every", type=int)
parser.add_argument("--save_every", type=int)
parser.add_argument("--grad_accumulation_steps", type=int)


args = parser.parse_args()

cfg = QwenConfig(
    block_size=128,
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

ds = load_from_disk(args.dataset_path)
ds.set_format("torch", columns=["input_ids", "attention_mask"])
train_ds = ds['train']
val_ds = ds['validation']

train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
valid_dl = DataLoader(val_ds, batch_size=1, shuffle=False)

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

trainer = Trainer(
    config=testcfg,
    checkpoint_dir=args.checkpoint_dir,
    log_file=args.logfile_path,
    wandb_project_name=args.wandb_project_name,
    wandb_entity=args.wandb_entity,
    wandb_run_name=args.wandb_run_name,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=args.device
)

trainer.train(
    train_dataloader=train_dl,
    val_dataloader=valid_dl,
    num_epochs=args.num_epochs,
    eval_every=args.eval_every,
    save_every=args.save_every,
    grad_accumulation_steps=args.grad_accumulation_steps,
)