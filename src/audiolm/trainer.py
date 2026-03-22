
import wandb
import logging
import argparse
from tqdm import tqdm
import dataclasses
import torch
import torch.nn as nn


class Trainer:
    def __init__(
            self,
            config,
            checkpoint_dir: str,
            log_file: str,
            wandb_project_name: str,
            wandb_entity: str,
            wandb_run_name: str,
            model: nn.Module,
            loss_fn,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
            # compile: bool = True,
            device: str | None = "cuda",
    ) -> None:
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compile = compile
        self.checkpoint_dir = checkpoint_dir
        self.config = config
        self.epoch: int = 0
        self.global_step: int = 0

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
        formatter = logging.Formatter("{levelname:<8} {message}", style="{")
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        # if self.compile:
        #     self.model = torch.compile(self.model)
        #     self.logger.info("Model is compiled!")

        wandb.init(
            entity=wandb_entity,
            project=wandb_project_name,
            name=wandb_run_name,
            config = dataclasses.asdict(config)
        )

    def _prepare_batch(self, batch):
        """
        Shift inputs/labels und baue alle Masken auf.

        Returns:
            text_inputs:    (B, T-1)
            text_labels:    (B, T-1)     — -100 an Padding- und Audio-Positionen
            audio_inputs:   (B, K, T-1)
            audio_labels:   (B, K, T-1)  — -100 an Positionen ohne Audio
            audio_mask:     (B, T-1) bool
            attn_mask:      (B, 1, T-1, T-1) causales Masking
        """
        text_ids = batch['text_ids'].to(self.device)  # (B, T)
        audio_codes = batch['audio_codes'].to(self.device)  # (B, K, T)
        pad_mask = batch['attention_mask'].to(self.device)  # (B, T) — 1=valide

        # --- Shift by 1 ---
        text_inputs = text_ids[:, :-1]  # (B, T-1)
        text_labels = text_ids[:, 1:]  # (B, T-1)
        audio_inputs = audio_codes[:, :, :-1]  # (B, K, T-1)
        audio_labels = audio_codes[:, :, 1:]  # (B, K, T-1)
        pad_mask = pad_mask[:, :-1]  # (B, T-1)

        T = text_inputs.size(1)

        # --- Text-Labels maskieren ---
        # Positionen wo pad_mask==0 → ignorieren
        text_labels = text_labels.masked_fill(pad_mask == 0, -100)
        # Audio-Positionen (AUDIO_TOKEN_ID) im Text-Stream → nicht als Text-Loss rechnen
        audio_pos = text_inputs == self.config.audio_token_id
        text_labels = text_labels.masked_fill(audio_pos, -100)

        # --- Audio-Labels maskieren ---
        # -1 in audio_codes bedeutet ∅ (Delay-Padding) → ignorieren
        audio_labels = audio_labels.masked_fill(audio_labels < 0, -100)
        # Nicht-Audio-Positionen → ignorieren
        # (An Text-Positionen sind audio_codes sowieso -1, aber sicherheitshalber)
        text_pos = ~audio_pos  # (B, T-1)
        # Für alle K Codebooks: Text-Positionen ignorieren
        audio_labels = audio_labels.masked_fill(
            text_pos.unsqueeze(1).expand_as(audio_labels), -100
        )

        # --- Audio-Mask ---
        audio_mask = audio_pos  # (B, T-1) bool

        # --- Kausales Attention-Mask ---
        # Kombiniert: kausales Dreieck + Padding-Mask
        causal = torch.tril(torch.ones(T, T, device=self.device))  # (T, T)
        # Padding-Mask auf Keys anwenden: (B, 1, 1, T) * (1, 1, T, T)
        attn_mask = causal.unsqueeze(0).unsqueeze(0) * pad_mask.unsqueeze(1).unsqueeze(2)
        # Shape: (B, 1, T, T)

        return text_inputs, text_labels, audio_inputs, audio_labels, audio_mask, attn_mask

    def _common_step(self, batch) -> torch.Tensor:
        (
            text_inputs,
            text_labels,
            audio_inputs,
            audio_labels,
            audio_mask,
            attn_mask,
        ) = self._prepare_batch(batch)

        logits_audio, logits_text = self.model(
            token_ids=text_inputs,
            audio_codes=audio_inputs,
            audio_mask=audio_mask,
            attention_mask=attn_mask,
        )

        loss = self.loss_fn(
            logits_audio=logits_audio,
            logits_text=logits_text,
            audio_labels=audio_labels,
            text_labels=text_labels,
        )
        return loss


    def evaluate(self, val_dataloader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                loss = self._common_step(batch)
                total_loss += loss.item()

        return total_loss / len(val_dataloader)


    def train(
            self, 
            train_dataloader, 
            val_dataloader = None,
            num_epochs: int = 1,
            eval_every: int = 5000,
            save_every: int = 5000,
            grad_accumulation_steps: int = 1,
            grad_clip_max_norm: float = 1.0
    ) -> None:
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            total_loss = 0.0
            tokens_seen = 0
            epoch_steps = 0
            val_loss = None

            with tqdm(total=len(train_dataloader), desc=f"Training Epoch {self.epoch}", dynamic_ncols=True) as pbar:
                for idx, batch in enumerate(train_dataloader):

                    # --------- Training --------
                    # tokens_seen += (batch['input_ids'].size(0) * batch['input_ids'].size(1))
                    tokens_seen += batch['text_ids'].numel() + batch['audio_codes'].numel() # store no. of tokens model has seen in each batch
                    self.model.train()
                    
                    loss = self._common_step(batch)
                    loss_scaled = loss / grad_accumulation_steps
                    loss_scaled.backward()
                    

                    if ((idx + 1) % grad_accumulation_steps == 0) or (idx + 1 == len(train_dataloader)):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=grad_clip_max_norm
                        )
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scheduler:
                            self.scheduler.step()

                    self.global_step += 1
                    epoch_steps += 1
                    total_loss += loss_scaled.item()

                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(f"Train epoch: {self.epoch} step: {self.global_step} Tokens seen: {tokens_seen} Train Loss: {loss.item()}")
                    wandb.log({'train_loss': loss.item(), 'tokens_seen': tokens_seen, 'lr': current_lr, 'epoch': self.epoch, 'step': self.global_step}, step=self.global_step)
                
                    # --------- evaluation ---------
                    if self.global_step % eval_every == 0 and val_dataloader and self.global_step > 0 or epoch_steps == len(train_dataloader):
                        val_loss = self.evaluate(val_dataloader)
                        self.logger.info(f"Eval Step: {self.global_step} Tokens seen: {tokens_seen} Val Loss : {val_loss}")
                        wandb.log({'val_loss': val_loss, 'tokens_seen': tokens_seen, 'epoch': self.epoch, 'step': self.global_step}, step=self.global_step)

                    # --------- checkpointing ---------
                    if self.global_step % save_every == 0:
                        self.save_checkpoint()
                        self.logger.info(f"Checkpoint Saved | Train loss: {loss} | Eval loss: {val_loss}")

                    pbar.set_description(f"Train Loss step : {loss.item():.4f} | Val Loss : {val_loss}")
                    pbar.update(1)
                self.logger.info(f"Epoch {self.epoch} ended with train loss {total_loss / epoch_steps} validation loss {val_loss}")


    def predict(self, batch):
        raise NotImplementedError

    def save_checkpoint(self, push_to_hub: bool = False, repo_id: str = None):
        save_path = f"{self.checkpoint_dir}/checkpoint_epoch-{self.epoch}-step-{self.global_step}.pth"
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }, save_path)

        if push_to_hub and repo_id:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=save_path,
                path_in_repo=f"checkpoints/checkpoint_step{self.global_step}.pth",
                repo_id=repo_id,
                repo_type="model",
            )
            self.logger.info(f"Checkpoint pushed to {repo_id}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler and ckpt["scheduler"]:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.logger.info(f"Resumed from {path} @ epoch {self.epoch} step {self.global_step}")