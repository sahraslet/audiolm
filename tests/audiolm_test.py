"""
Smoke-Test für AudioLM — alles in einer Datei, läuft lokal in PyCharm.

Tests:
  1. Forward Pass — Shapes korrekt
  2. Loss-Berechnung — kein NaN/Inf
  3. Overfit auf 1 Batch — Loss muss fallen
  4. TTS + STT gemischter Batch — Forward Pass stabil
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functools import partial


# ===========================================================================
# Dummy-Batch
# ===========================================================================

def make_dummy_batch(
    batch_size:      int = 2,
    seq_len:         int = 64,
    n_codebooks:     int = 8,
    text_vocab_size: int = 151939,
    audio_vocab_size: int = 1024,
    audio_token_id:  int = 151936,
    pad_token_id:    int = 151643,
    device:          str = "cpu",
) -> dict:
    """
    Synthetischer Batch der das reale Preprocessing-Format nachbildet.
    Erste Hälfte TTS (Audio am Ende), zweite Hälfte STT (Audio am Anfang).

    audio_codes: raw [0, audio_vocab_size) — kein Offset, -1 für ∅
    """
    B = batch_size
    K = n_codebooks
    T = seq_len

    text_ids    = torch.randint(0, text_vocab_size - 10, (B, T), dtype=torch.long)
    audio_codes = torch.full((B, K, T), -1, dtype=torch.long)

    half = B // 2 if B >= 2 else 1

    # TTS: Audio in zweiter Hälfte der Sequenz
    tts_start = T // 2
    text_ids[:half, tts_start:] = audio_token_id
    for k in range(K):
        audio_codes[:half, k, tts_start:] = torch.randint(
            0, audio_vocab_size, (half, T - tts_start)
        )

    # STT: Audio in erster Hälfte der Sequenz
    stt_end = T // 3
    text_ids[half:, :stt_end] = audio_token_id
    for k in range(K):
        audio_codes[half:, k, :stt_end] = torch.randint(
            0, audio_vocab_size, (B - half, stt_end)
        )

    return {
        "text_ids":       text_ids.to(device),
        "audio_codes":    audio_codes.to(device),
        "attention_mask": torch.ones(B, T, dtype=torch.long, device=device),
    }


# ===========================================================================
# Dataset + Collator (für echtes Dataset später)
# ===========================================================================

class AudioLMDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "text_ids":       torch.tensor(sample["text_ids"],       dtype=torch.long),
            "audio_codes":    torch.tensor(sample["audio_codes"],    dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
        }


class AudioLMCollator:
    def __init__(self, text_pad_token_id: int):
        self.text_pad_token_id = text_pad_token_id

    def __call__(self, samples: list[dict]) -> dict:
        max_T = max(s["text_ids"].size(0) for s in samples)
        K     = samples[0]["audio_codes"].size(0)

        text_ids_batch, audio_codes_batch, attention_mask_batch = [], [], []

        for s in samples:
            T       = s["text_ids"].size(0)
            pad_len = max_T - T

            text_ids_batch.append(torch.cat([
                s["text_ids"],
                torch.full((pad_len,), self.text_pad_token_id, dtype=torch.long)
            ]))
            audio_codes_batch.append(torch.cat([
                s["audio_codes"],
                torch.full((K, pad_len), -1, dtype=torch.long)
            ], dim=1))
            attention_mask_batch.append(torch.cat([
                s["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long)
            ]))

        return {
            "text_ids":       torch.stack(text_ids_batch),
            "audio_codes":    torch.stack(audio_codes_batch),
            "attention_mask": torch.stack(attention_mask_batch),
        }


# ===========================================================================
# Smoke-Test
# ===========================================================================

def run_smoke_test(model, config, device="cpu"):
    """
    Args:
        model:   AudioLM Instanz
        config:  QwenConfig mit n_codebooks, audio_vocab_size,
                 text_vocab_size, audio_token_id, pad_token_id
        device:  "cpu" oder "cuda"
    """
    from audiolm.functional import audio_lm_loss

    model = model.to(device)

    loss_fn = partial(
        audio_lm_loss,
        n_codebooks=config.n_codebooks,
        audio_vocab_size=config.audio_vocab_size,
    )

    print("=" * 60)
    print("SMOKE TEST — AudioLM")
    print("=" * 60)

    # Batch erstellen
    batch = make_dummy_batch(
        batch_size=2,
        seq_len=64,
        n_codebooks=config.n_codebooks,
        text_vocab_size=config.text_vocab_size,
        audio_vocab_size=config.audio_vocab_size,
        audio_token_id=config.audio_token_id,
        pad_token_id=config.pad_token_id,
        device=device,
    )

    # Shift by 1
    text_inputs  = batch["text_ids"][:, :-1]         # (B, T-1)
    text_labels  = batch["text_ids"][:, 1:]          # (B, T-1)
    audio_inputs = batch["audio_codes"][:, :, :-1]   # (B, K, T-1)
    audio_labels = batch["audio_codes"][:, :, 1:]    # (B, K, T-1)
    pad_mask     = batch["attention_mask"][:, :-1]    # (B, T-1)

    B, T = text_inputs.shape
    K    = audio_inputs.shape[1]

    # Masken
    audio_mask = text_inputs == config.audio_token_id  # (B, T) bool
    causal     = torch.tril(torch.ones(T, T, device=device))
    attn_mask  = causal.unsqueeze(0).unsqueeze(0) * pad_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, T, T)

    # Text-Labels maskieren
    text_labels = text_labels.masked_fill(pad_mask == 0, -100)
    text_labels = text_labels.masked_fill(audio_mask, -100)

    # Audio-Labels maskieren: -1 → -100
    audio_labels = audio_labels.masked_fill(audio_labels < 0, -100)
    audio_labels = audio_labels.masked_fill(
        (~audio_mask).unsqueeze(1).expand_as(audio_labels), -100
    )

    # ------------------------------------------------------------------
    # TEST 1: Forward Pass — Shape Check
    # ------------------------------------------------------------------
    print("\n[1/4] Forward Pass — Shape Check")
    model.eval()
    with torch.no_grad():
        logits_audio, logits_text = model(
            token_ids=text_inputs,
            audio_codes=audio_inputs,
            audio_mask=audio_mask,
            attention_mask=attn_mask,
        )

    assert logits_text.shape  == (B, T, config.text_vocab_size), \
        f"logits_text shape falsch: {logits_text.shape}"
    assert logits_audio.shape == (B, K, T, config.audio_vocab_size), \
        f"logits_audio shape falsch: {logits_audio.shape}"

    print(f"  logits_text:  {tuple(logits_text.shape)}  ✓")
    print(f"  logits_audio: {tuple(logits_audio.shape)} ✓")

    # ------------------------------------------------------------------
    # TEST 2: Loss Berechnung — kein NaN/Inf
    # ------------------------------------------------------------------
    print("\n[2/4] Loss Berechnung")
    with torch.no_grad():
        loss = loss_fn(
            logits_audio=logits_audio,
            logits_text=logits_text,
            audio_labels=audio_labels,
            text_labels=text_labels,
        )

    assert not torch.isnan(loss), "Loss ist NaN!"
    assert not torch.isinf(loss), "Loss ist Inf!"
    print(f"  Loss: {loss.item():.4f} ✓")

    # ------------------------------------------------------------------
    # TEST 3: Overfit auf 1 Batch — Loss muss fallen
    # ------------------------------------------------------------------
    print("\n[3/4] Overfit Test — Loss soll fallen")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []

    for step in range(20):
        optimizer.zero_grad()
        logits_audio, logits_text = model(
            token_ids=text_inputs,
            audio_codes=audio_inputs,
            audio_mask=audio_mask,
            attention_mask=attn_mask,
        )
        loss = loss_fn(
            logits_audio=logits_audio,
            logits_text=logits_text,
            audio_labels=audio_labels,
            text_labels=text_labels,
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1:2d}: loss = {loss.item():.4f}")

    assert losses[-1] < losses[0], \
        f"Loss ist nicht gefallen! {losses[0]:.4f} → {losses[-1]:.4f}"
    print(f"  Loss gefallen: {losses[0]:.4f} → {losses[-1]:.4f} ✓")

    # ------------------------------------------------------------------
    # TEST 4: TTS + STT gemischter Batch
    # ------------------------------------------------------------------
    print("\n[4/4] TTS + STT — gemischter Batch Forward Pass")
    model.eval()
    with torch.no_grad():
        logits_audio, logits_text = model(
            token_ids=text_inputs,
            audio_codes=audio_inputs,
            audio_mask=audio_mask,
            attention_mask=attn_mask,
        )
    assert logits_audio is not None
    print(f"  Gemischter Batch Forward Pass ✓")

    print("\n" + "=" * 60)
    print("ALLE TESTS BESTANDEN ✓")
    print("=" * 60)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    from audiolm.audiolm import AudioLM
    from audiolm.config import QwenConfig

    # Mini-Config für schnellen lokalen Test
    cfg = QwenConfig(
        block_size=128,
        d_model=64,
        d_ffn=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_positional_embed=512,
        rmsnorm_eps=1e-06,
        rope_theta=10000.0,
        dropout=0.0,
        vocab_size=151939,
        text_vocab_size=151939,
        activation='silu',
        pad_token_id=151643,
        audio_token_id=151936,
        tie_word_embeddings=True,
        audio_vocab_size=1024,
        n_codebooks=8,
    )

    model = AudioLM(cfg)
    print(f"Parameter: {sum(p.numel() for p in model.parameters()):,}")

    run_smoke_test(model, cfg, device="cpu")