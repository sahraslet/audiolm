import math
import torch
from torch import Tensor 
import torch.nn.functional as F 
from contextlib import nullcontext
from sacrebleu import BLEU
from transformers import PreTrainedTokenizer
from evaluate import load
import torch.nn as nn




def attention(
        query: Tensor, 
        key: Tensor, 
        value: Tensor, 
        scale: float | None = None, 
        dropout: float = 0.0,
        mask : Tensor | None = None
) -> tuple[Tensor, Tensor]:
    

    if scale is None:
        scale = math.sqrt(key.size(-1))

    scores = (query @ key.transpose(-2, -1)) / scale

    if mask is not None:
        scores = scores.masked_fill(mask==0, float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    attention_weights = F.dropout(attention_weights, p=dropout)
    attention = attention_weights @ value 

    return attention, attention_weights


def split_heads(x: Tensor, n_heads: int) -> Tensor:
    """Split tensor into n_heads tensors for individual attention heads

    Args:
        x (Tensor): tensor to be split
        n_heads (int): number of heads

    Returns:
        Tensor: A different view of the tensor of form [Batch n_heads Sequence d_head]
    """
    # [batch sequence d_model]
    B, S, D = x.shape
    d_head = D // n_heads
    # [batch sequence n_heads d_head] => [batch n_heads sequence d_head]
    return x.view(B, S, n_heads, d_head).transpose(1, 2)

def merge_heads(x: Tensor) -> Tensor:
    """Merge tensor after after attention calculation

    Args:
        x (Tensor): Tensor [Batch n_heads Sequence d_head] to be merged

    Returns:
        Tensor: Merged tensor of shape [Batch Sequence d_model]
    """
    B, Nh, S, Dh = x.shape
    return x.transpose(1, 2).contiguous().view(B, S, Nh * Dh)
 
def rotate_half(x:Tensor) -> Tensor:
    """ Rotate the tensor by 90° on last dim

    Args:
        x (Tensor): tensor to be rotated

    Returns:
        Tensor: rotated tensor
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(
        q: Tensor, 
        k: Tensor, 
        cos: Tensor, 
        sin: Tensor, 
        unsqueeze_dim: int = 1
) -> tuple[Tensor, Tensor]:
    """Apply RoPE

    Args:
        q (Tensor): query tensor
        k (Tensor): key tensor
        cos (Tensor): cosine part of rope
        sin (Tensor): sine part of rope
        unsqueeze_dim (int, optional): unsqueeze sine and cosine tensors across specific dimension. Defaults to 1.

    Returns:
        tuple[Tensor, Tensor]: query and key embeddings with rope
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
 
def maybe_autocast(
    device_type: str,
    dtype: torch.dtype | None = None,
    enabled: bool = True,
    cache_enabled: bool | None = None,
):
    """
    Context manager that only autocasts if:

    - `autocast` is already enabled in this context
    - Or this call to `maybe_autocast` has `enabled=True`

    This prevents `autocast` being added to the graph when it is effectively a no-op.
    Which makes graph splitting in `torch.compile` more flexible as it removes the
    requirement that partition IDs be monotonically increasing.
    """
    if torch.is_autocast_enabled(device_type) or enabled:
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    else:
        return nullcontext()
 
 
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def compute_bleu(predicted: torch.Tensor, ground_truth: torch.Tensor, tokenizer: PreTrainedTokenizer) -> float:
    """
    Compute BLEU score on translation validation set.
    Returns:
        float: BLEU score
    """
    predictions = tokenizer.batch_decode(predicted, skip_special_tokens=True)
    references = tokenizer.batch_decode(ground_truth, skip_special_tokens=True)

    score = BLEU().corpus_score(predictions, [references])
    return score.score

def compute_wer(predicted: torch.Tensor, ground_truth: torch.Tensor, tokenizer: PreTrainedTokenizer)    -> float:
    """
    Compute WER score on translation validation set.
    Returns:
        float: WER score
    """
    wer_metric = load("wer")

    predictions = tokenizer.batch_decode(predicted, skip_special_tokens=True)
    references = tokenizer.batch_decode(ground_truth, skip_special_tokens=True)

    score = wer_metric.compute(predictions=predictions, references=references)

    return score



def apply_delay_pattern(codes: torch.Tensor, bos_token_id: int, pad_token_id: int) -> torch.Tensor:
    """
    Apply MusicGen-style delay pattern. Codebook k is shifted right by k positions.
    Input:  [n_codebooks, seq_len]
    Output: [n_codebooks, seq_len + n_codebooks - 1]
    """
    n_codebooks = codes.shape[0]
    rows = []
    for k in range(n_codebooks):
        bos = torch.full((k,), bos_token_id, dtype=torch.long, device=codes.device)
        pad = torch.full((n_codebooks - k - 1,), pad_token_id, dtype=torch.long, device=codes.device)
        rows.append(torch.cat([bos, codes[k], pad]))
    return torch.stack(rows)

def deinterleave_audio_tokens(interleaved_audio, bos_token_id, pad_token_id):
    """
    Deinterleave audio tokens into separate codebooks. interleaved_audio has shape [n_codebooks, n_tokens_per_codebook + n_codebooks - 1]
    """
    n_codebooks = interleaved_audio.shape[0]
    deinterleaved = []

    for codebook in range(n_codebooks):
        tokens = interleaved_audio[codebook]
        mask = (tokens != bos_token_id) & (tokens != pad_token_id)
        deinterleaved.append(tokens[mask])

    return torch.stack(deinterleaved)


def build_audio_mask(token_ids: torch.Tensor, audio_token_id: int) -> torch.Tensor:
    """
    Build a boolean mask that is True wherever token_ids == audio_token_id.

    Args:
        token_ids: (B, T)
    Returns:
        (B, T) bool
    """
    return token_ids == audio_token_id


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Standard causal (lower-triangular) attention mask.

    Returns:
        (1, 1, seq_len, seq_len) float mask with 0 / -inf
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)


# ===========================================================================
# Loss
# ===========================================================================

def audio_lm_loss(
        logits_audio: torch.Tensor | None,  # (B, K, T, audio_vocab_size)
        logits_text: torch.Tensor,  # (B, T, text_vocab_size)
        audio_labels: torch.Tensor | None,  # (B, K, T)  values in [0, audio_vocab_size) or -100
        text_labels: torch.Tensor,  # (B, T)     values in [0, text_vocab_size)  or -100
        n_codebooks: int,
        audio_vocab_size: int,
        text_loss_weight: float = 1.0,
        audio_loss_weight: float = 1.0,
) -> torch.Tensor:
    """
    Combined cross-entropy loss for text + K audio codebooks.

    Shape contract:
        logits_audio:  (B, K, T, audio_vocab_size)   ← K zuerst
        audio_labels:  (B, K, T)                      ← K zuerst, -100 = ignore
        logits_text:   (B, T, text_vocab_size)
        text_labels:   (B, T)                          ← -100 = ignore

    Audio-Labels müssen NICHT offset-kodiert sein — sie sind raw [0, audio_vocab_size).
    apply_audio_offset() wird nur für das interleaved flat-token-stream Format benötigt,
    nicht für separate Codebook-Heads.
    """
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    # --- Text Loss ---
    # logits_text: (B, T, V) → (B*T, V);  text_labels: (B*T,)
    loss = text_loss_weight * ce(
        logits_text.reshape(-1, logits_text.size(-1)),
        text_labels.reshape(-1),
    )

    # --- Audio Loss (K codebooks) ---
    if logits_audio is not None and audio_labels is not None:
        # logits_audio: (B, K, T, audio_vocab_size)
        # audio_labels: (B, K, T)
        assert logits_audio.shape[1] == n_codebooks, (
            f"logits_audio.shape[1]={logits_audio.shape[1]} != n_codebooks={n_codebooks}"
        )
        assert audio_labels.shape[1] == n_codebooks, (
            f"audio_labels.shape[1]={audio_labels.shape[1]} != n_codebooks={n_codebooks}"
        )

        audio_loss = torch.tensor(0.0, device=logits_audio.device)
        for k in range(n_codebooks):
            logits_k = logits_audio[:, k, :, :]  # (B, T, audio_vocab_size)
            labels_k = audio_labels[:, k, :]  # (B, T)

            # Sicherstellen dass nur valide Labels in [0, audio_vocab_size) sind
            # -100 wird von CrossEntropyLoss ignoriert
            audio_loss += ce(
                logits_k.reshape(-1, audio_vocab_size),
                labels_k.reshape(-1),
            )

        loss = loss + audio_loss_weight * (audio_loss / n_codebooks)

    return loss
