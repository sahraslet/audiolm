from dataclasses import dataclass

@dataclass
class QwenConfig:
    block_size: int
    d_model: int
    d_ffn: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    max_positional_embed: int
    rmsnorm_eps: float
    rope_theta: float
    dropout: float
    vocab_size: int          # muss == text_vocab_size sein (Qwen embed_tokens + special tokens)
    text_vocab_size: int     # ← neu: für den assert in AudioLM und Loss
    activation: str
    pad_token_id: int
    audio_token_id: int      # ← neu: ID von <|audio|> token
    tie_word_embeddings: bool
    audio_vocab_size: int
    n_codebooks: int