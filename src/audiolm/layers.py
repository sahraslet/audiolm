from .functional import merge_heads, split_heads, attention, apply_rope, maybe_autocast, repeat_kv
from .config import QwenConfig

from collections.abc import Callable
from typing import Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

# TODO create BaseConfig / ModelConfig that each model's config inherits from to get rid of Union[...]

ACT2FN = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
    'gelu': nn.GELU,
}

class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class QwenMLP(nn.Module):
    def __init__(self, config: QwenConfig):
        """
        Implements MLP in Qwen2.5 / Qwen3

        Args:
            config (QwenConfig): Model Config
        """
        super().__init__()
        self.up_proj: nn.Linear = nn.Linear(in_features=config.d_model, out_features=config.d_ffn, bias=False)
        self.gate_proj: nn.Linear = nn.Linear(in_features=config.d_model, out_features=config.d_ffn, bias=False)
        self.down_proj: nn.Linear = nn.Linear(in_features=config.d_ffn, out_features=config.d_model, bias=False)
        self.activation: nn.Module = ACT2FN[config.activation]()

    def forward(self, x: torch.Tensor):
        x, swish = self.up_proj(x), self.activation(self.gate_proj(x))
        down_proj = self.down_proj(x * swish)
        return down_proj
    
class QwenRoPE(nn.Module):
    def __init__(self, config: QwenConfig):
        """
        Implements Rotary Position Embeddings for Qwen2.5 / Qwen3

        Args:
            config (QwenConfig): Model Config
        """
        super().__init__()
        self.base = config.rope_theta
        self.dim = config.d_model // config.n_heads
        inv_freq = 1. / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).to(dtype=torch.float) / self.dim)
        ) 
    
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:

        # [f] -> [1, f, 1] -> [B, f, 1]
        expanded_inv_freq = self.inv_freq[None, :, None].float().to(x.device).expand(position_ids.shape[0], -1, -1) # inv_freq is a buffer and is by default stored on cpu
        expanded_position_ids = position_ids[:, None, :].float().to(x.device) # [B, S] -> [B, 1, S]
        with maybe_autocast(device_type=x.device.type, dtype=torch.float32, enabled=False): # PRECISION-PROBLEM?
            freqs = (expanded_inv_freq.float() @ expanded_position_ids.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(x.dtype), sin.to(x.dtype)
        

class QwenAttention(nn.Module):
    """
    QwenAttention block 
    Uses Grouped Query Attention with config.n_groups groups
    """
    def __init__(self, config: QwenConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups = config.n_heads // config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads # 896 // 14 = 64
        self.q_proj = nn.Linear(in_features=config.d_model, out_features=(config.n_heads * self.head_dim), bias=True) # 896-896
        self.k_proj = nn.Linear(in_features=config.d_model, out_features=(config.n_kv_heads * self.head_dim), bias=True) # 896 - 128
        self.v_proj = nn.Linear(in_features=config.d_model, out_features=(config.n_kv_heads * self.head_dim), bias=True) # 896 - 128
        self.o_proj = nn.Linear(in_features=(config.n_heads * self.head_dim), out_features=config.d_model, bias=False) # 896 - 896
        self.register_buffer(
            "causal_mask", 
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            persistent=False
                )

    def forward(
            self,
            hidden_states: torch.Tensor,
            mask: torch.Tensor,
            position_embeddings: torch.Tensor,
    ):
        B, T, C = hidden_states.size()
        query = split_heads(self.q_proj(hidden_states), n_heads=self.n_heads) # [b, s, 896] -> [b, s, 14, 64] -> [b, 14, s, 64]
        key = split_heads(self.k_proj(hidden_states), n_heads=self.n_kv_heads) # [b, s, 128] -> [b, s, 2, 64] -> [b, 2, s, 64]
        value = split_heads(self.v_proj(hidden_states), n_heads=self.n_kv_heads) # [b, s, 128] -> [b, s, 2, 64] -> [b, 2, s, 64]
        cos, sin = position_embeddings
        query, key = apply_rope(q=query, k=key, cos=cos, sin=sin)
        key = repeat_kv(key, n_rep=self.n_groups)
        value = repeat_kv(value, n_rep=self.n_groups)

        # if mask is not None:
        #     attn_mask = self.causal_mask[:, :, :T, :T] + mask
        # else:
        #     attn_mask = self.causal_mask

        attn_mask = self.causal_mask[:, :, :T, :T]

        attention_scores, _ = attention(query=query, key=key, value=value, mask=attn_mask, scale=(self.head_dim**0.5), dropout=self.config.dropout)
        attention_output = merge_heads(attention_scores)
        attention_output = self.o_proj(attention_output)
        return attention_output

class QwenDecoderLayer(nn.Module):
    def __init__(self, config: QwenConfig, layer_idx: int):
        """
        Qwen2.5 Decoder layer

        Args:
            config (QwenConfig): Model Config
            layer_idx (int): index of layer to be used for caching
        """
        super().__init__()
        self.config = config
        self.self_attn = QwenAttention(config=config, layer_idx=layer_idx)
        self.mlp = QwenMLP(config=config)
        self.input_layernorm = QwenRMSNorm(hidden_size=config.d_model, eps=config.rmsnorm_eps)
        self.post_attention_layernorm = QwenRMSNorm(hidden_size=config.d_model, eps=config.rmsnorm_eps)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None,                       # Would love to use a PrefixLM mask. for now I will make a workaround.
                # position_ids: torch.LongTensor | None = None,         # for now, we initialize it from 0 ... len(seq) - 1
                # past_key_values: Cache | None = None,                 # for now, we dont use a cache
                # use_cache: bool | None = False,                       # same as before
                # cache_position: torch.LongTensor | None = None,       # same as before
                position_embedding: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> torch.Tensor:
        
        # TODO: Experiment with PrefixLM masking (stage 1) and then with block sparse attention
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask, position_embedding)
        x = x + residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        return x