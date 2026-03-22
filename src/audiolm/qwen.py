from .layers import QwenRoPE, QwenDecoderLayer, QwenRMSNorm
from .config import QwenConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


class QwenModel(nn.Module):
    def __init__(self, config: QwenConfig):
        """Qwen Model

        Args:
            config (QwenConfig): Model Config
        """
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(config=config, layer_idx=layer_idx) for layer_idx in range(config.n_layers)]
        )
        self.norm = QwenRMSNorm(hidden_size=config.d_model, eps=config.rmsnorm_eps)
        self.rotary_emb = QwenRoPE(config=config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:

        if inputs_embeds is not None:
            x= inputs_embeds
        else:
            x = self.embed_tokens(input_ids)

        B, S = x.shape[:2]
        position_ids = torch.arange(S, dtype = torch.long).unsqueeze(0).expand(B, -1) # all position ids start from 0
        position_embeds = self.rotary_emb(x, position_ids)
        for layer in self.layers:
            x = layer(x, attention_mask, position_embeds)
        x = self.norm(x)
        return x
    
class QwenCausalLM(nn.Module):
    def __init__(self, config: QwenConfig):
        """Qwen2.5 Causal LM

        Args:
            config (QwenConfig): Model Config
        """
        super().__init__()
        self.config = config
        self.model = QwenModel(config=config)
        self.lm_head = nn.Linear(in_features=config.d_model, out_features=config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        # print(f"NUMBER OF PARAMETERS IN QWEN 2.5 0.5B IS {self.count_params()} i.e. {self.count_params() / 1e6}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None)->torch.Tensor:
        x = self.model(input_ids, attention_mask)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None
    ) -> torch.Tensor:
        """
        Autoregressively generates tokens given a prompt.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with prompt token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature. Lower = more deterministic.
            top_k: Only sample from the top_k logits.
            eos_token_id: Optional token ID at which to stop generation.

        Returns:
            Tensor of shape (batch_size, seq_len + max_new_tokens) with generated sequences.
        """
        self.eval()
        B, S = input_ids.shape

        for _ in range(max_new_tokens):
            attention_mask = torch.ones((B, 1, 1, S), device=input_ids.device)  # assumes all tokens are valid
            logits = self.forward(input_ids, attention_mask)  # (batch, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]  # take the last token's logits

            # Apply temperature
            next_token_logits = next_token_logits / max(temperature, 1e-8) # avoid division by 0 

            # Top-k filtering
            if top_k is not None:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k,
                    torch.full_like(next_token_logits, -float('Inf')),
                    next_token_logits
                )

            # Sample the next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # Stop if all sequences have generated eos_token_id
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break

        return input_ids