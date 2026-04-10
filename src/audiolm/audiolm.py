import torch
import torch.nn as nn
from .qwen import QwenCausalLM


class AudioLM(nn.Module):
    """
    Fused Audio+Text LM auf Basis von Qwen.

    Architektur:
    - Qwen lm_head bleibt für Text-Prediction (weight-tied mit embed_tokens)
    - Fused Audio-Head: Linear(d_model, K * audio_vocab_size)
      → wird in forward() in (B, K, T, audio_vocab_size) reshapet
    - Embedding: Summe über K Codebook-Embeddings (MusicGen-Stil)
    - audio_codes überall: (B, K, T), raw [0, audio_vocab_size), -1 für ∅
    - logits_audio: (B, K, T, audio_vocab_size)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = QwenCausalLM(config=config)

        assert self.model.lm_head.out_features == config.vocab_size, (
            f"QwenCausalLM.lm_head.out_features ({self.model.lm_head.out_features}) "
            f"!= config.text_vocab_size ({config.text_vocab_size}). "
            "Stelle sicher dass config.vocab_size == config.text_vocab_size."
        )

        # K Codebook-Embedding-Tabellen (getrennt von Qwens embed_tokens)
        self.audio_embed = nn.ModuleList([
            nn.Embedding(config.audio_vocab_size, config.d_model)
            for _ in range(config.n_codebooks)
        ])

        # Gelerntes Embedding für ∅ Delay-Padding-Positionen (pro Codebook)
        self.audio_pad_embed = nn.Parameter(
            torch.zeros(config.n_codebooks, config.d_model)
        )

        # Fused Audio-Head: ein Matrix-Multiply für alle K Codebooks
        self.audio_lm_head = nn.Linear(
            config.d_model,
            config.n_codebooks * config.audio_vocab_size,
            bias=False,
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(
        self,
        token_ids: torch.Tensor,    # (B, T)
        audio_codes: torch.Tensor,  # (B, K, T) — raw [0, audio_vocab_size), -1 für ∅
        audio_mask: torch.Tensor,   # (B, T) bool
    ) -> torch.Tensor:              # (B, T, d_model)
        """
        Text-Positionen:  Qwen embed_tokens
        Audio-Positionen: Summe der K Codebook-Embeddings
        """
        embeds = self.model.model.embed_tokens(token_ids)  # (B, T, d)

        if audio_mask.any():
            audio_emb = torch.zeros_like(embeds)

            for k in range(self.config.n_codebooks):
                codes_k = audio_codes[:, k, :]          # (B, T)
                is_pad  = codes_k < 0                    # ∅-Positionen
                safe    = codes_k.clamp(min=0)

                looked_up = self.audio_embed[k](safe)        # (B, T, d)
                looked_up[is_pad] = self.audio_pad_embed[k]  # ∅ → gelerntes Pad-Embed

                audio_emb += looked_up

            embeds[audio_mask] = audio_emb[audio_mask]

        return embeds  # (B, T, d)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,                    # (B, T)
        audio_codes: torch.Tensor | None = None,    # (B, K, T), -1 für ∅
        audio_mask: torch.Tensor | None = None,     # (B, T) bool
        attention_mask: torch.Tensor | None = None, # (B, 1, T, T)
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Returns:
            logits_audio: (B, K, T, audio_vocab_size) oder None
            logits_text:  (B, T, text_vocab_size)
        """
        # --- Embeddings ---
        if audio_codes is not None and audio_mask is not None:
            inputs_embeds = self.embed(token_ids, audio_codes, audio_mask)
        else:
            inputs_embeds = self.model.model.embed_tokens(token_ids)

        # --- Backbone ---
        hidden_states = self.model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )  # (B, T, d_model)

        B, T, _ = hidden_states.shape

        # --- Text-Logits via Qwen lm_head (weight-tied) ---
        logits_text = self.model.lm_head(hidden_states)  # (B, T, text_vocab_size)

        # --- Audio-Logits via Fused Head ---
        logits_audio = None
        if audio_codes is not None:
            fused = self.audio_lm_head(hidden_states)  # (B, T, K * audio_vocab_size)
            logits_audio = fused.view(
                B, T,
                self.config.n_codebooks,
                self.config.audio_vocab_size,
            ).permute(0, 2, 1, 3)  # (B, K, T, audio_vocab_size)

        return logits_audio, logits_text

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_audio(
        self,
        prompt_token_ids: torch.Tensor,         # (B, T_prompt)
        max_audio_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_audio_token_id: int | None = None,
    ) -> torch.Tensor:                          # (B, K, T_gen)
        """
        Autoregressives TTS: generiert Audio-Codes gegeben Text-Prompt.
        Gibt raw Codes [0, audio_vocab_size) zurück — ohne Delay-Padding.
        """
        self.eval()
        B      = prompt_token_ids.size(0)
        K      = self.config.n_codebooks
        device = prompt_token_ids.device

        current_ids = prompt_token_ids  # (B, T)
        generated   = [[] for _ in range(K)]  # K Listen mit (B, 1) Tensoren
        audio_codes = None
        audio_mask  = None

        for step in range(max_audio_tokens):
            T = current_ids.size(1)
            attn_mask = torch.ones(B, 1, 1, T, device=device)

            logits_audio, _ = self.forward(
                token_ids=current_ids,
                audio_codes=audio_codes,
                audio_mask=audio_mask,
                attention_mask=attn_mask,
            )

            if logits_audio is None:
                break

            # Letzter Timestep, alle K Codebooks: (B, K, audio_vocab)
            next_logits = logits_audio[:, :, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                top_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold   = top_vals[..., -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float('-inf'))

            probs      = torch.softmax(next_logits, dim=-1)
            next_codes = torch.stack(
                [torch.multinomial(probs[:, k, :], 1) for k in range(K)],
                dim=1,
            )  # (B, K, 1)

            for k in range(K):
                generated[k].append(next_codes[:, k, :])  # (B, 1)

            # EOS-Check auf Codebook 0
            if eos_audio_token_id is not None:
                if (next_codes[:, 0, 0] == eos_audio_token_id).all():
                    break

            # Kontext für nächsten Step aufbauen
            T_prompt = prompt_token_ids.size(1)
            n_audio  = step + 1

            # Alle bisher generierten Codes: (B, K, n_audio)
            all_codes = torch.cat(
                [torch.cat(generated[k], dim=-1).unsqueeze(1) for k in range(K)],
                dim=1,
            )  # (B, K, n_audio)

            # token_ids: Prompt + Audio-Platzhalter
            audio_placeholder = torch.full(
                (B, n_audio), self.config.audio_token_id, dtype=torch.long, device=device
            )
            current_ids = torch.cat([prompt_token_ids, audio_placeholder], dim=1)

            # audio_codes: -1 an Prompt-Positionen, echte Codes an Audio-Positionen
            audio_codes = torch.full(
                (B, K, current_ids.size(1)), -1, dtype=torch.long, device=device
            )
            audio_codes[:, :, T_prompt:] = all_codes

            # audio_mask: True an Audio-Positionen
            audio_mask = torch.zeros(B, current_ids.size(1), dtype=torch.bool, device=device)
            audio_mask[:, T_prompt:] = True

        # (B, K, T_gen)
        if generated[0]:
            return torch.cat(
                [torch.cat(generated[k], dim=-1).unsqueeze(1) for k in range(K)],
                dim=1,
            )
        return torch.zeros(B, K, 0, dtype=torch.long, device=device)


