'''AudioLM class with multiple LM-heads for multiple codebooks'''

from .qwen import QwenCausalLM
import torch

class AudioLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = QwenCausalLM(config=config)
        self.lm_heads = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=config.d_model, out_features=config.audio_vocab_size, bias=False) for _ in range(config.n_codebooks)]
        )

    def forward(self, text_ids, audio_ids=None, attention_mask=None):
        text_embeds = self.model.model.embed_tokens(text_ids)

        if audio_ids is not None:
            audio_embeds = sum(
                self.model.model.embed_tokens(audio_ids[k,:, :])
                for k in range(self.config.n_codebooks)
            )
            inputs_embeds = torch.cat([text_embeds, audio_embeds], dim=1)
        else:
            inputs_embeds = text_embeds

        x = self.model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        logits_text = self.model.lm_head(x)

        if audio_ids is not None:
            logits_audio = torch.stack([head(x) for head in self.lm_heads], dim=1)
        else:
            logits_audio = None

        return logits_audio, logits_text


