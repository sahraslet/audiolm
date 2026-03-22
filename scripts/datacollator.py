import torch

class AudioLMCollator:
    def __init__(self, text_pad_token_id: int, n_codebooks: int = 8, max_length: int = 1024):
        self.text_pad_token_id = text_pad_token_id
        self.n_codebooks = n_codebooks
        self.max_length = max_length

    def __call__(self, samples: list[dict]) -> dict:
        # truncate long sequences (if any) to max_length
        truncated = []
        for s in samples:
            T = s["text_ids"].size(0)
            if T > self.max_length:
                entry = {"text_ids": s["text_ids"][:self.max_length],
                         "attention_mask": s["attention_mask"][:self.max_length]}
                if "audio_codes" in s:
                    entry["audio_codes"] = s["audio_codes"][:, :self.max_length]
                truncated.append(entry)
            else:
                truncated.append(s)
        samples = truncated

        max_T    = max(s["text_ids"].size(0) for s in samples)
        has_audio = "audio_codes" in samples[0]
        K        = self.n_codebooks

        text_ids_batch, audio_codes_batch, attention_mask_batch = [], [], []

        for s in samples:
            T       = s["text_ids"].size(0)
            pad_len = max_T - T

            text_ids_batch.append(torch.cat([
                s["text_ids"],
                torch.full((pad_len,), self.text_pad_token_id, dtype=torch.long)
            ]))

            if has_audio:
                audio_codes_batch.append(torch.cat([
                    s["audio_codes"],
                    torch.full((K, pad_len), -1, dtype=torch.long)
                ], dim=1))
            else:
                audio_codes_batch.append(torch.full((K, max_T), -1, dtype=torch.long))

            attention_mask_batch.append(torch.cat([
                s["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long)
            ]))

        return {
            "text_ids":       torch.stack(text_ids_batch),
            "audio_codes":    torch.stack(audio_codes_batch),
            "attention_mask": torch.stack(attention_mask_batch),
        }
