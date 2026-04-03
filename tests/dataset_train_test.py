from datasets import load_dataset
from torch.utils.data import DataLoader
from audiolm.audiolm import AudioLM
from audiolm.config import QwenConfig
from audiolm_test import AudioLMDataset, AudioLMCollator, run_smoke_test
from huggingface_hub import snapshot_download
from datasets import load_from_disk

# Repo lokal downloaden
local_path = snapshot_download(
    repo_id="sahara22/asr_librispeech_subset",
    repo_type="dataset"
)

# Processed Ordner laden
ds = load_from_disk(f"{local_path}/processed")
print(ds)
print(ds["train"][0].keys())
print(f"text_ids length:    {len(ds['train'][0]['text_ids'])}")
print(f"audio_codes shape:  {len(ds['train'][0]['audio_codes'])} x {len(ds['train'][0]['audio_codes'][0])}")
print(f"attention_mask sum: {sum(ds['train'][0]['attention_mask'])}")

# 2. Mini-Config (erst klein testen)
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

# 3. Modell
model = AudioLM(cfg)

# 4. Smoke-Test mit echtem Dataset
run_smoke_test(model, cfg)  # erst Dummy