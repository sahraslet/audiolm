"""
Preprocessing audio datasets for STT and TTS tasks.

Output format:
    Dataset({
        features: ['input_features', 'labels', 'input_length', 'language'],
        num_rows: <N>
    })

input_features shape: [Q, T] where
    Q = 8 codebooks (CB1 = semantic, CB2-8 = acoustic)
    T = number of time frames at 50Hz
"""

import torch
import torchaudio
from speechtokenizer import SpeechTokenizer
from transformers import AutoTokenizer
from datasets import load_dataset, Audio
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--text_column", type=str, default="text")
parser.add_argument("--speech_tokenizer_config", type=str, default="fnlp/SpeechTokenizer")
parser.add_argument("--speech_tokenizer_ckpt", type=str, default=None)
parser.add_argument("--text_tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--language", type=str, default=None)
parser.add_argument("--max_duration", type=float, default=30.0)
parser.add_argument("--sampling_rate", type=int, default=16000)
parser.add_argument("--output_dir", type=str, default="./preprocessed/combined")
args = parser.parse_args()


def load_speech_tokenizer(config_path: str, ckpt_path: str | None) -> SpeechTokenizer:
    """ Load SpeechTokenizer from Huggingfacs"""
    if ckpt_path is not None:
        model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    else:
        from huggingface_hub import hf_hub_download

        config_file = hf_hub_download(
            repo_id=config_path,
            filename="speechtokenizer_hubert_avg/config.json"
        )
        ckpt_file = hf_hub_download(
            repo_id=config_path,
            filename="speechtokenizer_hubert_avg/SpeechTokenizer.pt"
        )
        model = SpeechTokenizer.load_from_checkpoint(config_file, ckpt_file)

    model.eval()
    return model


def load_audio(path: str, target_sr: int):
    """Load audio, convert to mono, and resample if needed."""
    waveform, sr = torchaudio.load(path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True) # Convert to mono by averaging channels

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform


def prepare_dataset(batch, model: SpeechTokenizer, text_tokenizer):
    audio = batch["audio"]

    waveform = torch.tensor(audio["array"], dtype=torch.float32)

    # SpeechTokenizer expects [B, C, T] → [1, 1, T]
    waveform = waveform.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        codes = model.encode(waveform)

    batch["input_features"] = codes[:, 0, :].cpu().numpy() # drop batch dimension
    batch["labels"] = text_tokenizer(batch["text"]).input_ids
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    return batch


def filter_audio_length(dataset, duration_threshold):
    return dataset.filter(
        lambda x: x < duration_threshold,
        input_columns=["input_length"]
    )


def make_asr_dataset(args: argparse.Namespace) -> None:
    assert args.sampling_rate == 16000, "SpeechTokenizer requires 16kHz audio."

    model = load_speech_tokenizer(
        args.speech_tokenizer_config,
        args.speech_tokenizer_ckpt
    )

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)

    # Load dataset
    dataset = load_dataset(args.path, args.name, split=args.split)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sampling_rate))

    if args.text_column != "text":
        dataset = dataset.rename_column(args.text_column, "text")

    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, model, text_tokenizer),
        remove_columns=dataset.column_names,
    )

    dataset = filter_audio_length(dataset, args.max_duration)

    dataset = dataset.map(lambda _: {"language": args.language})

    dataset.save_to_disk(args.output_dir)

    print(f"Saved {len(dataset)} samples → {args.output_dir}")
    print(dataset)


if __name__ == "__main__":
    make_asr_dataset(args)
