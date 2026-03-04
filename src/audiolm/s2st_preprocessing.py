"""
Preprocessing CVSS dataset for Speech-to-Speech Translation (S2ST).

CVSS has two audio streams:
    - source audio (e.g. English)
    - target audio (e.g. German)

Output format:
    Dataset({
        features: [
            'input_features',   # [8, T_src] source audio tokens
            'target_features',  # [8, T_tgt] target audio tokens
            'labels',           # text token ids of the translation
            'input_length',     # source audio length in seconds
            'target_length',    # target audio length in seconds
            'language_pair',         # language pair e.g. "en_de"
        ]
    })

"""

import torch
from speechtokenizer import SpeechTokenizer
from transformers import AutoTokenizer
from datasets import load_dataset, Audio
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="HuggingFace dataset path.")
parser.add_argument("--name", type=str, default=None, help="HuggingFace dataset name (subset), e.g. 'en_de'.")
parser.add_argument("--split", type=str, default="train", help="Dataset split to use.")
parser.add_argument("--source_audio_column", type=str, default="audio",
                    help="Column name for source audio.")
parser.add_argument("--target_audio_column", type=str, default="target_audio",
                    help="Column name for target audio.")
parser.add_argument("--translation_column", type=str, default="translation",
                    help="Column name for target text translation.")
parser.add_argument("--speech_tokenizer_config", type=str, default="fnlp/SpeechTokenizer",
                    help="HuggingFace repo or local path for SpeechTokenizer.")
parser.add_argument("--speech_tokenizer_ckpt", type=str, default=None,
                    help="Path to SpeechTokenizer checkpoint (.pt). If None, downloaded from HF hub.")
parser.add_argument("--text_tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--language", type=str, default=None, help="Language pair e.g. 'en_de'.")
parser.add_argument("--max_duration", type=float, default=30.0,
                    help="Maximum audio duration in seconds. Applied to both source and target.")
parser.add_argument("--sampling_rate", type=int, default=16000,
                    help="Target sampling rate. SpeechTokenizer expects 16kHz.")
parser.add_argument("--output_dir", type=str, default="./preprocessed/s2st",
                    help="Output directory to save the preprocessed dataset.")
args = parser.parse_args()


def load_speech_tokenizer(config_path: str, ckpt_path: str | None) -> SpeechTokenizer:
    """Load SpeechTokenizer from HuggingFace hub or local checkpoint."""
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


def encode_audio(audio: dict, model: SpeechTokenizer) -> tuple:
    """
    Encode a single audio sample into SpeechTokenizer codes.

    Args:
        audio: HuggingFace audio dict with keys 'array' and 'sampling_rate'
        model: SpeechTokenizer model

    Returns:
        codes: numpy array of shape [8, T]
        length: audio length in seconds
    """
    waveform = torch.tensor(audio["array"], dtype=torch.float32)

    # SpeechTokenizer expects [B, C, T] → [1, 1, T]
    waveform = waveform.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        # codes shape: [n_q, B, T] = [8, 1, T]
        codes = model.encode(waveform)

    # Drop batch dim → [8, T]
    codes = codes[:, 0, :].cpu().numpy()
    length = len(audio["array"]) / audio["sampling_rate"]

    return codes, length


def prepare_dataset(batch, model: SpeechTokenizer, text_tokenizer,
                    source_col: str, target_col: str, translation_col: str):
    """
    Encode source and target audio with SpeechTokenizer and tokenize translation.

    Args:
        batch: HuggingFace dataset batch
        model: SpeechTokenizer model
        text_tokenizer: text tokenizer for translation labels
        source_col: column name for source audio
        target_col: column name for target audio
        translation_col: column name for target text
    """
    # Encode source audio (e.g. English)
    source_codes, source_length = encode_audio(batch[source_col], model)
    batch["input_features"] = source_codes       # [8, T_src]
    batch["input_length"] = source_length

    # Encode target audio (e.g. German)
    target_codes, target_length = encode_audio(batch[target_col], model)
    batch["target_features"] = target_codes      # [8, T_tgt]
    batch["target_length"] = target_length

    # Tokenize translation text as labels
    batch["labels"] = text_tokenizer(batch[translation_col]).input_ids

    return batch


def filter_audio_length(dataset, duration_threshold):
    """Filter samples where either source or target audio is too long."""
    def both_in_range(input_length, target_length):
        return input_length < duration_threshold and target_length < duration_threshold
    return dataset.filter(both_in_range, input_columns=["input_length", "target_length"])


def make_s2st_dataset(args: argparse.Namespace) -> None:
    assert args.sampling_rate == 16000, "SpeechTokenizer requires 16kHz audio."

    model = load_speech_tokenizer(args.speech_tokenizer_config, args.speech_tokenizer_ckpt)
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)

    dataset = load_dataset(args.path, args.name, split=args.split)

    # Resample both audio columns to 16kHz
    dataset = dataset.cast_column(args.source_audio_column, Audio(sampling_rate=args.sampling_rate))
    dataset = dataset.cast_column(args.target_audio_column, Audio(sampling_rate=args.sampling_rate))

    dataset = dataset.map(
        lambda batch: prepare_dataset(
            batch, model, text_tokenizer,
            source_col=args.source_audio_column,
            target_col=args.target_audio_column,
            translation_col=args.translation_column,
        ),
        remove_columns=dataset.column_names,
    )

    dataset = filter_audio_length(dataset, args.max_duration)
    dataset = dataset.map(lambda _: {"language": args.language})

    dataset.save_to_disk(args.output_dir)

    print(f"Saved {len(dataset)} samples → {args.output_dir}")
    print(dataset)


if __name__ == "__main__":
    make_s2st_dataset(args)