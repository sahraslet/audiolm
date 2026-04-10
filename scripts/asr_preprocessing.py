"""
Preprocessing audio datasets for ASR (Speech-to-Text) and TTS (Text-to-Speech) tasks.

Follows the same structure as the T2T preprocessing script,
including flip support (TTS direction: text → audio).

Output format:
    Dataset({
        features: ['text_ids', 'audio_ids', 'attention_mask', 'input_length', 'language'],
        num_rows: <N>
    })

    text_ids:  list of text token IDs (task token + text content)
    audio_ids: numpy array of shape [n_codebooks, audio_seq_len] with offset + delay pattern applied

Template:
    STT: text_ids = [task_token, output_sep, text_tokens, eos]
         audio_ids = [n_codebooks, audio_seq_len]

    TTS: text_ids = [task_token, text_tokens, output_sep]
         audio_ids = [n_codebooks, audio_seq_len]  ← target audio
"""

import torch
import argparse
from functools import partial
from speechtokenizer import SpeechTokenizer
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets, Audio
from huggingface_hub import hf_hub_download
from audiolm.functional import apply_delay_pattern


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--text_column", type=str, default="text")
parser.add_argument("--audio_column", type=str, default="audio")
parser.add_argument("--speech_tokenizer_config", type=str, default="fnlp/SpeechTokenizer")
parser.add_argument("--speech_tokenizer_ckpt", type=str, default=None)
parser.add_argument("--text_tokenizer", type=str, default="Qwen/Qwen2.5-0.5B")
parser.add_argument("--language", type=str, default=None)
parser.add_argument("--max_duration", type=float, default=30.0)
parser.add_argument("--sampling_rate", type=int, default=16000)
parser.add_argument("--max_text_length", type=int, default=256)
parser.add_argument("--flip_ratio", type=float, default=0.5)
parser.add_argument("--train_test_ratio", type=float, default=0.05)
parser.add_argument("--num_proc", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="./preprocessed/asr")
parser.add_argument("--text_vocab_size", type=int, required=True)
parser.add_argument("--audio_vocab_size", type=int, default=1024)
parser.add_argument("--n_codebooks", type=int, default=8)
args = parser.parse_args()


def load_speech_tokenizer(config_path: str, ckpt_path: str | None) -> SpeechTokenizer:
    if ckpt_path is not None:
        model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    else:
        config_file = hf_hub_download(repo_id=config_path, filename="speechtokenizer_hubert_avg/config.json")
        ckpt_file = hf_hub_download(repo_id=config_path, filename="speechtokenizer_hubert_avg/SpeechTokenizer.pt")
        model = SpeechTokenizer.load_from_checkpoint(config_file, ckpt_file)
    model.eval()
    return model


def encode_audio(audio: dict, model: SpeechTokenizer) -> tuple:
    """Encode audio into SpeechTokenizer codes. Returns [n_codebooks, T] and length in seconds."""
    waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        codes = model.encode(waveform)  # [n_codebooks, B, T]
    codes = codes[:, 0, :].cpu()       # [n_codebooks, T]
    length = len(audio["array"]) / audio["sampling_rate"]
    return codes, length

def preprocess_dataset(
        dataset: Dataset,
        speech_tokenizer: SpeechTokenizer,
        text_tokenizer: PreTrainedTokenizer,
        task: str,
        src_lang: str,
        tgt_lang: str,
        audio_column: str,
        text_column: str,
        text_vocab_size: int,
        audio_vocab_size: int,
        max_text_length: int,
        num_proc: int,
        flip: bool = False,
) -> Dataset:

    def apply_template(example):
        actual_task = "TTS" if flip else "STT"  # ← diese Zeile hinzufügen
        bos_token_id = text_tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        pad_token_id = text_tokenizer.convert_tokens_to_ids("<|audio_pad|>")

        # Sicherheitscheck
        assert bos_token_id != text_tokenizer.unk_token_id, "<|audio_bos|> nicht im Vocab!"
        assert pad_token_id != text_tokenizer.unk_token_id, "<|audio_pad|> nicht im Vocab!"
        audio_token_id = text_tokenizer.convert_tokens_to_ids("<|audio|>")

        # 1. Encode → (K, T_audio)
        codes, length = encode_audio(example[audio_column], speech_tokenizer)

        # 2. Delay + Offset → (K, T_audio + K - 1)
        codes_delayed = apply_delay_pattern(codes, bos_token_id, pad_token_id)
        codes_delayed[codes_delayed == bos_token_id] = -1
        codes_delayed[codes_delayed == pad_token_id] = -1
        K, T_audio    = codes_delayed.shape

        # 3. Text tokenisieren
        task_prefix  = text_tokenizer(f"[{actual_task} en text] ", add_special_tokens=False).input_ids
        output_sep   = text_tokenizer(" OUTPUT: ",                 add_special_tokens=False).input_ids
        text_tokens  = text_tokenizer(example[text_column],        add_special_tokens=False).input_ids
        eos          = [text_tokenizer.eos_token_id]

        if flip:  # TTS: Text rein → Audio raus
            prefix_ids = task_prefix + text_tokens + output_sep
            suffix_ids = eos
            T_pre      = len(prefix_ids)
            text_ids   = prefix_ids + [audio_token_id] * T_audio + suffix_ids
        else:     # STT: Audio rein → Text raus
            prefix_ids = task_prefix
            suffix_ids = output_sep + text_tokens + eos
            T_pre      = len(prefix_ids)
            text_ids   = prefix_ids + [audio_token_id] * T_audio + suffix_ids

        T_total = len(text_ids)

        # 4. audio_codes auf volle Sequenzlänge expandieren — -1 an Text-Positionen
        audio_codes_full = torch.full((K, T_total), -1, dtype=torch.long)
        audio_codes_full[:, T_pre : T_pre + T_audio] = codes_delayed

        # 5. Padding auf max_text_length + max_audio_length
        max_len = max_text_length + T_audio  # dynamisch, Collator macht den Rest
        attention_mask = [1] * T_total

        return {
            "text_ids":       text_ids,
            "audio_codes":    audio_codes_full.tolist(),
            "attention_mask": attention_mask,
            "input_length":   length,
        }

    map_fn = partial(apply_template, task=task, src_lang=src_lang, tgt_lang=tgt_lang, flip=flip)
    return dataset.map(map_fn, batched=False, remove_columns=dataset.column_names, num_proc=num_proc)


def filter_audio_length(dataset, duration_threshold):
    return dataset.filter(lambda x: x < duration_threshold, input_columns=["input_length"])


def create_datasets(args: argparse.Namespace) -> None:
    assert args.sampling_rate == 16000, "SpeechTokenizer requires 16kHz audio."

    speech_tokenizer = load_speech_tokenizer(args.speech_tokenizer_config, args.speech_tokenizer_ckpt)
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)

    dataset = load_dataset(args.path, args.name, split=args.split)
    dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=args.sampling_rate))

    if args.text_column != "text":
        dataset = dataset.rename_column(args.text_column, "text")

    flip_split = int(args.flip_ratio * len(dataset))
    indices = list(range(len(dataset)))
    flip_ds = dataset.select(indices[:flip_split])
    no_flip_ds = dataset.select(indices[flip_split:])

    common_kwargs = dict(
        speech_tokenizer=speech_tokenizer,
        text_tokenizer=text_tokenizer,
        task="STT",
        src_lang=args.language or "src",
        tgt_lang="text",
        audio_column=args.audio_column,
        text_column="text",
        text_vocab_size=args.text_vocab_size,
        audio_vocab_size=args.audio_vocab_size,
        max_text_length=args.max_text_length,
        num_proc=args.num_proc,
    )

    flipped_dataset = preprocess_dataset(flip_ds, **common_kwargs, flip=True)
    unflipped_dataset = preprocess_dataset(no_flip_ds, **common_kwargs, flip=False)

    generated_dataset = concatenate_datasets([flipped_dataset, unflipped_dataset]).shuffle(seed=1337)
    generated_dataset = filter_audio_length(generated_dataset, args.max_duration)
    generated_dataset = generated_dataset.map(lambda _: {"language": args.language})

    split_dataset = generated_dataset.train_test_split(test_size=args.train_test_ratio, seed=1337)
    split_dataset["validation"] = split_dataset.pop("test")

    split_dataset.save_to_disk(args.output_dir)
    print(f"Saved {len(generated_dataset)} samples → {args.output_dir}")
    print(split_dataset)
    open(f"{args.output_dir}/.done", "w").close()
    print("Done flag set.")


if __name__ == "__main__":
    create_datasets(args)