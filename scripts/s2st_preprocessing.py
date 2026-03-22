"""
Preprocessing CVSS dataset for Speech-to-Speech Translation (S2ST).

Output format:
    text_ids:       List[int]        (T_total,)   — Task-Prefix + audio_token_id Platzhalter
    audio_codes:    List[List[int]]  (K, T_total) — -1 an Text-Positionen
    attention_mask: List[int]        (T_total,)
    input_length:   float
    language:       str

Template:
    [S2ST en de] <audio>*T_src OUTPUT: <audio>*T_tgt <eos>
    flip=True → Richtung umkehren: [S2ST de en] ...
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
parser.add_argument("--path",                    type=str, required=True)
parser.add_argument("--name",                    type=str, default=None)
parser.add_argument("--split",                   type=str, default="train")
parser.add_argument("--source_audio_column",     type=str, default="DE_audio")
parser.add_argument("--target_audio_column",     type=str, default="EN_audio")
parser.add_argument("--speech_tokenizer_config", type=str, default="fnlp/SpeechTokenizer")
parser.add_argument("--speech_tokenizer_ckpt",   type=str, default=None)
parser.add_argument("--text_tokenizer",          type=str, default="Qwen/Qwen2.5-0.5B")
parser.add_argument("--src_lang",                type=str, default="en")
parser.add_argument("--tgt_lang",                type=str, default="de")
parser.add_argument("--max_duration",            type=float, default=30.0)
parser.add_argument("--sampling_rate",           type=int, default=16000)
parser.add_argument("--max_text_length",         type=int, default=32)
parser.add_argument("--max_audio_length",        type=int, default=1024)
parser.add_argument("--flip_ratio",              type=float, default=0.5)
parser.add_argument("--train_test_ratio",        type=float, default=0.05)
parser.add_argument("--num_proc",                type=int, default=1)
parser.add_argument("--output_dir",              type=str, default="./preprocessed/s2st")
parser.add_argument("--audio_vocab_size",        type=int, default=1024)
parser.add_argument("--n_codebooks",             type=int, default=8)
args = parser.parse_args()


# =========================
# LOAD TOKENIZER
# =========================

def load_speech_tokenizer(config_path, ckpt_path):
    if ckpt_path is not None:
        model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    else:
        config_file = hf_hub_download(repo_id=config_path, filename="speechtokenizer_hubert_avg/config.json")
        ckpt_file   = hf_hub_download(repo_id=config_path, filename="speechtokenizer_hubert_avg/SpeechTokenizer.pt")
        model = SpeechTokenizer.load_from_checkpoint(config_file, ckpt_file)
    model.eval()
    return model


# =========================
# AUDIO ENCODING
# =========================

def encode_audio(audio, model):
    """Encode audio → (K, T) codes and length in seconds."""
    waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        codes = model.encode(waveform)  # (K, B, T)
    codes = codes[:, 0, :].cpu()        # (K, T)
    length = len(audio["array"]) / audio["sampling_rate"]
    return codes, length


# =========================
# PREPROCESS
# =========================

def preprocess_dataset(
    dataset,
    speech_tokenizer,
    text_tokenizer,
    src_lang,
    tgt_lang,
    source_audio_column,
    target_audio_column,
    max_text_length,
    max_audio_length,
    num_proc,
    flip=False,
):
    def apply_template(example):
        bos_token_id   = text_tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        pad_token_id   = text_tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        audio_token_id = text_tokenizer.convert_tokens_to_ids("<|audio|>")

        assert bos_token_id != text_tokenizer.unk_token_id, "<|audio_bos|> nicht im Vocab!"
        assert pad_token_id != text_tokenizer.unk_token_id, "<|audio_pad|> nicht im Vocab!"

        # 1. Encode beide Audio-Streams → (K, T)
        src_codes, src_length = encode_audio(example[source_audio_column], speech_tokenizer)
        tgt_codes, tgt_length = encode_audio(example[target_audio_column], speech_tokenizer)

        # 2. Delay Pattern → (K, T + K - 1), BOS/PAD → -1
        src_codes_delayed = apply_delay_pattern(src_codes, bos_token_id, pad_token_id)
        src_codes_delayed[src_codes_delayed == bos_token_id] = -1
        src_codes_delayed[src_codes_delayed == pad_token_id] = -1

        tgt_codes_delayed = apply_delay_pattern(tgt_codes, bos_token_id, pad_token_id)
        tgt_codes_delayed[tgt_codes_delayed == bos_token_id] = -1
        tgt_codes_delayed[tgt_codes_delayed == pad_token_id] = -1

        # 3. Flip: Übersetzungsrichtung umkehren
        if flip:
            input_lang, output_lang   = tgt_lang, src_lang
            input_codes, output_codes = tgt_codes_delayed, src_codes_delayed
            input_length              = tgt_length
        else:
            input_lang, output_lang   = src_lang, tgt_lang
            input_codes, output_codes = src_codes_delayed, tgt_codes_delayed
            input_length              = src_length

        # 4. Truncaten auf max_audio_length
        K            = input_codes.shape[0]
        input_codes  = input_codes[:,  :max_audio_length]
        output_codes = output_codes[:, :max_audio_length]
        T_src        = input_codes.shape[1]
        T_tgt        = output_codes.shape[1]

        # 5. Task-Prefix tokenisieren
        task_prefix = text_tokenizer(
            f"[S2ST {input_lang} {output_lang}] ",
            add_special_tokens=False
        ).input_ids[:max_text_length]

        output_sep = text_tokenizer(" OUTPUT: ", add_special_tokens=False).input_ids
        eos        = [text_tokenizer.eos_token_id]

        # 6. Flacher text_ids Stream:
        #    [S2ST en de] <audio>*T_src OUTPUT: <audio>*T_tgt <eos>
        T_pre = len(task_prefix)
        T_sep = len(output_sep)

        text_ids = (
            task_prefix
            + [audio_token_id] * T_src
            + output_sep
            + [audio_token_id] * T_tgt
            + eos
        )

        T_total = len(text_ids)
        T_mid   = T_pre + T_src + T_sep  # Start Output-Audio

        # 7. audio_codes: (K, T_total), -1 an Text-Positionen
        audio_codes_full = torch.full((K, T_total), -1, dtype=torch.long)
        audio_codes_full[:, T_pre : T_pre + T_src] = input_codes
        audio_codes_full[:, T_mid : T_mid + T_tgt] = output_codes

        attention_mask = [1] * T_total

        return {
            "text_ids":       text_ids,
            "audio_codes":    audio_codes_full.tolist(),
            "attention_mask": attention_mask,
            "input_length":   input_length,
        }

    return dataset.map(
        apply_template,
        batched=False,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )


# =========================
# FILTER
# =========================

def filter_audio_length(dataset, duration_threshold):
    return dataset.filter(lambda x: x < duration_threshold, input_columns=["input_length"])


# =========================
# MAIN PIPELINE
# =========================

def create_datasets(args):
    assert args.sampling_rate == 16000, "SpeechTokenizer requires 16kHz audio."

    print("Loading tokenizers...")
    speech_tokenizer = load_speech_tokenizer(args.speech_tokenizer_config, args.speech_tokenizer_ckpt)
    text_tokenizer   = AutoTokenizer.from_pretrained(args.text_tokenizer)
    special_tokens = ["<|audio|>", "<|audio_bos|>", "<|audio_pad|>"]
    text_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Checken ob es geklappt hat:
    for t in special_tokens:
        tid = text_tokenizer.convert_tokens_to_ids(t)
        print(f"{t} → {tid}")

    print("Loading dataset...")
    dataset = load_dataset(args.path, split=args.split)
    dataset = dataset.filter(lambda x: x["score"] >= 0.8)
    print(f"Nach Score-Filter (>= 0.8): {len(dataset)} Samples")
    dataset = dataset.cast_column(args.source_audio_column, Audio(sampling_rate=args.sampling_rate))
    dataset = dataset.cast_column(args.target_audio_column, Audio(sampling_rate=args.sampling_rate))
    print(f"Dataset size: {len(dataset)}")

    flip_split = int(args.flip_ratio * len(dataset))
    indices    = list(range(len(dataset)))
    flip_ds    = dataset.select(indices[:flip_split])
    no_flip_ds = dataset.select(indices[flip_split:])

    common_kwargs = dict(
        speech_tokenizer=speech_tokenizer,
        text_tokenizer=text_tokenizer,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        source_audio_column=args.source_audio_column,
        target_audio_column=args.target_audio_column,
        max_text_length=args.max_text_length,
        max_audio_length=args.max_audio_length,
        num_proc=args.num_proc,
    )

    print(f"Processing flipped ({args.tgt_lang}→{args.src_lang})...")
    flipped_dataset   = preprocess_dataset(flip_ds,   **common_kwargs, flip=True)

    print(f"Processing normal ({args.src_lang}→{args.tgt_lang})...")
    unflipped_dataset = preprocess_dataset(no_flip_ds, **common_kwargs, flip=False)

    print("Merging + shuffling...")
    generated_dataset = concatenate_datasets([flipped_dataset, unflipped_dataset]).shuffle(seed=1337)

    print("Filtering by audio duration...")
    generated_dataset = filter_audio_length(generated_dataset, args.max_duration)
    generated_dataset = generated_dataset.map(lambda _: {"language": f"{args.src_lang}_{args.tgt_lang}"})

    print("Train/val split...")
    split_dataset = generated_dataset.train_test_split(test_size=args.train_test_ratio, seed=1337)
    split_dataset["validation"] = split_dataset.pop("test")

    print(f"Saving to {args.output_dir}...")
    split_dataset.save_to_disk(args.output_dir)
    print(f"Saved {len(generated_dataset)} samples → {args.output_dir}")
    print(split_dataset)
    open(f"{args.output_dir}/.done", "w").close()
    print("Done flag set.")


if __name__ == "__main__":
    create_datasets(args)