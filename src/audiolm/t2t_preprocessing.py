"""Preprocessing pipeline for translation datasets used in Text-to-Text tasks."""

from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import argparse

parser = argparse.ArgumentParser(description="Preprocess audio datasets.")
parser.add_argument("--path", type=str, required=True, help="Hugging Face dataset path(s), comma-separated for multiple datasets.")
parser.add_argument("--subset", type=str, default=None, help="Subset(s) of the dataset if applicable, comma-separated for multiple datasets.")
parser.add_argument("--split", type=str, default="train", help="Dataset split(s) to use, comma-separated for multiple datasets.")
parser.add_argument("--src_lang", type=str, required=True, help="Source language column name.")
parser.add_argument("--tgt_lang", type=str, required=True, help="Target language column name.")
parser.add_argument("--tokenizer", type=str, default="bert-base-multilingual-cased", help="Pretrained tokenizer name.")
parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length.")
parser.add_argument("--output_dir", type=str, default="./preprocessed", help="Directory to save the preprocessed dataset.")
args = parser.parse_args()


# Reduce dataset to necessary columns
def reduce_columns(dataset, columns_to_keep):
    '''Reduce dataset to only specified columns.'''
    all_columns = dataset.column_names
    columns_to_remove = set(all_columns) - set(columns_to_keep)

    return dataset.remove_columns(columns_to_remove)


def translation_direction(src_lang, tgt_lang):
    """Add explicit translation direction prefix."""
    prefix = f"<{src_lang}2{tgt_lang}> "
    return prefix


def preprocess_function(batch, tokenizer, max_length, src_lang, tgt_lang):
    """Tokenize source and target text."""
    inputs = [translation_direction(src_lang, tgt_lang) + example for example in batch[src_lang]]
    targets = [example for example in batch[tgt_lang]]
    model_inputs = tokenizer(inputs, text_target =targets, max_length=max_length, truncation=True)
    return model_inputs


def make_ttt_dataset(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    datasets_list = []

    dataset_paths = args.path.split(",")
    subsets = args.subset.split(",")
    splits = args.split.split(",")
    src_langs = args.src_lang.split(",")
    tgt_langs = args.tgt_lang.split(",")

    for path, subset, split, src, tgt in zip(dataset_paths, subsets, splits, src_langs, tgt_langs):
        kwargs = {}
        if subset and subset != "None":
            kwargs["subset"] = subset
        if split and split != "None":
            kwargs["split"] = split
        dataset = load_dataset(path, **kwargs)
        # Reduce to src and tgt columns
        dataset = dataset.remove_columns([c for c in dataset.column_names if c not in [src, tgt]])

        # Rename columns to unified names
        if src != "src":
            dataset = dataset.rename_column(src, "src")
        if tgt != "tgt":
            dataset = dataset.rename_column(tgt, "tgt")

        # Tokenize
        dataset = dataset.map(lambda batch: preprocess_function(batch, tokenizer, src_lang="src", tgt_lang="tgt",
                                                                max_length=args.max_length),
                              batched=True)
        datasets_list.append(dataset)

    # Combine all datasets
    final_dataset = concatenate_datasets(datasets_list)
    out_path = f"{args.output_dir}"
    final_dataset.save_to_disk(out_path)
    print(f"Saved {len(final_dataset)} samples → {out_path}")

make_ttt_dataset(args=args)



