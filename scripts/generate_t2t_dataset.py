from datasets import load_dataset, Dataset, concatenate_datasets
import argparse
from functools import partial
from transformers import AutoTokenizer, PreTrainedTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
# parser.add_argument("--subset", type=str)
parser.add_argument("--split", type=float)
parser.add_argument("--flip_ratio", type=float)
parser.add_argument("--num_proc", type=int)
parser.add_argument("--src_column", type=str)
parser.add_argument("--tgt_column", type=str)
parser.add_argument("--train_test_ratio", type=float)
parser.add_argument("--max_length", type=int)
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--data_dir", type=str)

args = parser.parse_args()


def preprocess_dataset(
        dataset: Dataset, 
        tokenizer: PreTrainedTokenizer, 
        task: str,
        src_column: str,
        tgt_column: str,
        max_length: int,
        num_proc: int,
        flip=False,
    ) -> Dataset:
    
    def apply_template( example: dict, task: str, src: str, tgt: str, tokenizer: PreTrainedTokenizer, max_length: int = 128) -> dict:
        """
        A general template for multi-task formatting.

        [T2T English German]  Resumption of the session OUTPUT:  Wiederaufnahme der Sitzungsperiode <|endoftext|>
          |     |       |                    |                              |                            |
         task   src    tgt              src sentnece                  tgt sentence                    eos_token

        Args:
            task (str): Task abreviation
            src (str): source language
            tgt (str): target language
            example (DatasetDict): a row from the dataset to be mapped
            tokenizer (AutoTokenizer): the tokenizer
            max_length (int): maximum length of tokenized output.

        Returns:
            dict: text formatted to the template
        """
        text = f"[{task} {src} {tgt}] {example[src]} OUTPUT: {example[tgt]} <|endoftext|>"
        tokenized = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
        return {
            'text' : text,
            **tokenized
        }
    
    if flip:
        src, tgt = tgt_column, src_column
    else:
        src, tgt = src_column, tgt_column

    map_fn = partial(
        apply_template,
        task=task,
        src=src,
        tgt=tgt,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    dataset = dataset.map(
        map_fn,
        batched=False,
        remove_columns=dataset.column_names,
        num_proc=num_proc
    )

    return dataset


def create_datasets(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = load_dataset(args.dataset_name)
    print(f"using {args.tokenizer} on {args.dataset_name}")
    dataset = dataset['train'].select(range(int(args.split * len(dataset['train']))))
    flip_split = int(args.flip_ratio * len(dataset))
    indices = list(range(len(dataset)))
    flip = dataset.select(indices[:flip_split])
    no_flip = dataset.select(indices[flip_split:])

    flipped_dataset = preprocess_dataset(
        flip,
        tokenizer,
        task = 'T2T',
        src_column=args.src_column,
        tgt_column=args.tgt_column,
        max_length=args.max_length,
        num_proc=args.num_proc,
        flip=True
    )

    unflipped_dataset = preprocess_dataset(
        no_flip,
        tokenizer,
        task = 'T2T',
        src_column=args.src_column,
        tgt_column=args.tgt_column,
        max_length=args.max_length,
        num_proc=args.num_proc,
        flip=False
    )

    generated_dataset = concatenate_datasets(
        [flipped_dataset, unflipped_dataset]
    ).shuffle(seed=1337)

    split_dataset = generated_dataset.train_test_split(test_size=args.train_test_ratio, seed=1337)
    split_dataset['validation'] = split_dataset.pop('test')
    
    split_dataset.save_to_disk(args.data_dir)


create_datasets(args=args)
