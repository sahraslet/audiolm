# merge_datasets.py

import argparse
from datasets import load_from_disk, concatenate_datasets, DatasetDict

parser = argparse.ArgumentParser()
parser.add_argument("--input_dirs", nargs="+", required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--push_to_hub",  action="store_true", default=False)
parser.add_argument("--hub_repo_id",  type=str, default="sahara22/audiolm-eval")
args = parser.parse_args()

train_splits = []
val_splits   = []

for path in args.input_dirs:
    # Prüfen ob Dataset fertig ist
    import os
    assert os.path.exists(f"{path}/.done"), f"{path} nicht fertig! Preprocessing nochmal starten."

    ds = load_from_disk(path)
    train_splits.append(ds["train"])
    val_splits.append(ds["validation"])
    print(f"Loaded {path}: {len(ds['train'])} train / {len(ds['validation'])} val")

full_train = concatenate_datasets(train_splits).shuffle(seed=args.seed)
full_val   = concatenate_datasets(val_splits).shuffle(seed=args.seed)

full_ds = DatasetDict({"train": full_train, "validation": full_val})
full_ds.save_to_disk(args.output_dir)

# Done-Flag setzen
open(f"{args.output_dir}/.done", "w").close()

print(f"Saved merged dataset → {args.output_dir}")
print(f"Train: {len(full_train)} | Val: {len(full_val)}")
if args.push_to_hub:
    print("Pushing validation split to HuggingFace...")
    full_ds["validation"].push_to_hub(
        args.hub_repo_id,
        split="validation",
    )
    print(f"Validation split pushed to {args.hub_repo_id}")