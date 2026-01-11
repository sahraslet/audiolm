#!/bin/bash

uv run python t2t_preprocessing.py \
  --path="Darth-Vaderr/English-German,nvidia/granary" \
  --subset="None,de" \
  --split="None,ast" \
  --src_lang="en,de" \
  --tgt_lang="de,en" \
  --tokenizer='bert-base-multilingual-uncased' \
  --max_length=128 \
  --output_dir="ttt_dataset_english_german"
