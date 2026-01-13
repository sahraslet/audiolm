#!/bin/bash

uv run audio_tokenization.py \
  --model_name="facebook/w2v-bert-2.0" \
  --dataset_path="asr_dataset_english, asr_dataset_german" \
  --num_clusters=1024 \
  --batch_size=16 \
  --output_dir="./discretized_dataset_english_german"

