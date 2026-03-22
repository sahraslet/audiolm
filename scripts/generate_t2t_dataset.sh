#!/bin/bash

source /nethome/sslet/audiolm/.venv/bin/activate

python scripts/generate_t2t_dataset.py \
    --dataset_name='Darth-Vaderr/English-German' \
    --split="train[:30%]" \
    --flip_ratio=0.5 \
    --num_proc=2 \
    --src_column='German' \
    --tgt_column='English' \
    --train_test_ratio=0.1 \
    --max_length=128 \
    --tokenizer='Qwen/Qwen2.5-0.5B' \
    --data_dir='/data/users/sslet/audiolm/data/t2t'
