# !/bin/bash

uv run scripts/generate_t2t_dataset.py \
    --dataset_name='dvgodoy/yoda_sentences' \
    --split=0.5 \
    --flip_ratio=0.5 \
    --num_proc=2 \
    --src_column='sentence' \
    --tgt_column='translation' \
    --train_test_ratio=0.1 \
    --max_length=128 \
    --tokenizer='Qwen/Qwen2.5-0.5B' \
    --data_dir'=data/'
