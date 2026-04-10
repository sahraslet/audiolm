#!/bin/bash
# train.sh

source /nethome/sslet/audiolm/.venv/bin/activate

python src/audiolm/scripts/train.py \
    --tokenizer_path Qwen/Qwen2.5-0.5B \
    --dataset_path "sahara22/audiolm-eval" \
    --checkpoint_dir /data/users/sslet/audiolm/checkpoints \
    --logfile_path /data/users/sslet/audiolm/logs/train.log \
    --wandb_project_name audio-lm \
    --wandb_entity sahratls-universit-t-des-saarlandes-saarland-university \
    --wandb_run_name first-run \
    --lr 3e-4 \
    --num_epochs 2 \
    --batch_size 4 \
    --eval_every 500 \
    --save_every 500 \
    --grad_accumulation_steps 4 \
    --device cuda \
    --push_to_hub \
    --hub_repo_id sahara22/audio_language_model