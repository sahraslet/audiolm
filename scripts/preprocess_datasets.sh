#!/bin/bash
# preprocess_datasets.sh
# Runs all preprocessing tasks and merges datasets.
# Safe to rerun — skips already completed tasks.

echo "USER: $(whoami)"
echo "HOME: $HOME"
ls -ld $HOME

#apt get install python -y
python3 -m venv ./venv
#PYTHON=/nethome/sslet/audiolm/.venv/bin/python
PYTHON=./venv/bin/python
#DATA_DIR=/data/users/sslet/audiolm/data
DATA_DIR=./data

pip install -r requirements.txt

#cd /nethome/sslet/audiolm
mkdir -p $DATA_DIR
ls -al
if [ ! -f "$DATA_DIR/asr_english/.done" ]; then
    echo "=== [1/5] ASR English ==="
    $PYTHON scripts/asr_preprocessing.py \
        --path='openslr/librispeech_asr' \
        --name='clean' \
        --split='train[:10%]' \
        --text_column='text' \
        --speech_tokenizer_config='fnlp/SpeechTokenizer' \
        --text_tokenizer='Qwen/Qwen2.5-0.5B' \
        --language='en' \
        --max_duration=30.0 \
        --output_dir=$DATA_DIR/asr_english \
        --sampling_rate=16000
    if [ $? -ne 0 ]; then echo "ERROR: ASR English failed!"; exit 1; fi
    touch $DATA_DIR/asr_english/.done
else
    echo "=== [1/5] ASR English — skipping (already done)"
fi

if [ ! -f "$DATA_DIR/asr_german/.done" ]; then
    echo "=== [2/5] ASR German ==="
    $PYTHON scripts/asr_preprocessing.py \
        --path='flozi00/asr-german-mixed' \
        --split='train[:10%]' \
        --text_column='transkription' \
        --speech_tokenizer_config='fnlp/SpeechTokenizer' \
        --text_tokenizer='Qwen/Qwen2.5-0.5B' \
        --language='de' \
        --max_duration=30.0 \
        --output_dir=$DATA_DIR/asr_german \
        --sampling_rate=16000
    if [ $? -ne 0 ]; then echo "ERROR: ASR German failed!"; exit 1; fi
    touch $DATA_DIR/asr_german/.done
else
    echo "=== [2/5] ASR German — skipping (already done)"
fi

if [ ! -f "$DATA_DIR/s2st/.done" ]; then
    echo "=== [3/5] S2ST ==="
    $PYTHON scripts/s2st_preprocessing.py \
        --path='PedroDKE/LibriS2S' \
        --split='train' \
        --source_audio_column='DE_audio' \
        --target_audio_column='EN_audio' \
        --src_lang='de' \
        --tgt_lang='en' \
        --speech_tokenizer_config='fnlp/SpeechTokenizer' \
        --text_tokenizer='Qwen/Qwen2.5-0.5B' \
        --max_duration=30.0 \
        --output_dir=$DATA_DIR/s2st
    if [ $? -ne 0 ]; then echo "ERROR: S2ST failed!"; exit 1; fi
    touch $DATA_DIR/s2st/.done
else
    echo "=== [3/5] S2ST — skipping (already done)"
fi

if [ ! -f "$DATA_DIR/t2t/.done" ]; then
    echo "=== [4/5] T2T ==="
    $PYTHON scripts/generate_t2t_dataset.py \
        --dataset_name='Darth-Vaderr/English-German' \
        --split='train[:30%]' \
        --flip_ratio=0.5 \
        --num_proc=2 \
        --src_column='German' \
        --tgt_column='English' \
        --train_test_ratio=0.1 \
        --max_length=128 \
        --tokenizer='Qwen/Qwen2.5-0.5B' \
        --data_dir=$DATA_DIR/t2t
    if [ $? -ne 0 ]; then echo "ERROR: T2T failed!"; exit 1; fi
    touch $DATA_DIR/t2t/.done
else
    echo "=== [4/5] T2T — skipping (already done)"
fi

if [ ! -f "$DATA_DIR/all/.done" ]; then
    echo "=== [5/5] Merging datasets ==="
    $PYTHON scripts/merge_datasets.py \
        --input_dirs $DATA_DIR/asr_english $DATA_DIR/asr_german $DATA_DIR/s2st $DATA_DIR/t2t \
        --output_dir $DATA_DIR/all \
        --seed 1337 \
        --push_to_hub \
        --hub_repo_id sahara22/audiolm-eval
    if [ $? -ne 0 ]; then echo "ERROR: Merge failed!"; exit 1; fi
    touch $DATA_DIR/all/.done
else
    echo "=== [5/5] Merge — skipping (already done)"
fi

echo "=== Preprocessing pipeline complete ==="
echo "Dataset ready at $DATA_DIR/all"
