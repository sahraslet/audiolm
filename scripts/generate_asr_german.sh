#!/bin/bash
# Script to preprocess audio data for ASR tasks using a Python script.
# German asr dataset

source /nethome/sslet/audiolm/.venv/bin/activate

python src/audiolm/asr_preprocessing.py \
    --path='flozi00/asr-german-mixed' \
    --split='train[10:]' \
    --text_column='transkription' \
    --speech_tokenizer_config='fnlp/SpeechTokenizer' \
    --text_tokenizer='Qwen/Qwen2.5-0.5B' \
    --language='de' \
    --max_duration=30.0 \
    --output_dir='/data/users/sslet/audiolm/data/asr_german' \
    --sampling_rate=16000
