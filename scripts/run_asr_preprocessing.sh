#!/bin/bash
# Script to preprocess audio data for ASR tasks using a Python script.

# English asr dataset

uv run python src/audiolm/asr_preprocessing.py \
    --path='openslr/librispeech_asr' \
    --name='clean' \
    --split='train' \
    --text_column='text' \
    --speech_tokenizer_config='fnlp/SpeechTokenizer' \
    --text_tokenizer='mistralai/Mistral-7B-v0.1'
    --language='en' \
    --max_duration=30.0 \
    --output_dir='asr_dataset_english' \
    --sampling_rate=24000

# German asr dataset

uv run python src/audiolm/asr_preprocessing.py \
    --path='flozi00/asr-german-mixed' \
    --split='train' \
    --text_column='transkription' \
    --speech_tokenizer_config='fnlp/SpeechTokenizer' \
    --text_tokenizer='mistralai/Mistral-7B-v0.1' \
    --language='de' \
    --max_duration=30.0 \
    --output_dir='asr_dataset_german' \
    --sampling_rate=24000
