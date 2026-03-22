#!/bin/bash
# Script to preprocess audio data for ASR tasks using a Python script.

# English asr dataset

uv run python src/audiolm/asr_preprocessing.py \
    --path='openslr/librispeech_asr' \
    --name='clean' \
    --split='train[:10%]' \
    --text_column='text' \
    --speech_tokenizer_config='fnlp/SpeechTokenizer' \
    --text_tokenizer='Qwen/Qwen2.5-0.5B'
    --language='en' \
    --max_duration=30.0 \
    --output_dir='data/' \
    --sampling_rate=16000


