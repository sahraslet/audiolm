#!/bin/bash
# Script to preprocess audio data for ASR tasks using a Python script.

# English asr dataset

uv run python src/audiolm/asr_preprocessing.py \
    --path='openslr/librispeech_asr' \
    --name='clean' \
    --split='train' \
    --text_column='text' \
    --audio_processor='facebook/w2v-bert-2.0' \
    --text_tokenizer='facebook/wav2vec2-base-960h'
    --max_duration=30.0 \
    --output_dir='asr_dataset_english' \
    --sampling_rate=16000

# German asr dataset

uv run python src/audiolm/asr_preprocessing.py \
    --path='flozi00/asr-german-mixed' \
    --split='train' \
    --text_column='transkription' \
    --audio_processor='facebook/w2v-bert-2.0' \
    --text_tokenizer='facebook/wav2vec2-base-960h' \
    --max_duration=30.0 \
    --output_dir='asr_dataset_german' \
    --sampling_rate=16000
