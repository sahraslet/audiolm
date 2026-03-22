uv run python src/audiolm/s2st_preprocessing.py \
    --path='PedroDKE/LibriS2S' \
    --split='train' \
    --speech_tokenizer_config='fnlp/SpeechTokenizer' \
    --text_tokenizer='Qwen/Qwen2.5-0.5B' \
    --language='en_de' \
    --output_dir='data/s2st'