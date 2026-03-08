uv run python src/audiolm/s2st_preprocessing.py \
    --path='google/cvss' \
    --name='en_de' \
    --split='train' \
    --speech_tokenizer_config='fnlp/SpeechTokenizer' \
    --text_tokenizer='mistralai/Mistral-7B-v0.1' \
    --language='en_de' \
    --output_dir='s2st_dataset_en_de'