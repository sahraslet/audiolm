'''Preprocessing datasets for audio tasks.'''
from transformers import AutoProcessor
from datasets import load_dataset, concatenate_datasets, Audio
import argparse

def reduce_columns(dataset, columns_to_keep):
    '''Reduce dataset to only specified columns.'''
    all_columns = dataset.column_names
    columns_to_remove = set(all_columns) - set(columns_to_keep)

    return dataset.remove_columns(columns_to_remove)

def prepare_dataset(batch, processor):
    '''Prepare dataset by processing audio and text.'''
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["text"])

    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    return batch

def filter_audio_length(dataset, duration_threshold):
    '''Filter audio samples longer than duration_threshold seconds.'''
    def is_audio__length_in_range(input_length):
        return input_length < duration_threshold
    return dataset.filter(is_audio__length_in_range, input_columns=["input_length"])

def main(language):
    model_checkpoint = "openai/whisper-small"
    processor = AutoProcessor.from_pretrained(model_checkpoint)

    datasets_list = []

    if language in ["arabic", "german_arabic","english_arabic", "all"]:
        # ASR datasets for Arabic
        arabic_dataset_1 = load_dataset("UBC-NLP/Casablanca", split="validation")
        arabic_dataset_2 = load_dataset("Ahmed107/arabic-90", split="train")
        arabic_dataset_3 = load_dataset("MadLook/arabic-whisper-multidialect", split="train")

        # Standardize audio column to 16kHz
        arabic_dataset_1 = arabic_dataset_1.cast_column("audio", Audio(sampling_rate=16000))
        arabic_dataset_2 = arabic_dataset_2.cast_column("audio", Audio(sampling_rate=16000))
        arabic_dataset_3 = arabic_dataset_3.cast_column("audio", Audio(sampling_rate=16000))

        # Reduce to necessary columns
        arabic_dataset_1 = reduce_columns(arabic_dataset_1, ["audio", "text"])
        arabic_dataset_2 = reduce_columns(arabic_dataset_2, ["audio", "text"])
        arabic_dataset_3 = reduce_columns(arabic_dataset_3, ["audio", "sentence"])

        # Prepare first Arabic dataset
        arabic_dataset_1 = arabic_dataset_1.map(lambda batch: prepare_dataset(batch, processor), remove_columns=arabic_dataset_1.column_names)
        arabic_dataset_1 = filter_audio_length(arabic_dataset_1, duration_threshold=30.0)

        # Prepare second Arabic dataset
        arabic_dataset_2 = arabic_dataset_2.map(lambda batch: prepare_dataset(batch, processor), remove_columns=arabic_dataset_2.column_names)
        arabic_dataset_2 = filter_audio_length(arabic_dataset_2, duration_threshold=30.0)

        # Prepare third Arabic dataset
        arabic_dataset_3 = arabic_dataset_3.rename_column("sentence", "text")
        arabic_dataset_3 = arabic_dataset_3.map(lambda batch: prepare_dataset(batch, processor), remove_columns=arabic_dataset_3.column_names)
        arabic_dataset_3 = filter_audio_length(arabic_dataset_3, duration_threshold=30.0)

        # Concatenate Arabic datasets
        arabic_dataset = concatenate_datasets([arabic_dataset_1, arabic_dataset_2, arabic_dataset_3])

        datasets_list.append(arabic_dataset)

    if language in ["english","german_english", "english_arabic", "all"]:
        # ASR dataset for English
        english_dataset = load_dataset("openslr/librispeech_asr", "clean", split="train")

        # Standardize audio column to 16kHz
        english_dataset = english_dataset.cast_column("audio", Audio(sampling_rate=16000))

        # Reduce to necessary columns
        english_dataset = reduce_columns(english_dataset, ["audio", "text"])

        # Prepare English dataset
        english_dataset = english_dataset.map(lambda batch: prepare_dataset(batch, processor), remove_columns=english_dataset.column_names)
        english_dataset = filter_audio_length(english_dataset, duration_threshold=30.0)

        datasets_list.append(english_dataset)

    if language in ["german", "german_english","german_arabic", "all"]:
        # ASR dataset for German
        german_dataset = load_dataset("flozi00/asr-german-mixed", split="train")

        # Standardize audio column to 16kHz
        german_dataset = german_dataset.cast_column("audio", Audio(sampling_rate=16000))

        # Reduce to necessary columns
        german_dataset = reduce_columns(german_dataset, ["audio", "transkription"])

        # Rename column to 'text'
        german_dataset = german_dataset.rename_column("transkription", "text")

        # Prepare German dataset
        german_dataset = german_dataset.map(lambda batch: prepare_dataset(batch, processor), remove_columns=german_dataset.column_names)
        german_dataset = filter_audio_length(german_dataset, duration_threshold=30.0)

        datasets_list.append(german_dataset)

    # Combine all selected datasets
    if len(datasets_list) > 1:
        final_dataset = concatenate_datasets(datasets_list)
    else:
        final_dataset = datasets_list[0]

    # Save the final preprocessed dataset
    final_dataset.save_to_disk("./preprocessed/combined")
    print(f"Final dataset size: {len(final_dataset)} samples.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio datasets.")
    parser.add_argument("--language", type = str, choices=["arabic", "english", "german", "german_english",
                                                           "german_arabic","english_arabic", "all"],
                                                            help="Choose one more more languages.")
    args = parser.parse_args()
    main(args.language)


