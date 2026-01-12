"""
Preprocessing audio datasets for STT and TTS tasks.:

Output format:
    Dataset({
        features: ['input_features', 'labels', 'input_length'],
        num_rows: <N>
    })

"""

import torch
from transformers import AutoProcessor, Wav2Vec2CTCTokenizer, Wav2Vec2BertModel
from datasets import load_dataset, concatenate_datasets, Audio
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="Hugging Face dataset path.")
parser.add_argument("--name", type=str, default=None, help="Hugging Face dataset name (subset).")
parser.add_argument("--split", type=str, default="train", help="Dataset split to use.")
parser.add_argument("--text_column", type=str, default="text", help="Name of the text transcription column.")
parser.add_argument("--audio_processor", type=str, default="facebook/w2v-bert-2.0",
                    help="Pretrained model name for processor.")
parser.add_argument("--text_tokenizer", type=str, default="facebook/wav2vec2-base-960h")
parser.add_argument("--max_duration", type=float, default=30.0, help="Maximum audio duration in seconds.")
parser.add_argument("--sampling_rate", type=int, default=16000, help="Target sampling rate for audio.")
parser.add_argument("--output_dir", type=str, default="./preprocessed/combined",
                    help="Output directory to save the preprocessed dataset.")
args = parser.parse_args()


def prepare_dataset(batch, model, audio_processor, text_tokenizer):
    '''Prepare dataset by processing audio and text.'''
    audio = batch["audio"]
    batch = audio_processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**audio)
    batch["input_features"] = outputs.last_hidden_state.squeeze(0).cpu().numpy()

    batch["labels"] = text_tokenizer(batch["text"]).input_ids

    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    return batch

def filter_audio_length(dataset, duration_threshold):
    '''Filter audio samples longer than duration_threshold seconds.'''
    def is_audio_length_in_range(input_length):
        return input_length < duration_threshold
    return dataset.filter(is_audio_length_in_range, input_columns=["input_length"])
    
def make_asr_dataset(args: argparse.Namespace) -> None:
    audio_processor = AutoProcessor.from_pretrained(args.audio_processor)
    text_tokenizer = Wav2Vec2CTCTokenizer(args.text_tokenizer)
    model = Wav2Vec2BertModel.from_pretrained(args.audio_processor)
    model.eval() # to get deterministic hidden states and turn off dropout

    # Load dataset
    dataset = load_dataset(args.path, args.name, split = args.split)

    # Standardize audio column to target sampling rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sampling_rate))

    if args.text_column != "text":
        dataset = dataset.rename_column(args.text_column, "text")

    dataset = dataset.map(lambda batch: prepare_dataset(batch, model, audio_processor, text_tokenizer), remove_columns=dataset.column_names)

    dataset = filter_audio_length(dataset, duration_threshold=args.max_duration)

    dataset.save_to_disk(args.output_dir)
    print(f"Saved {len(dataset)} samples → {args.output_dir}")

make_asr_dataset(args=args)

