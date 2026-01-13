"""
Audio tokenization using k-means clustering.
Extracts discrete audio tokens.
"""

import argparse
import numpy as np
from sklearn.cluster import KMeans
from datasets import load_from_disk, concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="facebook/w2v-bert-2.0")
parser.add_argument("--dataset_path", type=str, default="hf-internal-testing/librispeech_asr")
parser.add_argument("--num_clusters", type=int, default=1024)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--output_path", type=str, default="./discretized_dataset")
args = parser.parse_args()
def discretize_audio(args: argparse.Namespace):
    '''Discretize audio into discrete tokens using K-Means.'''

    datasets = args.dataset_path.split(",")
    datasets_list = []

    for ds in datasets:
        dataset = load_from_disk(ds)
        datasets_list.append(dataset)
    dataset = concatenate_datasets(datasets_list)

    feature_vectors = dataset["input_features"] #(seq_len, hidden_dim)

    # We need shape (n_frames, hidden_dim) for K-Means, so stack all feature vectors
    stacked_features = np.concatenate(feature_vectors, axis=0) #(total_frames, hidden_dim)

    # K-Means clustering on all features
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42)
    kmeans.fit(stacked_features)

    def assign_clusters(batch):
        '''Assign cluster labels to each feature vector/frame in the batch.'''
        batch["cluster"] = [kmeans.predict(feature) for feature in batch["input_features"]] # List of arrays
        return batch

    # Assign clusters to each feature vector in the dataset
    dataset = dataset.map(assign_clusters, batched=True, batch_size=args.batch_size)

    dataset.save_to_disk(args.output_path)




