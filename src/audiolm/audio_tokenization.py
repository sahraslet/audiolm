"""
Audio tokenization using k-means clustering.
Extracts discrete audio tokens.
"""

import argparse
import numpy as np
from sklearn.cluster import KMeans
from datasets import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="facebook/w2v-bert-2.0")
parser.add_argument("--dataset_path", type=str, default="hf-internal-testing/librispeech_asr")
parser.add_argument("--num_clusters", type=int, default=1024)
args = parser.parse_args()
def discretize_audio(args: argparse.Namespace):
    '''Discretize audio into discrete tokens using K-Means.'''

    dataset = load_from_disk(args.dataset_path)
    feature_vectors = dataset["input_features"] #(seq_len, hidden_dim)

    # We need shape (n_frames, hidden_dim) for K-Means, so stack all feature vectors
    stacked_features = np.concatenate(feature_vectors, axis=0) #(total_frames, hidden_dim)

    # K-Means clustering on all features
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42)
    kmeans.fit(stacked_features)

    def assign_clusters(batch):
        '''Assign cluster labels to each feature vector/frame in the batch.'''
        batch["cluster"] = kmeans.predict(batch["input_features"])
        return batch

    # Assign clusters to each feature vector in the dataset
    dataset = dataset.map(assign_clusters, batched=True)

    return dataset




