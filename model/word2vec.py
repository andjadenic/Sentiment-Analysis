import gensim.downloader as api
import numpy as np
import torch
import torch.nn as nn


if __name__ == "__main__":
    word2vec = api.load('word2vec-google-news-300')

    word2idx = {"<pad>": 0, "<unk>": 1, "king": 2, "queen": 3, "computer": 4}

    embedding_dim = word2vec.vector_size
    vocab_size = len(word2idx)

    # Initialize with random normal vectors
    embedding_matrix = np.random.normal(0, 1, (vocab_size, embedding_dim))
    # Fill with pretrained vectors
    for word, idx in word2idx.items():
        if word in word2vec:
            embedding_matrix[idx] = word2vec[word]
        # else: leave as random

    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

    # Create embedding layer initialized with pretrained weights
    embedding_layer = nn.Embedding.from_pretrained(
        embedding_matrix,
        freeze=True   # freeze=True = do not train embeddings
    )