import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import gensim.downloader as api
from SentimentAnalysis.utils.config import *


class LSTM_Classifier(nn.Module):
    def __init__(self, word2id):
        super(LSTM_Classifier, self).__init__()

        # Load pre-trained word2vec model
        word2vec = api.load('word2vec-google-news-300')

        self.embedding_dim = word2vec.vector_size
        self.vocab_size = len(word2id)

        embedding_matrix = torch.zeros(self.vocab_size, self.embedding_dim, dtype=torch.float32)
        no_embedding = 0
        for word, id in word2id.items():
            if word in word2vec:
                embedding_matrix[id] = torch.tensor(word2vec[word])
            else:  # leave as zero
                no_embedding += 1
        print(f'{no_embedding} words don\'t have embedding')

        # Create embedding layer initialized with pretrained weights
        self.embedding_layer = nn.Embedding.from_pretrained(
            embedding_matrix.float(),
            freeze=True  # freeze=True = do not train embeddings
        )

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=True,
                            batch_first=True,
                            dropout=dropout)
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=1,
                                bias=True)
        print('LSTM Classifier model is successfully built.')


    def forward(self, batch_texts, batch_len):
        '''
        batch_texts: (Nb, L) tensor
            of Nb embedded and padded reviews
        batch_lengths: (Nb, ) tensor
            of lengths of reviews
        '''
        Nb = batch_texts.shape[0]
        embedded_batch = self.embedding_layer(batch_texts)  # (Nb, L, embedding_dim)
        pack_batch = pack_padded_sequence(embedded_batch,
                                          batch_len,
                                          batch_first=True,
                                          enforce_sorted=False)
        out, (h, c) = self.lstm(pack_batch)  # (Nb, L, hidden_size)
        lstm_out = h[-1]  # (Nb, hidden_size)

        out_logits = self.linear(lstm_out)  # (Nb, 1)
        out_probs = nn.functional.sigmoid(out_logits)  # (Nb, 1)
        return out_probs.squeeze(1)  # (Nb, )