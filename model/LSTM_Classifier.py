import torch.nn.functional

from SentimentAnalysis.preprocessing.preprocessing import *
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from SentimentAnalysis.utils.config import *


class LSTM_Classifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_size):
        super(LSTM_Classifier, self).__init__()

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
        out_probs = torch.nn.functional.sigmoid(out_logits) # (Nb, 1)
        return out_probs.squeeze(1)  # (Nb, )