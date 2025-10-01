import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from SentimentAnalysis.utils.config import *


class LSTM_Classifier(nn.Module):
    def __init__(self):
        super(LSTM_Classifier, self).__init__()
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
        batch_texts: (Nb, L, embedded_dim) tensor
            of Nb embedded and padded reviews
        batch_lengths: (Nb, ) tensor
            of lengths of reviews
        '''
        Nb = batch_texts.shape[0]
        batch_input = pack_padded_sequence(batch_texts,
                                           batch_len,
                                           batch_first=True,
                                           enforce_sorted=False)
        h, _ = self.lstm(batch_input)  # (Nb, L, hidden_size)
        lstm_out = h[-1]

        out_logits = self.linear(lstm_out)  # (Nb, 1)
        out_probs = nn.functional.sigmoid(out_logits)  # (Nb, 1)
        return out_probs.squeeze(1)  # (Nb, )