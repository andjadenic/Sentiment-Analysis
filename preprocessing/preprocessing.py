# save as imdb_preproc.py and run with: python imdb_preproc.py
from SentimentAnalysis.utils.config import *
import re
import json
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence



def try_read_csv(path):
    """Try sensible encodings for IMDB CSVs."""
    for enc in ("utf-8", "latin1", "iso-8859-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError("Failed to read CSV with common encodings.")


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    # remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # keep letters, numbers and apostrophes, replace other chars with space
    text = re.sub(r"[^A-Za-z0-9'\s]", " ", text)
    # collapse whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def tokenize(text):
    # keep contractions like don't, i'm, it's
    return re.findall(r"[A-Za-z0-9']+", text)



if __name__ == "__main__":
    # Load the dataset
    df = try_read_csv(csv_path)
    
    # Clean and tokenize the train data (single pass)
    token_counter = Counter()
    tokenized_reviews = []  # list of tokenized reviews
    labels = []  # list of labels (coded sentiments)
    
    for id, row in df.iterrows():
        text = clean_text(row['review'])
        tokens = tokenize(text)

        tokens = tokens[:max_seq_len]  # truncate reviews
        tokenized_reviews.append(tokens)
        token_counter.update(tokens)

        label = 1 if row['sentiment'] == 'positive' else 0
        labels.append(label)


    # Build vocab
    '''most_common = [tok for tok, freq in token_counter.most_common() if freq >= min_freq]
    tokens_list = [pad_token, unk_token] + most_common
    token2id = {tok: i for i, tok in enumerate(tokens_list)}  # token -> index
    
    vocab_size = len(tokens_list)
    print("Vocab size:", vocab_size)'''

    # Save vocab
    '''with open("preprocessing/vocab.json", "w", encoding="utf8") as f:
        json.dump({"tokens_list": tokens_list}, f, ensure_ascii=False)'''

    # Reload built vocab
    with open("vocab.json", "r", encoding="utf8") as f:
        vocab = json.load(f)
    id2token = vocab["tokens_list"]
    token2id = {tok: i for i, tok in enumerate(id2token)}
    vocab_size = len(id2token)

    # Encode texts
    def encode(tokenized_review, token2id, unk_id=unk_id):
        return [token2id.get(t, unk_id) for t in tokenized_review]
    
    encoded_texts = [encode(tokenized_review, token2id) for tokenized_review in tokenized_reviews]

    # Split the dataset into train, validation and test datasets
    train_texts, train_labels = encoded_texts[:25000], labels[:25000]
    val_texts, val_labels = encoded_texts[25000:35000], labels[25000:35000]
    test_texts, test_labels = encoded_texts[35000:], labels[35000:]

    # Make Datasets
    class TextDataset(Dataset):
        def __init__(self, texts, labels=None):
            self.texts = texts
            self.labels = labels
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, id):
            return self.texts[id], self.labels[id]

    train_ds = TextDataset(train_texts, train_labels)
    val_ds = TextDataset(val_texts, val_labels)
    test_ds = TextDataset(test_texts, test_labels)


    def collate_fn(batch):
            """Return (padded_inputs, labels_tensor, lengths) or (padded_inputs, lengths) when no labels."""

            texts, labs = zip(*batch)

            lengths = torch.tensor([len(x) for x in texts], dtype=torch.long)
            tensors = [torch.tensor(x, dtype=torch.long) for x in texts]
            padded = pad_sequence(tensors, batch_first=True, padding_value=pad_id)

            labels_tensor = torch.tensor(labs, dtype=torch.long)
            return padded, labels_tensor, lengths



    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)
