from utils.config import *
import re
import json
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import gensim.downloader as api


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

def tokenize_text(text):
    # keep contractions like don't, i'm, it's
    return re.findall(r"[A-Za-z0-9']+", text)

def encode_text(tokenized_review, token2id, unk_id=unk_id):
    return [token2id.get(t, unk_id) for t in tokenized_review]


def build_vocab():
    # Load the dataset
    df = pd.read_csv(csv_data_path)

    # Clean and tokenize the data
    token_counter = Counter()
    tokenized_reviews = []  # list of tokenized reviews
    labels = []  # list of labels (coded sentiments)

    for id, row in df.iterrows():
        text = clean_text(row['review'])
        tokens = tokenize_text(text)

        tokens = tokens[:max_seq_len]  # truncate reviews
        tokenized_reviews.append(tokens)
        token_counter.update(tokens)

        label = 1 if row['sentiment'] == 'positive' else 0
        labels.append(label)

    most_common = [tok for tok, freq in token_counter.most_common() if freq >= min_freq]
    tokens_list = [pad_token, unk_token] + most_common
    token2id = {tok: i for i, tok in enumerate(tokens_list)}  # token -> index

    # Save vocab
    with open(json_vocab_path, "w", encoding="utf8") as f:
        json.dump({"tokens_list": tokens_list}, f, ensure_ascii=False)

    print(f'Vocabulary with {len(tokens_list)} words is successfully built and saved.')
    return tokens_list, token2id

def reload_built_vocab(json_vocab_path):
    with open(json_vocab_path, 'r', encoding='utf8') as f:
        vocab = json.load(f)
    id2token = vocab['tokens_list']
    token2id = {tok: i for i, tok in enumerate(id2token)}
    return id2token, token2id


def preprocess_data(csv_data_path, json_vocab_path):
    '''
    - Load the dataset from CSV file
    - Clean and tokenize the data
    - Reload built vocab
    - Encode reviews and sentimentd
    '''
    # Load the dataset
    df = pd.read_csv(csv_data_path)

    # Clean and tokenize the data
    token_counter = Counter()
    tokenized_reviews = []  # list of tokenized reviews
    labels = []  # list of labels (coded sentiments)

    for id, row in df.iterrows():
        text = clean_text(row['review'])
        tokens = tokenize_text(text)

        tokens = tokens[:max_seq_len]  # truncate reviews
        tokenized_reviews.append(tokens)
        token_counter.update(tokens)

        label = 1 if row['sentiment'] == 'positive' else 0
        labels.append(label)

    # Reload built vocab
    id2token, token2id = reload_built_vocab(json_vocab_path)

    # Encode text
    encoded_texts = [encode_text(tokenized_review, token2id) for tokenized_review in tokenized_reviews]
    print('Data is successfully preprocessed.')
    return token2id, encoded_texts, labels


def split_data(encoded_texts, labels, train_percentage, val_percentage):
    N = len(encoded_texts)

    N_train = int(N * train_percentage)
    N_val = int(N * val_percentage)

    train_texts, train_labels = encoded_texts[:N_train], labels[:N_train]
    val_texts, val_labels = encoded_texts[N_train:(N_train + N_val)], labels[N_train:(N_train + N_val)]
    test_texts, test_labels = encoded_texts[(N_train + N_val):], labels[(N_train + N_val):]
    print('Dataset is successfully split into train, validate and test subsets.')
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, id):
        return self.texts[id], self.labels[id]


def collate_fn(batch):
    """
    batch: list of (batch_text, batch_labels) tuples
    return: (padded_inputs, labels_tensor, lengths)
    """
    texts, labs = zip(*batch)

    lengths = torch.tensor([len(x) for x in texts], dtype=torch.long)
    tensors = [torch.tensor(x, dtype=torch.long) for x in texts]
    padded = pad_sequence(tensors, batch_first=True, padding_value=pad_id)
    labels_tensor = torch.tensor(labs, dtype=torch.long)
    return padded, labels_tensor, lengths


def make_embedding_matrix(word2id):
    # Load pre-trained word2vec model
    word2vec = api.load('word2vec-google-news-300')

    embedding_dim = word2vec.vector_size
    vocab_size = len(word2id)

    embedding_matrix = torch.zeros(vocab_size, embedding_dim, dtype=torch.float32)
    no_embedding = 0
    for word, id in word2id.items():
        if word in word2vec:
            embedding_matrix[id] = torch.tensor(word2vec[word])
        else:  # leave as zero
            no_embedding += 1
    print(f'{no_embedding} words don\'t have embedding')
    print('Embedding matrix is successfully built.')
    return embedding_matrix