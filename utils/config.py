# Dataset
csv_data_path = r'D:\Faks\MASTER\PyTorch\Sentiment Analysis\SentimentAnalysis\data\IMDB Dataset.csv'
json_vocab_path = r'D:\Faks\MASTER\PyTorch\Sentiment Analysis\SentimentAnalysis\preprocessing\vocab.json'

# Review preprocessing
max_seq_len = 1000  # truncate very long reviews
min_freq = 2
pad_token = '<PAD>'
unk_token = "<UNK>"
pad_id = 0
unk_id = 1

# Word embedding
embedding_dim = 300

# LSTM
hidden_size = 128
num_layers = 4
dropout = 0.0

# Training
batch_size = 4
#num_workers = 4
N_epochs = 10
lr = 1e-3
