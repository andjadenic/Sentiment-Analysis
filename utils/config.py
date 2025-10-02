import torch

# Paths
csv_data_path = r'D:\Faks\MASTER\PyTorch\Sentiment Analysis\SentimentAnalysis\data\IMDB Dataset.csv'
json_vocab_path = r'D:\Faks\MASTER\PyTorch\Sentiment Analysis\SentimentAnalysis\preprocessing\vocab.json'
training_model_path = r'D:\Faks\MASTER\PyTorch\Sentiment Analysis\SentimentAnalysis\model\training_model.pth'
best_model_path = r'D:\Faks\MASTER\PyTorch\Sentiment Analysis\SentimentAnalysis\model\best_model.pth'
all_models_info_path = r'D:\Faks\MASTER\PyTorch\Sentiment Analysis\SentimentAnalysis\Experiments\models_info.json'

# Review preprocessing
max_seq_len = 500  # truncate very long reviews / hyperparameter
min_freq = 2
pad_token = '<pad>'
unk_token = '<unk>'
pad_id = 0
unk_id = 1

# Word embedding
embedding_dim = 300

# LSTM
hidden_size = 128  # tuned hyperparameter
num_layers = 1    # hyperparameter
dropout = 0.0  # 0.3, hyperparameter

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64  # tuned hyperparameter
#num_workers = 4
N_epochs = 5  # hyperparameter
lr = 1e-3    # tuned hyperparameter
patience = 1000    # hyperparameter

# Define search space for hyperparameter tuning
param_grid = {
    'hidden_size': [256, 512],
    'batch_size': [64, 128],
    'lr': [1e-4, 1e-3, 1e-2]
}

