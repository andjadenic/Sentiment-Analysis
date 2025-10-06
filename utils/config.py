import torch

# Paths
csv_data_path = r'C:\Users\HP\Sentiment-Analysis\data\IMDB_Dataset.csv'
json_vocab_path = r'C:\Users\HP\Sentiment-Analysis\preprocessing\vocab.json'

training_model_path = r'C:\Users\HP\Sentiment-Analysis\model\training_model.pth'
all_models_info_path = r'C:\Users\HP\Sentiment-Analysis\experiments\models_info.json'

best_model_eval_info_path = r'C:\Users\HP\Sentiment-Analysis\experiments\best_model_eval_info.json'
best_model_path = r'C:\Users\HP\Sentiment-Analysis\model\best_model.pth'

# Review preprocessing
max_seq_len = 750  # truncate very long reviews / hyperparameter
min_freq = 2
pad_token = '<pad>'
unk_token = '<unk>'
pad_id = 0
unk_id = 1

# Word embedding
embedding_dim = 300

# LSTM
hidden_size = 256  # tuned hyperparameter
num_layers = 2    # hyperparameter
dropout = 0.3  # hyperparameter

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64  # tuned hyperparameter
num_workers = 4
N_epochs = 10  # hyperparameter
lr = 1e-3    # tuned hyperparameter
patience = 4    # hyperparameter

# Define search space for hyperparameter tuning
param_grid = {
    'hidden_size': [256, 512],
    'batch_size': [64, 128],
    'lr': [1e-4, 1e-3, 1e-2]
}

