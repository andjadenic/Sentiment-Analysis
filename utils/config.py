csv_path = r'data/IMDB Dataset.csv' # path to downloaded CSV

max_seq_len = 1000 # truncate very long reviews
min_freq = 2
batch_size = 64
#num_workers = 4
pad_token = '<PAD>'
unk_token = "<UNK>"
pad_id = 0
unk_id = 1