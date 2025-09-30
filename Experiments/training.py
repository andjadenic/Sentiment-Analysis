import torch.optim as optim
from SentimentAnalysis.model.LSTM_Classifier import *
from SentimentAnalysis.preprocessing.preprocessing import *


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv(csv_data_path)

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

    # Reload built vocab
    id2token, token2id = reload_built_vocab(json_vocab_path)
    vocab_size = len(id2token)

    # Encode text
    encoded_texts = [encode(tokenized_review, token2id) for tokenized_review in tokenized_reviews]

    # Split the dataset into train, validation and test datasets
    train_texts, train_labels = encoded_texts[:25000], labels[:25000]
    val_texts, val_labels = encoded_texts[25000:35000], labels[25000:35000]
    test_texts, test_labels = encoded_texts[35000:], labels[35000:]

    # Make Datasets and DataLoaders
    train_ds = TextDataset(train_texts, train_labels)
    val_ds = TextDataset(val_texts, val_labels)
    test_ds = TextDataset(test_texts, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn)

    # Define Model
    model = LSTM_Classifier()

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for epoch in range(N_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for embedded, lengths, labels in train_loader:
            optimizer.zero_grad()

            # Forward
            outputs = model(embedded, lengths)  # (batch_size,)

            # Loss
            labels = labels.float()  # BCE expects float labels
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            # Stats
            epoch_loss += loss.item() * labels.size(0)
            preds = (outputs >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

        acc = correct / total
        avg_loss = epoch_loss / total
        print(f"Epoch [{epoch + 1}/{N_epochs}] Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
