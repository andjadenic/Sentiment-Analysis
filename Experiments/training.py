import torch.optim as optim
from SentimentAnalysis.model.LSTM_Classifier import *
from SentimentAnalysis.preprocessing.preprocessing import *


if __name__ == '__main__':
    # Preprocess data
    encoded_texts, labels = preprocess_data(csv_data_path, json_vocab_path)

    # Split the dataset into train, validation and test datasets
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_data(encoded_texts, labels,
                                                                                           .5, .3)
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

    for i, (padded, labels_tensor, lengths) in enumerate(train_loader):
        if i == 0:
            print(padded.shape)
            print(padded)
        else:
            break

    # Define Model
    model = LSTM_Classifier()

    '''
    WORD2VEC
    '''

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    '''print('Training starts.')
    for epoch in range(N_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for padded, labels_tensor, lengths in train_loader:
            Nb = padded.shape[0]
            padded: (Nb, L) - numericized and padded words. need embedding
            labels_tensor: (Nb, )
            lengths: (Nb, )
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(padded, lengths)  # (batch_size,)

            # Loss
            labels_tensor = labels_tensor.float()  # BCE expects float labels
            loss = criterion(outputs, labels_tensor)

            # Backward
            loss.backward()
            optimizer.step()

            # Stats
            epoch_loss += loss.item() * labels_tensor.size(0)
            preds = (outputs >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

        acc = correct / total
        avg_loss = epoch_loss / total
        print(f"Epoch [{epoch + 1}/{N_epochs}] Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
    print('Model training has been successfully completed.')

    # Save model parameters
    torch.save(model.state_dict(), trained_model_path)'''
