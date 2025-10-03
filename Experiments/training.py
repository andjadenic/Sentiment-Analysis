import torch.optim as optim
from SentimentAnalysis.model.LSTM_Classifier import *
from SentimentAnalysis.preprocessing.preprocessing import *
from torch.utils.data import DataLoader
import time
from SentimentAnalysis.Experiments.evaluation import *
from itertools import product
import json


def train_model(embedding_matrix, train_ds, val_ds, hidden_size, Nb, lr, training_model_path):
    # Define train and validation DataLoaders with given batch size Nb
    train_loader = DataLoader(train_ds, batch_size=Nb, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=Nb, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            collate_fn=collate_fn)

    # Define Model with given hidden_size
    model = LSTM_Classifier(embedding_matrix, hidden_size).to(device)

    # Define loss and optimizer with given learning rate lr
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    print('Training starts.')
    info = {
            'hidden_size': hidden_size,
            'batch_size': Nb,
            'lr': lr,
            'epoch_avg_train_loss': [],
            'epoch_train_acc': [],
            'epoch_avg_val_loss': [],
            'epoch_val_acc': [],
            'time': 0}

    # Start timer
    start_time = time.time()

    # Monitor validation loss and accuracy for early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(N_epochs):
        # ---- Training ----
        model.train()
        epoch_train_loss, epoch_train_correct, train_total = 0.0, 0, 0

        for padded, labels_tensor, lengths in train_loader:
            # Move batch to device
            padded = padded.to(device, non_blocking=True)
            labels_tensor = labels_tensor.to(device, non_blocking=True)

            batch_size = padded.shape[0]  # current batch size
            optimizer.zero_grad()

            # Forward
            outputs = model(padded, lengths)  # (Nb,)

            # Loss
            batch_loss = criterion(outputs, labels_tensor.float())

            # Backward
            batch_loss.backward()
            optimizer.step()

            # Model's predictions
            epoch_train_loss += batch_loss.item() * batch_size
            batch_preds = (outputs >= 0.5).long()
            epoch_train_correct += (batch_preds == labels_tensor).sum().item()
            train_total += batch_size

        epoch_avg_train_loss = epoch_train_loss / train_total  # average loss per sample across the whole training dataset in current epoch
        info['epoch_avg_train_loss'].append(epoch_avg_train_loss)
        epoch_train_acc = epoch_train_correct / train_total  # accuracy on training data in current epoch
        info['epoch_train_acc'].append(epoch_train_acc)

        # ---- Validation ----
        model.eval()
        epoch_val_loss, epoch_val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for padded, labels_tensor, lengths in val_loader:
                padded = padded.to(device, non_blocking=True)
                labels_tensor = labels_tensor.to(device, non_blocking=True)

                batch_size = padded.shape[0]

                outputs = model(padded, lengths)  # (Nb,)
                batch_loss = criterion(outputs, labels_tensor.float())

                epoch_val_loss += batch_loss.item() * batch_size
                batch_preds = (outputs >= 0.5).long()
                epoch_val_correct += (batch_preds == labels_tensor).sum().item()
                val_total += batch_size

        epoch_avg_val_loss = epoch_val_loss / val_total
        info['epoch_avg_val_loss'].append(epoch_avg_val_loss)
        epoch_val_acc = epoch_val_correct / val_total
        info['epoch_val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch + 1}/{N_epochs} | "
              f"Train Loss: {epoch_avg_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_avg_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}", "\n")

        # ---- Early Stopping ----
        if epoch_avg_val_loss < best_val_loss:
            best_val_loss = epoch_avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), training_model_path)  # save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # Load the best weights before returning
    model.load_state_dict(torch.load(training_model_path))

    # End timer
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    training_time = end_time - start_time
    info['time'] = training_time
    print(f'Model training has been successfully completed in {training_time / 60:.2f} minutes.')
    print('\n')
    print('\n')

    return info


def grid_search(param_grid, embedding_matrix, train_ds, val_ds):
    info_list = []
    for hidden_size, batch_size, lr in product(*param_grid.values()):
        curr_info = train_model(embedding_matrix, train_ds, val_ds,
                                hidden_size, batch_size, lr,
                                training_model_path)
        info_list.append(curr_info)

    # Save info about trained models
    with open(all_models_info_path, "w", encoding="utf-8") as f:
        json.dump(info_list, f, indent=4)


if __name__ == '__main__':
    # Preprocess data
    word2id, encoded_texts, labels = preprocess_data(csv_data_path, json_vocab_path)

    # Create embedding matrix
    embedding_matrix = make_embedding_matrix(word2id)

    # Split the dataset into train, validation and test datasets
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_data(encoded_texts, labels,
                                                                                           .5, .3)
    # Make Datasets
    train_ds, val_ds = TextDataset(train_texts, train_labels), TextDataset(val_texts, val_labels)

    # Train the model
    '''info = train_model(embedding_matrix, train_ds, val_ds,
                       hidden_size, batch_size, lr,
                       training_model_path)'''

    # Tune hyperparameters
    '''grid_search(param_grid, embedding_matrix, train_ds, val_ds)
    with open(all_models_info_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    print(loaded_data)'''

    # Evaluate the model that best performed on validation dataset
    # batch_size, hidden_sizeare set in config
    # model parameters are saved in best_model_path
    test_ds = TextDataset(test_texts, test_labels)
    eval_info = evaluate_model(test_ds, embedding_matrix, best_model_path, hidden_size, batch_size)