from model.LSTM_Classifier import *
from utils.config import *
from preprocessing.preprocessing import *
from torch.utils.data import DataLoader
import time
from itertools import product
import json

def evaluate_model(test_ds, embedding_matrix, path, hidden_size, Nb):
    # Define test DataLoader
    test_loader = DataLoader(test_ds, batch_size=Nb, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             collate_fn=collate_fn)

    # Load trained model
    model = LSTM_Classifier(embedding_matrix, hidden_size).to(device)
    model.load_state_dict(torch.load(path))

    # Define loss
    criterion = nn.BCELoss()

    # Evaluate the model
    print('Evaluation starts.')

    # Start timer
    start_time = time.time()

    info = {'avg_test_loss': [],
            'test_acc': [],
            'time': 0}

    # ---- Evaluation ----
    model.eval()
    epoch_test_loss, epoch_test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for padded, labels_tensor, lengths in test_loader:
            # Move batch to device
            padded = padded.to(device, non_blocking=True)
            labels_tensor = labels_tensor.to(device, non_blocking=True)

            batch_size = padded.shape[0]

            outputs = model(padded, lengths)  # (Nb,)
            batch_loss = criterion(outputs, labels_tensor.float())

            epoch_test_loss += batch_loss.item() * batch_size
            batch_preds = (outputs >= 0.5).long()
            epoch_test_correct += (batch_preds == labels_tensor).sum().item()
            test_total += batch_size

    avg_test_loss = epoch_test_loss / test_total
    info['avg_test_loss'] = avg_test_loss
    test_acc = epoch_test_correct / test_total
    info['epoch_test_acc'] = test_acc

    # End timer
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    testing_time = end_time - start_time
    info['time'] = testing_time
    print(f'Model evaluation has been successfully completed in {testing_time / 60:.2f} minutes.')
    print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Save evaluation info of best model
    with open(best_model_eval_info_path, "w", encoding="utf8") as f:
        json.dump(info, f, indent=4)

    return info
