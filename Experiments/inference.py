
if __name__ == '__main__':
    # Recreate the model structure
    model = LSTM_classifier()  # same architecture as before
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()  # set to evaluation mode if you’re using it for inference
