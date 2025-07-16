# run_lstm_inference.py

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# LSTM model with 1 layer (compatible with the saved model)
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1, output_dim=1):  # ← num_layers=1
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use last time step
        return out

# Function to create sequences from raw signal
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Inference logic
def run_inference(csv_path, model_path, seq_len=20):
    # Load signal and target
    df = pd.read_csv(csv_path)
    signal = df["Noisy_Signal"].values.reshape(-1, 1)
    alpha = df["Alpha"].values.reshape(-1, 1)

    X, y = create_sequences(signal, seq_len)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_true = y  # for plotting

    # Load model
    model = LSTMRegressor()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        y_pred = model(X_tensor).numpy()

    # Plot predictions vs truth
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="True Alpha", linewidth=2)
    plt.plot(y_pred, label="Predicted Alpha", linestyle="--")
    plt.title("LSTM Inference: Prediction vs Truth")
    plt.xlabel("Sample Index")
    plt.ylabel("Alpha")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/lstm_prediction_vs_truth.png")
    plt.close()
    print("Inference complete. Plot saved to plots/lstm_prediction_vs_truth.png")

# Run the script
if __name__ == "__main__":
    run_inference("../high_freq_ammonia_simulated.csv", "models/lstm_model.pt", seq_len=20)
