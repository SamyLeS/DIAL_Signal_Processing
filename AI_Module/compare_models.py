# compare_models.py

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# ----- MLP Definition -----
class MLPRegressor(nn.Module):
    def __init__(self, input_dim=2):
        super(MLPRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ----- LSTM Definition -----
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1, output_dim=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ----- Sequence creator -----
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ----- Run comparison -----
def compare_models(csv_path, mlp_path, lstm_path, seq_len=20):
    df = pd.read_csv(csv_path)

    # === Prepare data for MLP ===
    df["Smoothed"] = df["Noisy_Signal"].rolling(5).mean().bfill()
    X_mlp = df[["Noisy_Signal", "Smoothed"]].values
    y_all = df["Alpha"].values

    X_mlp_tensor = torch.tensor(X_mlp, dtype=torch.float32)

    mlp_model = MLPRegressor(input_dim=2)
    mlp_model.load_state_dict(torch.load(mlp_path))
    mlp_model.eval()

    with torch.no_grad():
        y_mlp_pred = mlp_model(X_mlp_tensor).numpy()

    # === Prepare data for LSTM ===
    signal = df["Noisy_Signal"].values.reshape(-1, 1)
    X_lstm, y_lstm_true = create_sequences(signal, seq_len)
    X_lstm_tensor = torch.tensor(X_lstm, dtype=torch.float32)

    lstm_model = LSTMRegressor()
    lstm_model.load_state_dict(torch.load(lstm_path))
    lstm_model.eval()

    with torch.no_grad():
        y_lstm_pred = lstm_model(X_lstm_tensor).numpy()

    # === Plot comparison ===
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(y_all[:len(y_lstm_pred)], label="True Alpha", linewidth=2)
    plt.plot(y_mlp_pred[:len(y_lstm_pred)], label="MLP Prediction", linestyle='--')
    plt.plot(y_lstm_pred, label="LSTM Prediction", linestyle=':')
    plt.title("MLP vs LSTM on Noisy DIAL Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Alpha")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/mlp_vs_lstm_comparison.png")
    plt.close()
    print("Comparison plot saved to plots/mlp_vs_lstm_comparison.png")

# Run
if __name__ == "__main__":
    compare_models(
        csv_path="high_freq_ammonia_simulated.csv",
        mlp_path="models/trained_model.pt",
        lstm_path="LSTM/models/lstm_model.pt"
    )
