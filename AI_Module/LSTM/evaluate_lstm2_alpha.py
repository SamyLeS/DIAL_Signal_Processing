#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_sequences(signal, seq_len):
    """Convert 1D signal into [N-seq_len, seq_len, 1] windows."""
    X = []
    for i in range(len(signal) - seq_len):
        X.append(signal[i:i+seq_len])
    return np.array(X)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',     required=True, help="Extended dataset CSV")
    p.add_argument('--model',   required=True, help="Path to LSTM v2 .pt file")
    p.add_argument('--seq_len', type=int, default=20)
    p.add_argument('--out_dir', default='eval_results')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load data + model
    df = pd.read_csv(args.csv)
    signal = df["Noisy_Signal"].values.reshape(-1,1).astype(np.float32)
    true_alpha = df["Alpha"].values.astype(np.float32)
    noise_levels = df["Noise_Level"].values

    model = LSTMRegressor()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # 2. Build sequences & run inference
    X = create_sequences(signal, args.seq_len)
    with torch.no_grad():
        preds = model(torch.tensor(X)).numpy().flatten()

    # 3. Pad predictions back to full length
    pred_alpha = np.full_like(true_alpha, np.nan, dtype=np.float32)
    pred_alpha[args.seq_len:] = preds
    df["Predicted_Alpha"] = pred_alpha

    # 4. Compute metrics per noise level
    records = []
    for lvl in np.unique(noise_levels):
        mask = (noise_levels == lvl) & (np.arange(len(df)) >= args.seq_len)
        t = true_alpha[mask]
        p = pred_alpha[mask]
        mse = np.mean((p - t)**2)
        mae = np.mean(np.abs(p - t))
        records.append({"Noise_Level": lvl, "MSE": mse, "MAE": mae})
    metrics_df = pd.DataFrame(records).set_index("Noise_Level")
    metrics_df.to_csv(os.path.join(args.out_dir, "metrics_by_noise.csv"))

    # 5. Plot True vs Predicted α (overall)
    plt.figure(figsize=(10,5))
    plt.plot(true_alpha,    label="True α", alpha=0.5)
    plt.plot(pred_alpha,    label="Predicted α", alpha=0.8)
    plt.title("True vs. Predicted Absorption Coefficient α (All Noise Levels)")
    plt.xlabel("Sample Index")
    plt.ylabel("α [1/m]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "true_vs_predicted_overall.png"))
    plt.close()

    # 6. Plot per noise level (overlay)
    plt.figure(figsize=(10,5))
    for lvl in metrics_df.index:
        mask = (noise_levels == lvl) & (np.arange(len(df)) >= args.seq_len)
        plt.scatter(np.where(mask)[0], pred_alpha[mask], s=4, label=f"Predicted, {lvl}")
        plt.scatter(np.where(mask)[0], true_alpha[mask], s=4, marker='x', label=f"True, {lvl}")
    plt.title("Predicted vs True α by Noise Level")
    plt.xlabel("Sample Index")
    plt.ylabel("α [1/m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "true_vs_predicted_by_noise.png"))
    plt.close()

    print("Evaluation complete.")
    print("Metrics per noise level:")
    print(metrics_df)
    print(f"Results saved under: {args.out_dir}")

if __name__ == '__main__':
    main()
