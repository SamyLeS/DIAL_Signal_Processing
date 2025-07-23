import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
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


def create_sequences(x, y, seq_len):
    X, Y = [], []
    for i in range(len(x) - seq_len):
        X.append(x[i:i+seq_len])
        Y.append(y[i+seq_len])
    return np.array(X), np.array(Y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--seq_len', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--out', default='models_v2/lstm_v2.pt')
    args = p.parse_args()

    # ensure output folder exists
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Load extended dataset
    df = pd.read_csv(args.csv)
    x = df['Noisy_Signal'].values.reshape(-1, 1)
    y = df['Alpha'].values
    X, Y = create_sequences(x, y, args.seq_len)

    # Train/val split
    N = len(X)
    n_val = int(0.2 * N)
    train_X, val_X = X[:-n_val], X[-n_val:]
    train_Y, val_Y = Y[:-n_val], Y[-n_val:]

    train_ds = TensorDataset(torch.tensor(train_X, dtype=torch.float32),
                             torch.tensor(train_Y, dtype=torch.float32))
    val_ds   = TensorDataset(torch.tensor(val_X, dtype=torch.float32),
                             torch.tensor(val_Y, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # Model, loss, optimizer
    model = LSTMRegressor(dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss, patience_ctr = float('inf'), 0

    # History for plotting
    history = {'epoch': [], 'val_loss': []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred.squeeze(), by)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = np.mean([
            criterion(model(bx).squeeze(), by).item()
            for bx, by in val_loader
        ])
        print(f"Epoch {epoch}/{args.epochs} - Val Loss: {val_loss:.6f}")

        # record history
        history['epoch'].append(epoch)
        history['val_loss'].append(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss, patience_ctr = val_loss, 0
            torch.save(model.state_dict(), args.out)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best val loss: {best_loss:.6f}. Model saved to {args.out}.")

    # --- Plot validation loss curve ---
    plt.figure(figsize=(8, 4))
    plt.plot(history['epoch'], history['val_loss'], '-o', label='Val Loss')
    plt.title('LSTM v2 Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.legend()
    curve_path = os.path.join(out_dir, 'loss_curve_v2.png')
    plt.savefig(curve_path)
    plt.close()
    print(f"Validation loss curve saved to {curve_path}")


if __name__ == '__main__':
    main()
