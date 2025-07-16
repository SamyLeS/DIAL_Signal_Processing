import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ----------- Data Preparation -----------
def create_sequences(data, targets, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(targets[i+seq_len])
    return np.array(X), np.array(y)

def load_lstm_data(path, seq_len=10):
    df = pd.read_csv(path)
    signal = df['Noisy_Signal'].values
    alpha = df['Alpha'].values

    X, y = create_sequences(signal, alpha, seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM expects 3D input
    y = y.reshape(-1, 1)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ----------- LSTM Model -----------
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ----------- Training & Plotting -----------
def plot_training_loss(losses):
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)
    plt.savefig("plots/lstm_training_loss.png")
    plt.close()

def plot_prediction_vs_truth(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(y_test, label='True Alpha', linewidth=2)
    plt.plot(preds, label='Predicted Alpha', linestyle='--')
    plt.title("Prediction vs Ground Truth (LSTM)")
    plt.xlabel("Sample Index")
    plt.ylabel("Alpha")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/lstm_prediction_vs_truth.png")
    plt.close()

# ----------- Main training function -----------
def train_lstm_model(path, seq_len=10, epochs=100):
    X_train, X_test, y_train, y_test = load_lstm_data(path, seq_len)

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = LSTMRegressor(input_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_model.pt")
    print("Model saved to models/lstm_model.pt")

    # Plot results
    plot_training_loss(losses)
    plot_prediction_vs_truth(model, X_test, y_test)

# ----------- Run script -----------
if __name__ == "__main__":
    train_lstm_model("../high_freq_ammonia_simulated.csv", seq_len=20)

