import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define a slightly deeper regression model
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim):
        super(SimpleRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# Load and prepare data
def load_data(path):
    df = pd.read_csv(path)

    # Additional feature (optional)
    df["Smoothed"] = df["Noisy_Signal"].rolling(5).mean().bfill()


    # Select features and target
    X = df[["Noisy_Signal", "Smoothed"]].values
    y = df["Alpha"].values.reshape(-1, 1)

    return train_test_split(X, y, test_size=0.2, random_state=42)

# Plot training loss
def plot_training_loss(losses):
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)
    plt.savefig("plots/training_loss.png")
    plt.close()

# Plot prediction vs truth
def plot_prediction_vs_truth(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    plt.figure(figsize=(8, 6))
    plt.plot(y_test, label="True Alpha", linewidth=2)
    plt.plot(predictions, label="Predicted Alpha", linestyle="--")
    plt.title("Prediction vs Ground Truth")
    plt.xlabel("Sample Index")
    plt.ylabel("Alpha")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/prediction_vs_truth.png")
    plt.close()

# Train model
def train_model(path):
    X_train, X_test, y_train, y_test = load_data(path)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = SimpleRegressor(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    epochs = 100
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
    torch.save(model.state_dict(), "models/trained_model.pt")
    print("Model saved to models/trained_model.pt")

    # Plot and save
    plot_training_loss(losses)
    plot_prediction_vs_truth(model, X_test, y_test)

# Run
if __name__ == "__main__":
    train_model("high_freq_ammonia_simulated.csv")
