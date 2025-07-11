# run_inference.py

import os
import pandas as pd
import torch
import torch.nn as nn

# Modèle identique à celui utilisé pour l'entraînement
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

# Fonction d'inférence
def run_inference(csv_path, model_path="models/trained_model.pt"):
    df = pd.read_csv(csv_path)

    # Feature additionnelle comme dans le training
    df["Smoothed"] = df["Noisy_Signal"].rolling(5).mean().bfill()

    # Préparation des données
    X = df[["Noisy_Signal", "Smoothed"]].values
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Chargement du modèle
    model = SimpleRegressor(input_dim=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    df["Predicted_Alpha"] = predictions

    os.makedirs("predictions", exist_ok=True)
    df.to_csv("predictions/inference_results.csv", index=False)
    print("Inference complete. Results saved to predictions/inference_results.csv")

# Exécution
if __name__ == "__main__":
    run_inference("high_freq_ammonia_simulated.csv")
