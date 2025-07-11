# plot_inference_results.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(csv_path):
    df = pd.read_csv(csv_path)

    if "Alpha" not in df.columns or "Predicted_Alpha" not in df.columns:
        raise ValueError("Missing required columns in CSV: 'Alpha' and/or 'Predicted_Alpha'.")

    plt.figure(figsize=(10, 6))
    plt.plot(df["Alpha"], label="True Alpha", linewidth=2)
    plt.plot(df["Predicted_Alpha"], label="Predicted Alpha", linestyle="--")
    plt.title("Predicted vs True Absorption Coefficient (Alpha)")
    plt.xlabel("Sample Index")
    plt.ylabel("Alpha")
    plt.legend()
    plt.grid(True)

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/inference_vs_truth.png")
    plt.close()
    print("Plot saved to plots/inference_vs_truth.png")

if __name__ == "__main__":
    plot_results("predictions/inference_results.csv")
