import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
distance = np.linspace(0, 1000, 1000)  # in meters
frequency = 0.1  # high frequency (oscillations per meter)
amplitude = 0.5
noise_level = 0.1

# Generate sinusoidal signal with noise
signal = amplitude * np.sin(2 * np.pi * frequency * distance)
noisy_signal = signal + np.random.normal(0, noise_level, size=distance.shape)

# Add ammonia absorption zones
alpha = np.zeros_like(distance)
ammonia_zones = [(200, 300), (500, 600), (800, 850)]
for start, end in ammonia_zones:
    mask = (distance >= start) & (distance <= end)
    alpha[mask] = 0.2
    noisy_signal[mask] *= (1 - alpha[mask])  # attenuate

# Save CSV
df = pd.DataFrame({
    "Distance_m": distance,
    "Noisy_Signal": noisy_signal,
    "Alpha": alpha
})
df.to_csv("high_freq_ammonia_simulated.csv", index=False)

# Plot for inspection
plt.figure(figsize=(12, 4))
plt.plot(distance, noisy_signal, label="Noisy Signal", color="orange")
plt.plot(distance, alpha * 2, label="Alpha (x2)", linestyle="--", color="red")
plt.title("Simulated High-Frequency DIAL Signal with Ammonia Absorption")
plt.xlabel("Distance (m)")
plt.ylabel("Signal Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
