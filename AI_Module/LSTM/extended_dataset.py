#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np

def build_extended_dataset(in_csv, out_csv):
    # 1) load the original simulated DIAL file
    df = pd.read_csv(in_csv)
    if 'Alpha' not in df.columns:
        raise KeyError("Input CSV must contain an 'Alpha' column with the true absorption values.")
    
    # 2) for each noise level, add gaussian noise to 'Alpha' → 'Noisy_Signal'
    noise_levels = [
        ('low',    0.005),
        ('medium', 0.01),
        ('high',   0.02),
    ]
    pieces = []
    for label, sigma in noise_levels:
        tmp = df.copy()
        tmp['Noisy_Signal'] = tmp['Alpha'] + np.random.normal(0, sigma, size=len(tmp))
        tmp['Noise_Level']  = label
        # keep only the four columns we need
        pieces.append(tmp[['Distance_m','Noisy_Signal','Alpha','Noise_Level']])
    
    # 3) concatenate & write out
    extended = pd.concat(pieces, ignore_index=True)
    extended.to_csv(out_csv, index=False)
    print(f"Extended dataset written to {out_csv}, shape = {extended.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build an extended ammonia DIAL dataset with multiple noise levels."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to the original high_freq_ammonia_simulated.csv"
    )
    parser.add_argument(
        "--out", default="extended_ammonia_dataset.csv",
        help="Where to write the combined dataset"
    )
    args = parser.parse_args()
    build_extended_dataset(args.csv, args.out)
