
# DIAL Signal Processing & AI Toolkit

A complete pipeline for **Differential Absorption LIDAR (DIAL)** signal simulation, processing, and machine‑learning‑based α‑recovery, developed in collaboration with VTEC Lasers & Sensors.

---

##  Project Overview

1. **Signal Simulation**  
   - `simulate_dial.ipynb` / `simulate_dial.py`  
   - Generates synthetic DIAL absorption‑coefficient profiles (α) with realistic atmospheric effects.

2. **Classical Denoising**  
   - Filters: Moving Average, Savitzky–Golay, Wavelets, Splines.  
   - Implemented in the Streamlit dashboard for quick comparison.

3. **Extended Dataset Generation**  
   - `AI_Module/LSTM/extended_dataset.py`  
   - Creates noisy ammonia signals at low/medium/high Gaussian noise levels.

4. **Second‑Gen LSTM Training**  
   - `AI_Module/LSTM/train_lstm_v2.py`  
   - Trains a robust LSTM (noise augmentation, L2 regularization, early stopping) on the extended dataset.  
   - Produces `lstm_v2.pt`.

5. **Professional Streamlit Dashboard**  
   - `AI_Module/LSTM/models/professional_dashboard.py`  
   - Combines classical filters, LSTM α‑recovery, Bayesian (or linear) calibration, and a 2D GP uncertainty map.  
   - Exportable PDF reports.
