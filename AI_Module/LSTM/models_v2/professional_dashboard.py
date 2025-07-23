import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from fpdf import FPDF
import io
import torch
from torch import nn

st.set_page_config(
    page_title="DIAL Signal & AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- LSTM Model Definition ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

@st.cache_resource
def load_lstm_model(path: str):
    model = LSTMRegressor()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# --- Helper Functions ---
def apply_filter(df, kind):
    if kind == "Moving Average":
        return df["Estimated_alpha"].rolling(7, center=True).mean()
    if kind == "Savitzky–Golay":
        return savgol_filter(df["Estimated_alpha"], 11, 2)
    if kind == "Wavelet":
        coeffs = pywt.wavedec(df["Estimated_alpha"], "db4", level=3)
        coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
        rec = pywt.waverec(coeffs, "db4")
        return rec[: len(df)]
    if kind == "Spline":
        x, y = df["Distance_m"], df["Estimated_alpha"]
        mask = ~np.isnan(y)
        spline = UnivariateSpline(x[mask], y[mask], s=1e-3)
        return spline(x)
    return df["Estimated_alpha"]

def make_pdf(report_title, fig_zoom, mse, mae):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, report_title, ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Segment MSE: {mse:.4e}", ln=True)
    pdf.cell(0, 8, f"Segment MAE: {mae:.4e}", ln=True)
    # embed zoomed figure
    buf = io.BytesIO()
    fig_zoom.savefig(buf, format="PNG")
    buf.seek(0)
    img_path = "tmp_zoom.png"
    with open(img_path, "wb") as f:
        f.write(buf.read())
    pdf.image(img_path, x=15, y=60, w=180)
    return pdf.output(dest="S").encode("latin1")

# --- Sidebar Inputs ---
st.sidebar.title("⚙️ Controls")
uploaded_exp = st.sidebar.file_uploader("1) Upload DIAL CSV", type="csv")
uploaded_sim = st.sidebar.file_uploader("2) Upload Simulated CSV", type="csv")
filter_choice = st.sidebar.selectbox(
    "3) Classical Filter", ["None", "Moving Average", "Savitzky–Golay", "Wavelet", "Spline"]
)
run_lstm = st.sidebar.checkbox("4) Run LSTM Prediction", value=True)

seg_start = st.sidebar.number_input("Segment Start [m]", value=300.0)
seg_end   = st.sidebar.number_input("Segment End [m]", value=600.0)

# --- Main Workflow ---
st.title("🚀 DIAL Signal Processing & AI Dashboard")

# 1) Experimental DIAL Data Panel
if uploaded_exp:
    df_exp = pd.read_csv(uploaded_exp)
    df_exp["Filtered"] = (
        df_exp["Estimated_alpha"] if filter_choice=="None"
        else apply_filter(df_exp, filter_choice)
    )
    mask = (df_exp.Distance_m>=seg_start)&(df_exp.Distance_m<=seg_end)
    true_seg = df_exp.True_alpha[mask]
    est_seg  = df_exp.Filtered[mask]
    mse_seg  = np.mean((true_seg-est_seg)**2)
    mae_seg  = np.mean(np.abs(true_seg-est_seg))

    st.subheader("1️⃣ Classical Denoising")
    fig1, ax1 = plt.subplots(figsize=(8,3))
    ax1.plot(df_exp.Distance_m, df_exp.True_alpha, label="True α", c="black", lw=2)
    ax1.plot(df_exp.Distance_m, df_exp.Estimated_alpha, "--", label="Raw", alpha=0.3)
    ax1.plot(df_exp.Distance_m, df_exp.Filtered, label=filter_choice)
    ax1.axvspan(seg_start, seg_end, color="grey", alpha=0.2)
    ax1.set_ylabel("α [1/m]"); ax1.set_xlabel("Distance [m]")
    ax1.legend(); ax1.grid(True)
    st.pyplot(fig1)

    st.markdown(f"""
    **Segment MSE**: {mse_seg:.4e}  
    **Segment MAE**: {mae_seg:.4e}  
    """)
else:
    st.info("Please upload your experimental DIAL CSV to begin.")

# 2) Simulated + LSTM Panel
if uploaded_sim and run_lstm:
    df_sim = pd.read_csv(uploaded_sim)
    model = load_lstm_model("lstm_v2.pt")
    # prepare sequences
    seq_len = 20
    signal = df_sim["Noisy_Signal"].values.reshape(-1,1)
    windows = []
    for i in range(len(signal)-seq_len):
        windows.append(signal[i:i+seq_len])
    X = torch.tensor(np.stack(windows), dtype=torch.float32)
    with torch.no_grad():
        preds = model(X).numpy().flatten()
    full_pred = np.full(len(df_sim), np.nan)
    full_pred[seq_len:] = preds
    df_sim["LSTM_pred"] = full_pred

    st.subheader("2️⃣ LSTM‑Based α Recovery")
    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.plot(df_sim.True_alpha, label="True α", c="orange", alpha=0.6)
    ax2.plot(df_sim.LSTM_pred, label="Predicted α", c="blue", alpha=0.6)
    ax2.set_title("All Samples"); ax2.set_ylabel("α [1/m]"); ax2.set_xlabel("Sample Index")
    ax2.legend(); ax2.grid(True)
    st.pyplot(fig2)

    # per‑noise metrics if column present
    if "Noise_Level" in df_sim.columns:
        metrics = []
        for nl in df_sim.Noise_Level.unique():
            sub = df_sim[df_sim.Noise_Level==nl].dropna(subset=["LSTM_pred"])
            mse = np.mean((sub.True_alpha - sub.LSTM_pred)**2)
            mae = np.mean(np.abs(sub.True_alpha - sub.LSTM_pred))
            metrics.append({"Noise": nl, "MSE": mse, "MAE": mae})
        st.table(pd.DataFrame(metrics).set_index("Noise"))

else:
    st.info("Simulated CSV not uploaded or LSTM disabled via sidebar.")

# 3) PDF Report
if uploaded_exp and st.button("📄 Download Full Report"):
    pdf_bytes = make_pdf(
        report_title="DIAL Signal Report",
        fig_zoom=fig1,  # reuse classical zoom plot
        mse=mse_seg, mae=mae_seg
    )
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name="DIAL_report.pdf",
        mime="application/pdf"
    )
