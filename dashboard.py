import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pywt
from scipy.interpolate import UnivariateSpline
from fpdf import FPDF
import io

st.set_page_config(layout="wide")

# ----------------------
# Title and Description
# ----------------------
st.title("🔬 DIAL Signal Processing Dashboard")
st.markdown("""
Welcome to the Differential Absorption LIDAR (DIAL) dashboard.  
Upload your simulated or experimental data, choose a denoising filter, visualize results, and compute accuracy.  
Use the manual segmentation panel below to zoom into a specific distance range for analysis.
""")

# ----------------------
# File Upload
# ----------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.sidebar.header("⚙️ Filter Options")
    filter_type = st.sidebar.selectbox("Choose filter", ["None", "Moving Average", "Savitzky-Golay", "Wavelet", "Spline"])

    # ----------------------
    # Apply Selected Filter
    # ----------------------
    df["Filtered_alpha"] = df["Estimated_alpha"]

    if filter_type == "Moving Average":
        df["Filtered_alpha"] = df["Estimated_alpha"].rolling(window=7, center=True).mean()

    elif filter_type == "Savitzky-Golay":
        df["Filtered_alpha"] = savgol_filter(df["Estimated_alpha"], window_length=11, polyorder=2)

    elif filter_type == "Wavelet":
        coeffs = pywt.wavedec(df["Estimated_alpha"], 'db4', level=3)
        coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
        df["Filtered_alpha"] = pywt.waverec(coeffs, 'db4')[:len(df)]

    elif filter_type == "Spline":
        x = df["Distance_m"]
        y = df["Estimated_alpha"]
        mask = ~np.isnan(y)
        spline = UnivariateSpline(x[mask], y[mask], s=0.001)
        df["Filtered_alpha"] = spline(x)

    # ----------------------
    # Manual Zoom Panel
    # ----------------------
    st.sidebar.header("🔎 Manual Zoom (Segment Selection)")
    seg_start = st.sidebar.slider("Segment start [m]", int(df["Distance_m"].min()), int(df["Distance_m"].max()), 300)
    seg_end = st.sidebar.slider("Segment end [m]", seg_start + 1, int(df["Distance_m"].max()), 600)

    zoom_mask = (df["Distance_m"] >= seg_start) & (df["Distance_m"] <= seg_end)

    # ----------------------
    # Compute Relative Error (Zoomed region)
    # ----------------------
    true_alpha = df["True_alpha"][zoom_mask]
    est_alpha = df["Filtered_alpha"][zoom_mask]

    valid = np.abs(true_alpha) > 1e-4
    rel_error = np.full_like(true_alpha, np.nan)
    rel_error[valid] = np.abs((est_alpha[valid] - true_alpha[valid]) / true_alpha[valid]) * 100

    st.subheader("📈 Estimation Accuracy (Selected Segment)")
    st.write(f"**Mean Relative Error:** {np.nanmean(rel_error):.2f}%")
    st.write(f"**Standard Deviation:** {np.nanstd(rel_error):.2f}%")

 # ----------------------
    # Spectral Analysis (FFT)
# ----------------------
    st.header("🔬 Spectral Analysis Panel (FFT)")

    # --- Compute FFT of the selected signal ---
    signal_to_analyze = df["Filtered_alpha"].fillna(0)  # Always use current filtered signal
    n = len(signal_to_analyze)
    dr = df["Distance_m"].iloc[1] - df["Distance_m"].iloc[0]  # spatial resolution
    freqs = np.fft.fftfreq(n, d=dr)
    fft_values = np.abs(np.fft.fft(signal_to_analyze))

    # Keep only positive frequencies
    positive_mask = freqs >= 0
    positive_freqs = freqs[positive_mask]
    fft_positive = fft_values[positive_mask]

    # --- Plot FFT ---
    fig_fft, ax_fft = plt.subplots(figsize=(10, 4))
    ax_fft.plot(positive_freqs, fft_positive, label=f"FFT of {filter_type} signal")
    ax_fft.set_title("Frequency Spectrum (FFT)")
    ax_fft.set_xlabel("Frequency [1/m]")
    ax_fft.set_ylabel("Amplitude")
    ax_fft.grid(True)
    ax_fft.legend()
    st.pyplot(fig_fft)




    # ----------------------
    # Plot Zoomed View
    # ----------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Distance_m"], df["True_alpha"], label="True α", linewidth=2)
    ax.plot(df["Distance_m"], df["Estimated_alpha"], '--', label="Raw Estimated α", alpha=0.5)
    ax.plot(df["Distance_m"], df["Filtered_alpha"], label=f"{filter_type} α", linewidth=2)
    ax.axvspan(seg_start, seg_end, color='gray', alpha=0.2, label="Selected Segment")
    ax.set_xlim(seg_start - 10, seg_end + 10)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Absorption Coefficient α [1/m]")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # ----------------------
    # Generate PDF Report
    # ----------------------
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "DIAL Signal Report", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Filter Used: {filter_type}", ln=True)
        pdf.cell(0, 10, f"Segment Range: {seg_start}m to {seg_end}m", ln=True)
        pdf.cell(0, 10, f"Mean Relative Error: {np.nanmean(rel_error):.2f}%", ln=True)
        pdf.cell(0, 10, f"Standard Deviation: {np.nanstd(rel_error):.2f}%", ln=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_path = "plot_temp.png"
        with open(img_path, "wb") as f:
            f.write(buf.read())
        pdf.image(img_path, x=10, y=60, w=190)
        return pdf.output(dest="S").encode("latin1")

    if st.button("📄 Download PDF Report"):
        pdf_bytes = generate_pdf()
        st.download_button(
            label="Download Report",
            data=pdf_bytes,
            file_name="dial_report.pdf",
            mime="application/pdf"
        )
