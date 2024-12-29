import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.io import wavfile
from multiprocessing import Value

from engine.dft import *
from engine.signal_generator import SignalGenerator



st.title("Frequency Extractor using DFT")
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file:
    sample_rate, data = wavfile.read(uploaded_file)
    if data.ndim > 1:  # Convert to mono if stereo
        data = np.mean(data, axis=1).astype(data.dtype)
    data = data - np.mean(data)  # Remove DC offset

    generator = SignalGenerator(sample_rate)

    N = len(data)

    # plot time domain signal
    t = np.linspace(0, N/sample_rate, N, endpoint=False)
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=t, y=data, mode="lines", name="Audio Signal"))
    fig_time.update_layout(
        title="Time-Domain Signal", 
        xaxis_title="Time (s)", 
        yaxis_title="Amplitude"
    )
    st.plotly_chart(fig_time)
    st.audio(uploaded_file, format="audio/wav")

    use_max_frequencies = st.checkbox("Use maximum number of frequencies", value=False, help=f"maximum number of frequencies: {N // 2 + 1}")
    # choose the number of top frequencies to visualize
    k = st.number_input(
        "Number of top frequencies to visualize and playback",
        min_value=1, max_value=N // 2 + 1, value=None, step=1, format="%d", disabled=use_max_frequencies
    )

    if use_max_frequencies:
        k = N // 2 + 1

    if st.button("Run DFT"):
        # st.write("Computing DFT...")

        progress_bar = st.progress(0, text="Computing DFT")  # Initialize Streamlit progress bar
        spectrum = dft_parallel(data, sample_rate, progress_bar, block_size=100)
        st.write("DFT Completed!")
        spectrum = sorted(spectrum, key=lambda item: item["amplitude"], reverse=True)

        # Extract frequencies, amplitudes, and phases
        freqs = np.array([item["frequency"] for item in spectrum])
        amps = np.array([item["amplitude"] for item in spectrum])
        phases = np.array([item["phase"] for item in spectrum])

        positive_freqs = np.where(freqs > 0)
        freqs = freqs[positive_freqs]
        amps = amps[positive_freqs]
        phases = phases[positive_freqs]

        top_indices = np.argsort(amps)[::-1][:k]
        top_freqs = freqs[top_indices]
        top_amps = amps[top_indices]
        top_phases = phases[top_indices]

        # top_freqs = freqs[:k]
        # top_amps = amps[:k]
        # top_phases = phases[:k]

        # plot frequency spectrum
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(x=freqs, y=amps, mode="lines", name="Frequency Spectrum"))
        fig_freq.add_trace(go.Scatter(x=top_freqs, y=top_amps, mode="markers", name="Top Frequencies", marker=dict(size=5, color="red")))
        fig_freq.update_layout(title="Frequency Spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
        st.plotly_chart(fig_freq)

        # reconstruct the signal
        reconstructed_signal = np.zeros_like(t)
        for f, a, p in zip(top_freqs, top_amps, top_phases):
            _, y = generator.generate_signal(f, a, p, N/sample_rate)
            reconstructed_signal += y

        print("signal reconstructed")
        # plot reconstructed signal
        fig_recon = go.Figure()
        fig_recon.add_trace(go.Scatter(x=t, y=reconstructed_signal, mode="lines", name="Reconstructed Signal"))
        fig_recon.update_layout(title="Reconstructed Signal", xaxis_title="Time (s)", yaxis_title="Amplitude")
        st.plotly_chart(fig_recon)

        # save reconstructed signal as WAV for playback
        reconstructed_signal_int = np.int16((reconstructed_signal / np.max(np.abs(reconstructed_signal))) * 32767)
        wav_path = "../data/topk_frequencies_signal.wav"
        wavfile.write(wav_path, sample_rate, reconstructed_signal_int)
        st.audio(wav_path, format="audio/wav")
