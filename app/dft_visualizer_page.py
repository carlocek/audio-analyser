import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.io import wavfile
import sys
sys.path.append("C:/Users/carlo/Desktop/Github_repos/audio-analyser")

from engine.signal import Signal
from engine.signal_generator import SignalGenerator
from engine.dft import *


def update_signal(i):
    frequency = st.session_state[f"frequency_{i}"]
    amplitude = st.session_state[f"amplitude_{i}"]
    phase = st.session_state[f"phase_{i}"]
    t, y = generator.generate_signal(frequency, amplitude, phase, default_duration)
    st.session_state.signals[i] = Signal(frequency, amplitude, phase, t, y)

st.title("Interactive DFT Visualizer")

max_signals = 10
sample_rate = 44100
default_frequency = 20.0
default_amplitude = 1.0
default_phase = 0.0
default_duration = 3.0

generator = SignalGenerator(sample_rate)

if 'signals' not in st.session_state:
    st.session_state.signals = []

st.subheader("Signal Visualization")
st.write("Let's start by generating some signals and visualize their summed signal")
if len(st.session_state.signals) < max_signals and st.button("Add Signal"):
    t, y = generator.generate_signal(default_frequency, default_amplitude, default_phase, default_duration)
    st.session_state.signals.append(Signal(default_frequency, default_amplitude, default_phase, t, y))
    i = len(st.session_state.signals) - 1
    st.session_state[f"frequency_{i}"] = default_frequency
    st.session_state[f"amplitude_{i}"] = default_amplitude
    st.session_state[f"phase_{i}"] = default_phase

# plot each signal with its sliders
for i, signal in enumerate(st.session_state.signals):
    st.write(f"Signal {i+1}")
    col1, col2, col3 = st.columns([1, 3, 1]) # slider and main columns
    
    with col1:
        frequency = st.slider(
            f"Frequency {i+1} (Hz)",
            min_value=20.0, max_value=2000.0, step=10.0,
            key=f"frequency_{i}",
            on_change=update_signal,
            args=(i,)
        )
        amplitude = st.slider(
            f"Amplitude {i+1}",
            min_value=0.0, max_value=1.0, step=0.1,
            key=f"amplitude_{i}",
            on_change=update_signal,
            args=(i,)
        )
        phase = st.slider(
            f"Phase {i+1} (radians)",
            min_value=0.0, max_value=2 * np.pi, step=0.1,
            key=f"phase_{i}",
            on_change=update_signal,
            args=(i,)
        )

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.signals[i].t, y=st.session_state.signals[i].y, mode="lines", name=f"Signal {i+1}"))
        fig.update_layout(
            yaxis=dict(range=[-1.0, 1.0]),
            xaxis=dict(range=[0.0, 0.05]),
            margin=dict(l=0, r=0, t=0, b=0),
            height=300
        )
        st.plotly_chart(fig)

    with col3:
        if st.button("Delete Signal", key=f"delete_{i}"):
            st.session_state.signals.pop(i)
            st.rerun()

# plot summed signal
st.write("Summed Signal")
t = np.linspace(0, default_duration, int(sample_rate * default_duration), endpoint=False)
if not st.session_state.signals:
    summed_signal = np.zeros_like(t)
else:
    summed_signal = sum(signal.y for signal in st.session_state.signals)
fig_sum = go.Figure()
fig_sum.add_trace(go.Scatter(x=t, y=summed_signal, mode="lines", name="Summed Signal", line=dict(color="red", width=3)))
fig_sum.update_layout(
    xaxis=dict(range=[0.0, 0.05]),
    margin=dict(l=0, r=0, t=0, b=0),
    height=200
)
st.plotly_chart(fig_sum)

# save summed signal to WAV file for playback
peak = np.max(np.abs(summed_signal))
normalized_signal = np.int16((summed_signal / peak) * min(peak, 1) * 32767)
# normalized_signal = np.int16(playback_signal * 32767)
wav_path = "../data/summed_signal.wav"
wavfile.write(wav_path, sample_rate, normalized_signal)
st.audio(wav_path, format="audio/wav")


# section on DFT visualization
st.subheader("Frequency Extraction Visualization")
st.markdown("""
            Let's see how the DFT is computed on the summed signal above. This signal, as defined by you, is composed of one or more sine waves having frequencies $f_i$ with $i \in \{0,\ldots, n\}$.
            The idea is to "wrap" the signal around a circle using a certain "wrapping frequency" $f_t$ and track the position of the centroid (or baricenter) of the signal samples.
            Ww then increase $f_t$ and whenever $f_t=f_i$ we observe a position-shift of the centroid -> that's the basic intuition on how we detect frequencies in time-domain signals using DFT
            """
)

if st.toggle("Show fixed wrapping frequency animation"):
    test_freq = st.slider("Wrapping Frequency (Hz)", 0.0, 2000.0, 440.0, 10.0)
    st.write("Signal projection on complex circumference for fixed test frequency")
    fig = wrapping_signal_fixedfreq_animation(t, summed_signal, test_freq)
    st.plotly_chart(fig, use_container_width=True)

if st.toggle("Show frequency extraction animation"):
    st.write("Signal projection on complex circumference for increasing test frequencies and relative centroids x-coordinates")
    fig = wrapping_signal_varyingfreq_animation(t, summed_signal)
    st.plotly_chart(fig, use_container_width=True)