import streamlit as st
import numpy as np
import plotly.graph_objects as go
from time import sleep
import sys
sys.path.append("C:/Users/carlo/Desktop/Github_repos/audio-analyser")

from engine.audio_loader import AudioLoader
from engine.signal import Signal
from engine.signal_generator import SignalGenerator

def main():
    # loader = AudioLoader("data/1980s-Casio-Piano-C5.wav")
    # loader.load()
    # print(f"Audio Info: {loader.get_info()}")

    def update_signal(i, frequency, amplitude, phase):
        t, y = generator.generate_signal(frequency, amplitude, phase, default_duration)
        st.session_state.signals[i] = Signal(frequency, amplitude, phase, t, y)

    st.title("Interactive Signal Visualizer")

    max_signals = 4
    sample_rate = 44100
    default_frequency = 1.0
    default_amplitude = 1.0
    default_phase = 0.0
    default_duration = 1.0

    generator = SignalGenerator(sample_rate)

    if 'signals' not in st.session_state:
        st.session_state.signals = []

    # button to add new signals
    st.header("Manage Signals")
    if len(st.session_state.signals) < max_signals and st.button("Add Signal"):
        t, y = generator.generate_signal(default_frequency, default_amplitude, default_phase, default_duration)
        st.session_state.signals.append(Signal(default_frequency, default_amplitude, default_phase, t, y))
        i = len(st.session_state.signals) - 1
        st.session_state[f"frequency_{i}"] = default_frequency
        st.session_state[f"amplitude_{i}"] = default_amplitude
        st.session_state[f"phase_{i}"] = default_phase

    # plot each signal with its sliders
    for i, signal in enumerate(st.session_state.signals):
        st.subheader(f"Signal {i+1}")
        col1, col2 = st.columns([1, 3]) # slider and main columns
        
        with col1:
            freq_key = f"frequency_{i}"
            amp_key = f"amplitude_{i}"
            phase_key = f"phase_{i}"

            frequency = st.slider(
                f"Frequency {i+1} (Hz)",
                min_value=1.0,
                max_value=50.0,
                step=1.0,
                key=freq_key
            )
            amplitude = st.slider(
                f"Amplitude {i+1}",
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                key=amp_key
            )
            phase = st.slider(
                f"Phase {i+1} (radians)",
                min_value=0.0,
                max_value=2 * np.pi,
                step=0.1,
                key=phase_key
            )

            update_signal(i, frequency, amplitude, phase)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=st.session_state.signals[i].t, y=st.session_state.signals[i].y, mode="lines", name=f"Signal {i+1}"))
            fig.update_layout(
                yaxis=dict(range=[-1.0, 1.0]),
                margin=dict(l=0, r=0, t=0, b=0),
                height=200  # fig height to line up with sliders
            )
            st.plotly_chart(fig)

    st.subheader("Summed Signal")
    t = np.linspace(0, default_duration, int(sample_rate * default_duration), endpoint=False)
    if not st.session_state.signals:
        summed_signal = np.zeros_like(t)
    else:
        summed_signal = sum(signal.y for signal in st.session_state.signals)
    fig_sum = go.Figure()
    fig_sum.add_trace(go.Scatter(x=t, y=summed_signal, mode="lines", name="Summed Signal", line=dict(color="red", width=3)))
    st.plotly_chart(fig_sum)

if __name__ == "__main__":
    main()
