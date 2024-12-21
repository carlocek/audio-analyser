import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
sys.path.append("C:/Users/carlo/Desktop/Github_repos/audio-analyser")

from engine.audio_loader import AudioLoader
from app.signal_list import SignalList

def main():
    # loader = AudioLoader("data/1980s-Casio-Piano-C5.wav")
    # loader.load()
    # print(f"Audio Info: {loader.get_info()}")

    st.title("Interactive Signal Visualizer")

    signal_list = SignalList()

    if 'signals' not in st.session_state:
        st.session_state.signals = []

    for signal in st.session_state.signals:
        signal_list.add_signal(signal["frequency"], signal["amplitude"], signal["phase"])

    max_signals = 4

    # button to add new signals
    st.header("Manage Signals")
    if len(st.session_state.signals) < max_signals:
        if st.button("Add Signal"):
            signal_list.add_signal(frequency=440, amplitude=1.0, phase=0.0)
            st.session_state.signals = signal_list.signals

    # plot each signal with its sliders
    for i, signal in enumerate(st.session_state.signals):
        st.subheader(f"Signal {i+1}")
        col1, col2 = st.columns([1, 3]) # slider and main columns
        
        with col1:
            frequency = st.slider(f"Frequency of Signal {i+1}", 20, 800, int(signal["frequency"]), step=10)
            amplitude = st.slider(f"Amplitude of Signal {i+1}", 0.0, 1.0, signal["amplitude"], step=0.1)
            phase = st.slider(f"Phase of Signal {i+1}", 0.0, 2 * np.pi, signal["phase"], step=0.1)
        
            # update signal if sliders change
            if signal["frequency"] != frequency or signal["amplitude"] != amplitude or signal["phase"] != phase:
                signal_list.update_signal(i, frequency, amplitude, phase)
                st.session_state.signals = signal_list.signals

        with col2:
            t, y = signal_list.generator.generate_signal(frequency, amplitude, phase)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=f"Signal {i+1}"))
            fig.update_layout(
                yaxis=dict(range=[-1.0, 1.0]),
                margin=dict(l=0, r=0, t=0, b=0),
                height=200  # Altezza del grafico per allinearlo con gli slider
            )
            st.plotly_chart(fig)

    st.subheader("Summed Signal")
    t, summed_signal = signal_list.sum_signals()
    fig_sum = go.Figure()
    fig_sum.add_trace(go.Scatter(x=t, y=summed_signal, mode="lines", name="Summed Signal", line=dict(color="red", width=3)))
    st.plotly_chart(fig_sum)

if __name__ == "__main__":
    main()
