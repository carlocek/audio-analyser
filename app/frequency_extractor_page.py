import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile

from engine.audio_loader import AudioLoader
from engine.signal import Signal
from engine.signal_generator import SignalGenerator


st.title("Frequency Extractor")
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    with open("../data/temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    loader = AudioLoader("temp.wav")
    loader.load()
    # audio_info = loader.get_info()
    # st.write("Audio Info:", audio_info)
    data = np.array(loader.data)
    sample_rate = loader.sample_rate
    N = len(data)

    # time domain signal
    t = np.linspace(0, N/sample_rate, N, endpoint=False)
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=t, y=data, mode="lines", name="Audio Signal"))
    fig_time.update_layout(
        title="Time-Domain Signal", 
        xaxis_title="Time (s)", 
        yaxis_title="Amplitude"
    )
    st.plotly_chart(fig_time)

    k = st.number_input("Number of top frequencies to extract", min_value=1, max_value=10, value=3, step=1)

    # FFT
    yf = fft(data)
    xf = fftfreq(N, 1/sample_rate)

    # take only positive frequencies
    idx = np.where(xf > 0)
    xf = xf[idx]
    yf = (2.0/N)*np.abs(yf[idx])

    # sort frequencies by descending amplitude and take top k
    top_indices = np.argsort(yf)[::-1][:k]
    top_frequencies = xf[top_indices]
    top_amplitudes = yf[top_indices]

    # frequency spectrum
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=yf, mode="lines", name="FFT"))
    fig.add_trace(go.Scatter(x=top_frequencies, y=top_amplitudes, mode="markers", name="Top Frequencies", marker=dict(size=10, color="red")))
    fig.update_layout(
        title="Frequency Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig)
    
    # for each frequency plot the associated signal
    st.subheader(f"Top {k} Frequencies and Their Sinusoids")
    plot_duration = 0.05 
    plot_t = np.linspace(0, plot_duration, int(sample_rate*plot_duration), endpoint=False)
    cols = st.columns(3)
    full_signals = []
    plot_signals = []

    for i, (freq, amp) in enumerate(zip(top_frequencies, top_amplitudes)):
        plot_signal = amp * np.sin(2 * np.pi * freq * plot_t)
        plot_signals.append(plot_signal)
        full_signal = amp * np.sin(2 * np.pi * freq * t)
        full_signals.append(full_signal)

        with cols[i % 3]:
            st.markdown(f"**Frequency:** {freq:.2f} Hz, **Amplitude:** {amp:.2f}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_t, y=plot_signal, mode="lines", name=f"Signal {i+1}"))
            fig.update_layout(
                height=200,
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # let user add generated signal with sliders to toggle freq, ampl and phase
    # max_signals = 4
    # sample_rate = 44100
    # default_frequency = 1.0
    # default_amplitude = 1.0
    # default_phase = 0.0
    # generator = SignalGenerator(sample_rate)

    # if "user_signals" not in st.session_state:
    #     st.session_state.user_signals = []

    # if len(st.session_state.user_signals) < max_signals and st.button("Add Signal"):
    #     t, y = generator.generate_signal(default_frequency, default_amplitude, default_phase, plot_duration)
    #     st.session_state.user_signals.append(Signal(default_frequency, default_amplitude, default_phase, t, y))
    #     i = len(st.session_state.user_signals) - 1
    #     st.session_state[f"frequency_{i}"] = default_frequency
    #     st.session_state[f"amplitude_{i}"] = default_amplitude
    #     st.session_state[f"phase_{i}"] = default_phase
    
    # def update_signal(i):
    #     frequency = st.session_state[f"frequency_{i}"]
    #     amplitude = st.session_state[f"amplitude_{i}"]
    #     phase = st.session_state[f"phase_{i}"]
    #     t, y = generator.generate_signal(frequency, amplitude, phase, plot_duration)
    #     st.session_state.user_signals[i] = Signal(frequency, amplitude, phase, t, y)
    
    # user_defined_signals = []
    # for i, signal in enumerate(st.session_state.user_signals):
    #     st.subheader(f"Signal {i+1}")
    #     col1, col2, col3 = st.columns([1, 3, 1]) # slider and main columns
        
    #     with col1:
    #         frequency = st.slider(
    #             f"Frequency {i+1} (Hz)",
    #             min_value=1.0, max_value=500.0, step=1.0,
    #             key=f"frequency_{i}",
    #             on_change=update_signal,
    #             args=(i,)
    #         )
    #         amplitude = st.slider(
    #             f"Amplitude {i+1}",
    #             min_value=0.0, max_value=top_amplitudes[0], step=1.0,
    #             key=f"amplitude_{i}",
    #             on_change=update_signal,
    #             args=(i,)
    #         )
    #         phase = st.slider(
    #             f"Phase {i+1} (radians)",
    #             min_value=0.0, max_value=2 * np.pi, step=0.1,
    #             key=f"phase_{i}",
    #             on_change=update_signal,
    #             args=(i,)
    #         )

    #         user_signal = signal.amplitude * np.sin(2 * np.pi * signal.frequency * signal.t + signal.phase)
    #         user_defined_signals.append(user_signal)

    #     with col2:
    #         fig = go.Figure()
    #         fig.add_trace(go.Scatter(x=st.session_state.user_signals[i].t, y=st.session_state.user_signals[i].y, mode="lines", name=f"Signal {i+1}"))
    #         fig.update_layout(
    #             yaxis=dict(range=[-1.0, 1.0]),
    #             margin=dict(l=0, r=0, t=0, b=0),
    #             height=200  # fig height to line up with sliders
    #         )
    #         st.plotly_chart(fig)

    #     with col3:
    #         if st.button("Delete Signal", key=f"delete_{i}"):
    #             st.session_state.user_signals.pop(i)
    #             st.rerun()

    # plot summed signal
    st.subheader(f"Sum of signals associated with top {k} frequencies and user defined signals")
    plot_summed_signal = np.sum(plot_signals, axis=0)
    fig_sum = go.Figure()
    fig_sum.add_trace(go.Scatter(x=plot_t, y=plot_summed_signal, mode="lines", name="Summed Signal", line=dict(color="red", width=3)))
    fig_sum.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_sum)

    # save summed signal to WAV file for playback
    full_summed_signal = np.sum(full_signals, axis=0)
    summed_signal_int = np.int16(full_summed_signal / np.max(np.abs(full_summed_signal)) * 32767)  # Normalize to 16-bit PCM
    wav_path = "../data/topk_frequencies_signal.wav"
    wavfile.write("../data/topk_frequencies_signal.wav", sample_rate, summed_signal_int)

    st.audio(wav_path, format="audio/wav")
       
