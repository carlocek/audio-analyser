import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt
from scipy.io import wavfile

from engine.audio_loader import AudioLoader
from engine.signal import Signal
from engine.signal_generator import SignalGenerator


st.title("Frequency Extractor")
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    # with open("../data/input.wav", "wb") as f:
    #     f.write(uploaded_file.read())
    # loader = AudioLoader("../data/input.wav")
    # loader.load()
    # st.write(loader.get_info())
    # data = np.array(loader.data)
    # sample_rate = loader.sample_rate

    sample_rate, data = wavfile.read(uploaded_file)
    # if stereo signal, convert to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1).astype(data.dtype)  # Converti a mono
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

    data = data - np.mean(data)

    duration = N / sample_rate
    padding_duration = 10 - (duration % 10) # Durata desiderata in secondi
    padding_length = int(sample_rate * padding_duration)
    data = np.pad(data, (0, padding_length), mode='constant')
    N_padded = len(data)

    generator = SignalGenerator(sample_rate)

    # def highpass_filter(data, cutoff, fs, order):
    #     nyquist = 0.5 * fs
    #     normal_cutoff = cutoff / nyquist
    #     b, a = butter(order, normal_cutoff, btype='high', analog=False)
    #     filtered_data = filtfilt(b, a, data)
    #     return filtered_data
    # data = highpass_filter(data, 250, sample_rate, 4)

    # FFT
    padded_length = int(2**np.ceil(np.log2(N)))
    print(N)
    # print(N_padded)
    print(padded_length)
    print("duration 1: ", N/sample_rate)
    # print("duration 2: ", N_padded/sample_rate)
    print("duration 3: ", padded_length/sample_rate)
    yf = fft(data, n=N_padded)
    xf = fftfreq(N_padded, 1/sample_rate)
    phases = -np.atan2(yf)

    # take only positive frequencies
    idx = np.where(xf > 0)
    xf = xf[idx]
    yf = yf[idx]
    phases = phases[idx]
    yf = (2.0/N)*np.abs(yf)[1:]

    # choose the number of top frequencies to visualize
    # k = st.number_input(
    #     "Number of Top Frequencies to visualize and playback",
    #     min_value=1, max_value=len(xf), value=1, step=1, format="%d"
    # )
    k = len(xf)

    print(yf)
    peak_indices = np.where((yf[1:-1] > yf[:-2]) & (yf[1:-1] > yf[2:]))[0] + 1
    sorted_peaks = peak_indices[np.argsort(yf[peak_indices])[::-1]]
    top_indices = sorted_peaks[:k]
    print(len(top_indices))

    # sort frequencies by descending amplitude and take top k
    # valid_indices = np.where(yf > 0.001 * np.max(yf))[0]  # Indici sul dominio originale
    # sorted_indices = np.argsort(yf[valid_indices])[::-1]  # Indici relativi a valid_indices
    # top_indices = valid_indices[sorted_indices[:k]]  # Converte in indici sul dominio originale

    # Seleziona le frequenze, ampiezze e fasi top
    top_frequencies = xf[top_indices]
    top_amplitudes = yf[top_indices]
    top_phases = phases[top_indices]

    is0Hz = top_indices == 0
    isNyquistFreq = (top_indices == len(xf) - 1) & (len(data) % 2 == 0)
    amplitude_scale = np.where(is0Hz | isNyquistFreq, 1.0, 2.0)
    top_amplitudes = top_amplitudes * amplitude_scale

    # frequency spectrum
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=yf, mode="lines", name="FFT"))
    fig.add_trace(go.Scatter(x=top_frequencies, y=top_amplitudes, mode="markers", name="Top Frequencies", marker=dict(size=5, color="red")))
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
    cols = st.columns(3)
    full_signals = []
    plot_signals = []

    for i, (freq, amp, phase) in enumerate(zip(top_frequencies, top_amplitudes, top_phases)):
        # show max 15 signals
        if i > 14:
            break
        plot_t, plot_y = generator.generate_signal(freq, amp, phase, plot_duration)
        plot_signals.append(plot_y)
        full_t, full_y = generator.generate_signal(freq, amp, phase, N/sample_rate)
        full_signals.append(full_y)

        with cols[i % 3]:
            st.markdown(f"**Frequency:** {freq:.2f} Hz, **Amplitude:** {amp:.2f}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_t, y=plot_y, mode="lines", name=f"Signal {i+1}"))
            fig.update_layout(
                height=200,
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        
    # plot summed signal
    st.subheader(f"Sum of signals associated with top {k} frequencies and user defined signals")
    plot_summed_y = np.sum(plot_signals, axis=0)
    fig_sum = go.Figure()
    fig_sum.add_trace(go.Scatter(x=plot_t, y=plot_summed_y, mode="lines", name="Summed Signal", line=dict(color="red", width=3)))
    fig_sum.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_sum)

    full_summed_y = np.sum(full_signals, axis=0)
    temp = go.Figure()
    temp.add_trace(go.Scatter(x=full_t, y=full_summed_y, mode="lines", name="Temp Summed Signal", line=dict(color="red", width=3)))
    temp.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(temp)

    # save summed signal to WAV file for playback
    summed_signal_int = np.int16((full_summed_y / np.max(np.abs(full_summed_y))) * 12767)  # Normalize to 16-bit PCM
    wav_path = "../data/topk_frequencies_signal.wav"
    wavfile.write(wav_path, sample_rate, summed_signal_int)
    st.audio(wav_path, format="audio/wav")

    print("Original signal amplitude:", np.max(data) - np.min(data))
    print("Summed signal amplitude:", np.max(full_summed_y) - np.min(full_summed_y))
    print("Max reconstructed signal before normalization:", np.max(full_summed_y))
    print("Min reconstructed signal before normalization:", np.min(full_summed_y))
       
