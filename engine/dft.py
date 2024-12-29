import numpy as np
import streamlit as st
from tqdm import tqdm
from joblib import Parallel, delayed


def dft(samples, sample_rate):
    """
    Perform a DFT using the circular representation of the signal.
    """
    num_samples = len(samples)
    num_frequencies = num_samples // 2 + 1
    spectrum = []
    frequency_step = sample_rate / num_samples  # Frequency resolution

    for freq_index in range(num_frequencies):
        # print(f"{freq_index} / {num_frequencies}")
        # Calculate the complex sum for this frequency
        sample_sum = np.zeros(2)  # 2D vector for circular representation [real, imag]
        for n in range(num_samples):
            angle = (n / num_samples) * (2 * np.pi) * freq_index
            test_point = np.array([np.cos(angle), np.sin(angle)])
            sample_sum += test_point * samples[n]

        # Compute the circular center
        sample_centre = sample_sum / num_samples

        # Handle amplitude scaling for 0 Hz and Nyquist frequencies
        is_0hz = freq_index == 0
        is_nyquist = freq_index == num_frequencies - 1 and num_samples % 2 == 0
        amplitude_scale = 1 if is_0hz or is_nyquist else 2
        amplitude = np.linalg.norm(sample_centre) * amplitude_scale

        # Frequency and phase
        frequency = freq_index * frequency_step
        phase = -np.arctan2(sample_centre[1], sample_centre[0])

        spectrum.append({"frequency": frequency, "amplitude": amplitude, "phase": phase})

    return spectrum

def calculate_frequency(samples, num_samples, sample_rate, freq_index):
    # print(f"\n{freq_index} / {num_samples // 2 + 1}")
    angles = 2 * np.pi * freq_index * np.arange(num_samples) / num_samples
    # print(angles.shape)
    real_part = np.sum(np.cos(angles) * samples)
    imag_part = np.sum(np.sin(angles) * samples)
    # print(real_part, imag_part)
    sample_centre = np.array([real_part, imag_part]) / num_samples
    # print(sample_centre)

    is_0hz = freq_index == 0
    is_nyquist = freq_index == num_samples // 2 and num_samples % 2 == 0
    amplitude_scale = 1 if is_0hz or is_nyquist else 2
    amplitude = np.linalg.norm(sample_centre) * amplitude_scale

    freq_resolution = sample_rate / num_samples
    frequency = freq_index * freq_resolution
    phase = -np.arctan2(sample_centre[1], sample_centre[0])

    return {"frequency": frequency, "amplitude": amplitude, "phase": phase}

def dft_parallel(samples, sample_rate, progress_bar, block_size):
    num_samples = len(samples)
    num_frequencies = num_samples // 2 + 1

    # progress_bar = st.progress(0, "Computing DFT...")

    # spectrum = []
    # for i, result in enumerate(
    #     Parallel(n_jobs=-1)(
    #         delayed(calculate_frequency)(samples, num_samples, sample_rate, freq_index)
    #         for freq_index in range(num_frequencies)
    #     )
    # ):
    #     spectrum.append(result)
    #     # Update the progress bar
    #     progress_bar.progress((i + 1) / num_frequencies, text="Computing DFT...")
    
    spectrum = []
    for start in range(0, num_frequencies, block_size):
        end = min(start + block_size, num_frequencies)
        block_results = Parallel(n_jobs=-1, backend="loky")(
            delayed(calculate_frequency)(samples, num_samples, sample_rate, freq_index)
            for freq_index in range(start, end)
        )
        spectrum.extend(block_results)
        progress_bar.progress((end) / num_frequencies, text=f"Computing DFT: {end}/{num_frequencies}")

    return spectrum