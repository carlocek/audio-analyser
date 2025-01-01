import numpy as np
from joblib import Parallel, delayed
import plotly.graph_objects as go


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
    spectrum = []
    for start in range(0, num_frequencies, block_size):
        end = min(start + block_size, num_frequencies)
        block_results = Parallel(n_jobs=-1, backend="loky")(
            delayed(calculate_frequency)(samples, num_samples, sample_rate, freq_index)
            for freq_index in range(start, end)
        )
        spectrum.extend(block_results)
        progress_text = f"Computing DFT: {end}/{num_frequencies}" if end != num_frequencies else "DFT Completed!"
        progress_bar.progress(end/num_frequencies, text=progress_text)

    return spectrum




def create_updatemenus(x, y, anim_duration):
    return [
        {
            "type": "buttons",
            "direction": "right",
            "x": x,
            "y": y,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": anim_duration, "redraw": True}, "fromcurrent": False}]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]
                },
                {
                    "label": "Reset",
                    "method": "update",
                    "args": [{"x": [[]], "y": [[]]}, {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]
                }
            ]
        }
    ]

# Funzione per creare la prima animazione
def wrapping_signal_fixedfreq_animation(t, summed_signal, test_freq):
    x_proj = summed_signal * np.cos(2 * np.pi * test_freq * t)
    y_proj = summed_signal * np.sin(2 * np.pi * test_freq * t)
    frames = [
        go.Frame(data=[
            go.Scatter(x=x_proj[:i], y=y_proj[:i], mode="markers", name="Projection Trace"),
            go.Scatter(x=[np.mean(x_proj[:i])], y=[np.mean(y_proj[:i])], mode="markers", name="Centroid"),
            go.Scatter(
                x=np.cos(np.linspace(0, 2 * np.pi, 100)),
                y=np.sin(np.linspace(0, 2 * np.pi, 100)),
                mode="lines", name="Reference Circle", line=dict(color="gray", dash="dash")
            ),
        ]) for i in range(1, len(t), 10)
    ]
    fig = go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode="markers", name="Projection Trace", marker=dict(color="blue", size=5)),
            go.Scatter(x=[], y=[], mode="markers", name="Centroid", marker=dict(color="red", size=8, symbol="cross")),
            go.Scatter(
                x=np.cos(np.linspace(0, 2 * np.pi, 100)),
                y=np.sin(np.linspace(0, 2 * np.pi, 100)),
                mode="lines", name="Reference Circle", line=dict(color="gray", dash="dash")
            ),
        ],
        layout=go.Layout(
            xaxis=dict(range=[-1.0, 1.0], dtick=1.0, showgrid=True),
            yaxis=dict(range=[-1.0, 1.0], dtick=1.0, showgrid=True, scaleanchor="x"),
            updatemenus=create_updatemenus(0.20, 1.20, 50),
            showlegend=False
        ),
        frames=frames
    )
    return fig

# Funzione per la seconda animazione
def wrapping_signal_varyingfreq_animation(t, summed_signal):
    frames_circle = []
    frames_centroid = []
    freq_values = np.linspace(1, 2000, 100)
    
    for test_freq in freq_values:
        x_proj = summed_signal * np.cos(2 * np.pi * test_freq * t)
        y_proj = summed_signal * np.sin(2 * np.pi * test_freq * t)
        centroid_x = np.mean(x_proj)

        frames_circle.append(
            go.Frame(data=[
                go.Scatter(x=x_proj, y=y_proj, mode="markers", name="Projection Trace"),
                go.Scatter(
                    x=np.cos(np.linspace(0, 2 * np.pi, 100)),
                    y=np.sin(np.linspace(0, 2 * np.pi, 100)),
                    mode="lines", name="Reference Circle", line=dict(color="gray", dash="dash")
                ),
            ])
        )
        frames_centroid.append(
            go.Frame(data=[
                go.Scatter(
                    x=freq_values[:len(frames_centroid) + 1], 
                    y=[np.mean(x_proj) for x_proj in [summed_signal * np.cos(2 * np.pi * freq * t) for freq in freq_values[:len(frames_centroid) + 1]]],
                    mode="lines+markers", name="Centroid x coordinate"),
            ])
        )

    fig_circle = go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode="markers", name="Projection Trace"),
            go.Scatter(
                x=np.cos(np.linspace(0, 2 * np.pi, 100)),
                y=np.sin(np.linspace(0, 2 * np.pi, 100)),
                mode="lines", name="Reference Circle", line=dict(color="gray", dash="dash")
            ),
        ],
        layout=go.Layout(
            xaxis=dict(range=[-1.0, 1.0], dtick=1.0, showgrid=True),
            yaxis=dict(range=[-1.0, 1.0], dtick=1.0, showgrid=True, scaleanchor="x"),
            updatemenus=create_updatemenus(0.45, 1.20, 500),
            showlegend=False
        ),
        frames=frames_circle
    )

    fig_centroid = go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode="lines+markers", name="Centroid X"),
        ],
        layout=go.Layout(
            xaxis=dict(range=[0, 2000], title="Test Frequency (Hz)", showgrid=True),
            yaxis=dict(range=[-1.0, 1.0], title="Centroid's x coordinate", showgrid=True),
            updatemenus=create_updatemenus(0.45, 1.20, 50),
            showlegend=False
        ),
        frames=frames_centroid
    )
    return fig_circle, fig_centroid
