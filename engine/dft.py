import numpy as np
from joblib import Parallel, delayed
import plotly.graph_objects as go


def calculate_frequency(samples, num_samples, sample_rate, freq_index):
    angles = 2 * np.pi * freq_index * np.arange(num_samples) / num_samples
    real_part = np.sum(samples * np.cos(angles))
    imag_part = np.sum(samples * np.sin(angles))
    sample_centre = np.array([real_part, imag_part]) / num_samples

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




def create_updatemenus(x, y, frame_duration, transition_duration):
    return [
        {
            "active": -1,
            "type": "buttons",
            "direction": "right",
            "x": x,
            "y": y,
            "xanchor": "left",
            "yanchor": "top",
            "buttons": [
                # Play/Pause Button
                {
                    "label": "Play/Pause",
                    "method": "animate",
                    "args": [
                        None,  # animate every frame
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "transition": {"duration": transition_duration},
                            "fromcurrent": True,  # start from last drawn frame
                        }
                    ],
                    "args2": [
                        [None],  # stop all frames
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": False},
                            "transition": {"duration": 0},
                        }
                    ]
                }
                # Reset Button
                # {
                #     "label": "Reset",
                #     "method": "animate",
                #     "args": [
                #         None,
                #         {
                #             "frame": {"duration": frame_duration, "redraw": True},
                #             "transition": {"duration": transition_duration},
                #             "fromcurrent": False,
                #         }
                #     ]
                # }
            ]
        }
    ]

def wrapping_signal_fixedfreq_animation(t, signal, test_freq):
    print("x")
    x_proj = signal * np.cos(2 * np.pi * test_freq * t)
    print("y")
    y_proj = signal * np.sin(2 * np.pi * test_freq * t)
    print("z")
    frames = [
        go.Frame(data=[
            go.Scatter(x=x_proj[:i], y=y_proj[:i], mode="markers", name="Projection Trace"),
            go.Scatter(x=[np.mean(x_proj[:i])], y=[np.mean(y_proj[:i])], mode="markers", name="Centroid"),
            go.Scatter(
                x=max(np.abs(signal))*np.cos(np.linspace(0, 2 * np.pi, 100)),
                y=max(np.abs(signal))*np.sin(np.linspace(0, 2 * np.pi, 100)),
                mode="lines", name="Reference Circle", line=dict(color="gray")
            ),
        ]) for i in range(1, len(t), 10)
    ]
    print("b")
    fig = go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode="markers", name="Projection Trace", marker=dict(size=3)),
            go.Scatter(x=[], y=[], mode="markers", name="Centroid", marker=dict(color="red", size=8, symbol="cross")),
            go.Scatter(
                x=max(np.abs(signal))*np.cos(np.linspace(0, 2 * np.pi, 100)),
                y=max(np.abs(signal))*np.sin(np.linspace(0, 2 * np.pi, 100)),
                mode="lines", name="Reference Circle", line=dict(color="gray")
            ),
        ],
        layout=go.Layout(
            xaxis=dict(range=[-max(np.abs(signal)), max(np.abs(signal))], dtick=1.0, showgrid=True),
            yaxis=dict(range=[-max(np.abs(signal)), max(np.abs(signal))], dtick=1.0, showgrid=True, scaleanchor="x"),
            updatemenus=create_updatemenus(0.0, 1.20, 50, 45),
            showlegend=False
        ),
        frames=frames
    )
    print("c")
    return fig


def wrapping_signal_varyingfreq_animation(t, signal):
    num_samples = len(signal)
    frames = []
    freq_values = np.arange(0, 2000, 10)
    
    for test_freq in freq_values:
        phase_correction = np.angle(signal[0] + 1j * signal[1])
        x_proj = signal * np.cos(2 * np.pi * test_freq * (t))
        y_proj = signal * np.sin(2 * np.pi * test_freq * (t))
        centroid_x = np.mean(x_proj)
        centroid_y = np.mean(y_proj)
        is_0hz = test_freq==0
        is_nyquist = test_freq==num_samples//2 and num_samples%2==0
        amplitude_scale = 1 if is_0hz or is_nyquist else 2

        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=x_proj, y=y_proj, mode="markers", name="Projection Trace"),
                    go.Scatter(x=[centroid_x], y=[centroid_y], mode="markers", name="Centroid", marker=dict(color="red", size=8, symbol="cross")),
                    go.Scatter(
                        x=max(np.abs(signal))*np.cos(np.linspace(0, 2 * np.pi, 100)),
                        y=max(np.abs(signal))*np.sin(np.linspace(0, 2 * np.pi, 100)),
                        mode="lines", name="Reference Circle", line=dict(color="gray")
                    ),
                    go.Scatter(
                        x=freq_values[:len(frames) + 1], 
                        y=[amplitude_scale*np.mean(x_proj) for x_proj in [signal * np.cos(2 * np.pi * freq * (t)) for freq in freq_values[:len(frames) + 1]]],
                        mode="lines+markers", name="Centroid x coordinate"
                    ),
                ],
                layout=go.Layout(
                    annotations=[dict(x=0.6, y=1.15, xref="paper", yref="paper", text=f"Wrapping Frequency: {test_freq:.2f} Hz", showarrow=False, font=dict(size=14, color="white"))]
                )
            )
        )

    fig = go.Figure(
        data=[
            # Circle Plot (pannello sinistro)
            go.Scatter(x=[], y=[], mode="markers", name="Projection Trace", marker=dict(size=3), xaxis="x1", yaxis="y1"),
            go.Scatter(x=[], y=[], mode="markers", name="Centroid", marker=dict(color="red"), xaxis="x1", yaxis="y1"),
            go.Scatter(
                x=max(np.abs(signal))*np.cos(np.linspace(0, 2 * np.pi, 100)),
                y=max(np.abs(signal))*np.sin(np.linspace(0, 2 * np.pi, 100)),
                mode="lines", name="Reference Circle", line=dict(color="gray"), xaxis="x1", yaxis="y1",
            ),
            # Centroid Plot (pannello destro)
            go.Scatter(x=[], y=[], mode="lines+markers", name="Centroid X", marker=dict(size=3), xaxis="x2", yaxis="y2"),
        ],
        layout=go.Layout(
            xaxis=dict(range=[-max(np.abs(signal)), max(np.abs(signal))], dtick=1.0, showgrid=True, domain=[0, 0.45]),
            yaxis=dict(range=[-max(np.abs(signal)), max(np.abs(signal))], dtick=1.0, showgrid=True, scaleanchor="x", domain=[0, 1.0]),
            xaxis2=dict(range=[0, 2000], title="Wrapping Frequency (Hz)", showgrid=True, domain=[0.55, 1.0]),
            yaxis2=dict(range=[-1.0, 1.0], title="Centroid x-coordinate", showgrid=True, anchor="x2", domain=[0, 1.0], position=0.55),
            updatemenus=create_updatemenus(0.0, 1.20, 1000, 500),
            showlegend=False
        ),
        frames=frames,
    )

    return fig