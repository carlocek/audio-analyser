import numpy as np

class SignalGenerator:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
    
    def generate_signal(self, frequency, amplitude=1.0, phase=0.0, duration=1.0):
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        return t, amplitude * np.sin(2 * np.pi * frequency * t + phase)