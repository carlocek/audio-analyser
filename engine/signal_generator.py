import numpy as np

class SignalGenerator:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
    
    def generate_signal(self, frequency=1.0, amplitude=1.0, phase=0.0, t = np.linspace(0, 1, int(44100 * 1), endpoint=False)):
        return amplitude * np.cos((2 * np.pi * frequency * t) + phase)