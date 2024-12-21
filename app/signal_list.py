import numpy as np
from engine.signal_generator import SignalGenerator

class SignalList:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.generator = SignalGenerator(sample_rate)
        self._signals = []

    def sum_signals(self, duration=1.0):
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        if not self._signals:
            return t, np.zeros_like(t)
        result = np.zeros_like(t)
        for signal in self.signals:
            _, y = self.generator.generate_signal(signal["frequency"], signal["amplitude"], signal["phase"], duration)
            result += y
        return t, result

    def add_signal(self, frequency, amplitude, phase):
        self._signals.append({
            "frequency": frequency,
            "amplitude": amplitude,
            "phase": phase
        })

    def update_signal(self, index, frequency, amplitude, phase):
        self._signals[index] = {
            "frequency": frequency,
            "amplitude": amplitude,
            "phase": phase
        }
    
    @property
    def signals(self):
        return self._signals
    
    @signals.setter
    def signals(self, value):
        self._signals = value