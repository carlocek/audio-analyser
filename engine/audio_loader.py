class AudioLoader:
    def __init__(self, file_path):
        self._file_path = file_path
        self._audio_format = None
        self._sample_rate = None
        self._num_channels = None
        self._bit_depth = None
        self._data = []

    def _little_endian_to_int(self, byte_data):
        """Converts a byte array (little endian encoding) to an integer
        by shifting left each byte according to its position in the array."""
        result = 0
        for i, byte in enumerate(byte_data):
            result += byte << (i * 8)
        return result
    
    def _parse_fmt_chunk(self, data):
        """Analizza il chunk 'fmt' per estrarre il formato audio."""
        self._audio_format = self._little_endian_to_int(data[:2])  # 2 byte: formato audio
        self._num_channels = self._little_endian_to_int(data[2:4])  # 2 byte: numero di canali
        self._sample_rate = self._little_endian_to_int(data[4:8])  # 4 byte: frequenza di campionamento
        self._bit_depth = self._little_endian_to_int(data[14:16])  # 2 byte: bit per campione

    def _parse_data_chunk(self, data):
        """Analizza il chunk 'data' per estrarre i campioni audio."""
        bytes_per_sample = self._bit_depth // 8
        for i in range(0, len(data), bytes_per_sample):
            sample = self._little_endian_to_int(data[i:i + bytes_per_sample])
            # I valori PCM a 16 bit con segno vanno da -32768 a 32767
            if self._bit_depth == 16:
                if sample >= 32768:
                    sample -= 65536  # Corregge i valori negativi
            self._data.append(sample)

    def load(self):
        with open(self._file_path, "rb") as f:
            # Lettura dell'intestazione RIFF
            riff_header = f.read(12)
            if riff_header[:4] != b"RIFF" or riff_header[8:12] != b"WAVE":
                raise ValueError("Il file non è un file WAV valido.")
            
            while True:
                # Leggi il chunk successivo
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break  # Fine del file

                chunk_id = chunk_header[:4]  # ID del chunk
                chunk_size = self._little_endian_to_int(chunk_header[4:8])  # Dimensione del chunk

                if chunk_id == b"fmt ":
                    # Leggi il chunk "fmt"
                    fmt_data = f.read(chunk_size)
                    self._parse_fmt_chunk(fmt_data)
                elif chunk_id == b"data":
                    # Leggi il chunk "data" (dati PCM)
                    data_chunk = f.read(chunk_size)
                    self._parse_data_chunk(data_chunk)
                else:
                    # Salta altri chunk
                    f.seek(chunk_size, 1)
    
    @property
    def audio_format(self):
        return self._audio_format
    
    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def bit_depth(self):
        return self._bit_depth
    
    @property
    def data(self):
        return self._data

    def get_info(self):
        return {
            "audio_format": self.audio_format,
            "sample_rate": self.sample_rate,
            "num_channels": self.num_channels,
            "bit_depth": self.bit_depth,
            "num_samples": len(self.data),
        }
