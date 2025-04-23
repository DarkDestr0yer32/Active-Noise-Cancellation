import librosa
import numpy as np

def audio_to_mag(filepath, sr=16000, n_fft=1024, hop_length=256):
    y, _ = librosa.load(filepath, sr=sr)
    spec = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(spec)
    return mag, y.shape[0]