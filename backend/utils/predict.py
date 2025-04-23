import torch
import numpy as np
import librosa
import soundfile as sf
import os
import gdown  # For downloading from Google Drive

# Define model path
MODEL_PATH = "backend/model/best_model.pth"

# Check if model exists, if not, download it
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    # Google Drive link (replace with your file ID)
    gdown.download("https://drive.google.com/uc?id=1Z_BV5E5pkR9dHwIqgrBTMh-iuKBwD6ud", MODEL_PATH, quiet=False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
N_FFT = 1024
HOP_LENGTH = 256

# Load model
model = UNet().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
model.eval()

def process_chunk(chunk):
    """Process a chunk with proper padding and denoising."""
    noisy_spec = librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP_LENGTH)
    noisy_mag = np.abs(noisy_spec)

    with torch.no_grad():
        input_tensor = torch.FloatTensor(noisy_mag).unsqueeze(0).unsqueeze(0).to(DEVICE)
        clean_mag = model(input_tensor).squeeze().cpu().numpy()

    denoised_chunk = librosa.griffinlim(
        clean_mag, n_iter=30, hop_length=HOP_LENGTH, win_length=N_FFT, length=len(chunk)
    )
    return denoised_chunk, noisy_mag, clean_mag

def denoise_audio(filepath):
    """Full pipeline for chunk-wise denoising and final result saving."""
    noisy_audio, _ = librosa.load(filepath, sr=SAMPLE_RATE)
    total_samples = len(noisy_audio)
    total_chunks = int(np.ceil(total_samples / CHUNK_SIZE))
    denoised_audio = np.zeros_like(noisy_audio)

    chunk_results = []

    for idx in range(total_chunks):
        start = idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total_samples)
        chunk = noisy_audio[start:end]
        denoised_chunk, noisy_mag, clean_mag = process_chunk(chunk)
        denoised_audio[start:end] = denoised_chunk[:end-start]
        chunk_results.append((chunk, denoised_chunk[:end-start], noisy_mag, clean_mag))

    # Save final output
    out_path = filepath.replace(".wav", "_denoised.wav")
    sf.write(out_path, denoised_audio, SAMPLE_RATE)

    return out_path, noisy_audio, denoised_audio, chunk_results
