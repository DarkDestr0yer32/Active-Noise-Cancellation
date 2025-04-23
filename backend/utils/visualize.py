# visualize.py
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

SAMPLE_RATE = 16000
HOP_LENGTH = 256

def generate_all_plots(filepath, denoised_audio, noisy_audio, chunk_results):
    base = os.path.splitext(os.path.basename(filepath))[0]
    folder = os.path.dirname(filepath)
    plot_paths = []

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    plt.plot(noisy_audio)
    plt.title("Noisy Full Waveform")

    plt.subplot(2, 2, 2)
    plt.plot(denoised_audio)
    plt.title("Denoised Full Waveform")

    plt.subplot(2, 2, 3)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(noisy_audio)), ref=np.max),
                              sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='log')
    plt.title("Noisy Full Spectrogram")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(denoised_audio)), ref=np.max),
                              sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='log')
    plt.title("Denoised Full Spectrogram")
    plt.colorbar()
    plt.tight_layout()

    full_plot_path = os.path.join(folder, f"{base}_full_plot.png")
    plt.savefig(full_plot_path)
    plot_paths.append(full_plot_path)
    plt.close()

    for i, (chunk, dchunk, noisy_mag, clean_mag) in enumerate(chunk_results):
        plt.figure(figsize=(15, 8))

        plt.subplot(2, 2, 1)
        plt.plot(chunk)
        plt.title(f"Chunk {i+1} - Noisy Waveform")

        plt.subplot(2, 2, 2)
        plt.plot(dchunk)
        plt.title(f"Chunk {i+1} - Denoised Waveform")

        plt.subplot(2, 2, 3)
        librosa.display.specshow(librosa.amplitude_to_db(noisy_mag, ref=np.max), sr=SAMPLE_RATE,
                                 hop_length=HOP_LENGTH, x_axis='time', y_axis='log')
        plt.colorbar()
        plt.title(f"Chunk {i+1} - Noisy Spectrogram")

        plt.subplot(2, 2, 4)
        librosa.display.specshow(librosa.amplitude_to_db(clean_mag, ref=np.max), sr=SAMPLE_RATE,
                                 hop_length=HOP_LENGTH, x_axis='time', y_axis='log')
        plt.colorbar()
        plt.title(f"Chunk {i+1} - Denoised Spectrogram")

        plt.tight_layout()
        chunk_path = os.path.join(folder, f"{base}_chunk_{i+1}.png")
        plt.savefig(chunk_path)
        plot_paths.append(chunk_path)
        plt.close()

    return plot_paths
