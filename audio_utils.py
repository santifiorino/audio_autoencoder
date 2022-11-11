# Utils
from glob import glob

# Numbers
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import IPython.display
from IPython.display import Image, Audio, HTML
import librosa.display

# Machine learning
import torch
import torchaudio
from torchaudio.functional import resample
from sklearn.model_selection import train_test_split
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, TensorDataset, DataLoader

# Audio
import torchaudio
import librosa

def format_waveform(waveform, original_sr, target_sr, duration):
    waveform = resample(waveform, original_sr, target_sr) # Resample to custom sample rate
    waveform = torch.mean(waveform, dim=0) # Convert to mono
    waveform = waveform[:target_sr * duration] # Trim to custom duration
    return waveform

def get_waveform(path, target_sr, duration):
    waveform, original_sr = torchaudio.load(path)
    waveform = format_waveform(waveform, original_sr, target_sr, duration)
    return waveform

def fourier(waveform, win_length, hop_length, Sclip):
    F = torch.stft(waveform, n_fft=win_length, hop_length=hop_length, win_length=win_length, return_complex=True).T
    S = 10*torch.log10(torch.abs(F)**2)
    S = S.clip(Sclip, None) - Sclip
    return torch.angle(F), S

def get_specgram(path, win_length, hop_length, Sclip, target_sr, duration):
    waveform = get_waveform(path, target_sr, duration)
    phase, S = fourier(waveform, win_length, hop_length, Sclip)
    return phase, S
    
def load_training_audio(path, win_length, hop_length, Sclip, target_sr, duration):
    path += "/*.wav"
    X = []
    y = []
    phases = []
    for i, filename in enumerate(glob(path)):
        phase, S = get_specgram(filename, win_length, hop_length, Sclip, target_sr, duration)
        phases.append(phase)
        X.append(S)
        y.append(torch.ones(S.shape[0]) * i)
    phases = torch.vstack(phases)
    X = torch.vstack(X)
    X_max = X.max()
    X = X / X_max
    y = torch.hstack(y)
    return X, phases, X_max, y
    
def save_specgram(specgram, hop_length):
    plt.figure(figsize=(14, 4))
    librosa.display.specshow(specgram.detach().numpy().T, y_axis='linear', x_axis='time', hop_length=hop_length);
    plt.colorbar();
    plt.savefig("test.png")