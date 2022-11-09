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

class AudioUtils():
    def __init__(self, sample_rate=22050, duration=120, hop_length_ms=20, Sclip=-60):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.hop_length = int(sample_rate * hop_length_ms / 1000)
        self.Sclip = Sclip
        self.win_length = 4 * self.hop_length
        
    def load_training_audio(self, path):
        path += "/*.wav"
        X = []
        y = []
        phases = []
        for i, filename in enumerate(glob(path)):
            phase, S = self.get_specgram(filename)
            phases.append(phase)
            X.append(S)
            y.append(torch.ones(S.shape[0]) * i)
        phases = torch.vstack(phases)
        X = torch.vstack(X)
        X_max = X.max()
        X = X / X_max
        y = torch.hstack(y)
        return X, phases, X_max, y
        
    def fourier(self, waveform):
        F = torch.stft(waveform, n_fft=self.win_length, hop_length=self.hop_length, win_length=self.win_length, return_complex=True).T
        S = 10*torch.log10(torch.abs(F)**2)
        S = S.clip(self.Sclip, None) - self.Sclip
        return torch.angle(F), S
    
    def format_waveform(self, waveform, original_sr):
        waveform = resample(waveform, original_sr, self.sample_rate) # Resample to custom sample rate
        waveform = torch.mean(waveform, dim=0) # Convert to mono
        waveform = waveform[:self.sample_rate * self.duration] # Trim to custom duration
        return waveform

    def get_waveform(self, path):
        waveform, original_sr = torchaudio.load(path)
        waveform = self.format_waveform(waveform, original_sr)
        return waveform

    def get_specgram(self, path):
        waveform = self.get_waveform(path)
        phase, S = self.fourier(waveform)
        return phase, S
        
    def save_specgram(self, specgram):
        plt.figure(figsize=(14, 4))
        librosa.display.specshow(specgram.detach().numpy().T, y_axis='linear', x_axis='time', hop_length=self.hop_length);
        plt.colorbar();
        plt.savefig("test.png")