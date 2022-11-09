#ip install pytorch-lightning

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

class Encoder(nn.Module):
    def __init__(self, layers_size):
        super().__init__()
        layers = []
        for i in range(len(layers_size)-1):
            layers.append(nn.Linear(layers_size[i], layers_size[i+1]))
            layers.append(nn.BatchNorm1d(layers_size[i+1]))
            layers.append(nn.ELU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, layers_size):
        super().__init__()
        layers = []
        for i in range(len(layers_size)-1, 1, -1):
            layers.append(nn.Linear(layers_size[i], layers_size[i-1]))
            layers.append(nn.BatchNorm1d(layers_size[i-1]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(layers_size[1], layers_size[0]))
        layers.append(nn.BatchNorm1d(layers_size[0]))
        # layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class AutoEncoder(pl.LightningModule):
    def __init__(self, layers_size, learning_rate, X, X_max, y):
        super().__init__()
        self.layers_size = layers_size
        self.encoder = Encoder(self.layers_size)
        self.decoder = Decoder(self.layers_size)
        self.X = X
        self.X_max = X_max
        self.y = y
        self.learning_rate = learning_rate
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, x)
        self.log("loss", loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def train_model(self, epochs=120, batch_size=512):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y,  test_size=0.05, shuffle=True)
        silence = np.zeros([int(X_train.shape[0]*0.1), X_train.shape[1]])
        X_data = torch.tensor(np.vstack([silence, X_train])).float()
        trainer = pl.Trainer(max_epochs=epochs)
        dataset = TensorDataset(X_data, X_data)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        trainer.fit(self, dataloader)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path)['state_dict'])

    def predict(self, specgram):
        specgram = specgram / self.X_max
        S_hat = self(specgram)
        return S_hat * self.X_max