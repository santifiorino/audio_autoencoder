import sys 

from autoencoder import AutoEncoder
from audio_utils import *

args_txt = sys.argv[1]

with open (args_txt) as f:
    layers = f.readline().split(' ')
    layers = [int(i) for i in layers]
    sample_rate = int(f.readline()) 
    duration = int(f.readline())
    hop_length_ms = int(f.readline())
    Sclip = int(f.readline())
    learning_rate = float(f.readline())
    audio_path = f.readline()[:-1]
    epochs = int(f.readline())
    batch_size = int(f.readline())

hop_length = int(sample_rate * hop_length_ms / 1000)
win_length = hop_length * 4
X, phases, X_max, y = load_training_audio(audio_path, win_length, hop_length, Sclip, sample_rate, duration)

ae = AutoEncoder(layers, learning_rate, X, X_max, y)
ae.train_model(epochs, batch_size)

audio_path = "audio_autoencoder-main/wavs/audio.wav"
og_phase, specgram = get_specgram(audio_path, win_length, hop_length, Sclip, sample_rate, duration)
predicted_specgram = ae.predict(specgram)
save_specgram(predicted_specgram, hop_length)