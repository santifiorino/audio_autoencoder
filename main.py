import sys 

from autoencoder import AutoEncoder
from audio_utils import AudioUtils

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

audio_utils = AudioUtils(sample_rate, duration, hop_length_ms, Sclip)
X, phases, X_max, y = audio_utils.load_training_audio(audio_path)

ae = AutoEncoder(layers, learning_rate, X, X_max, y)
ae.train_model(epochs, batch_size)

og_phase, specgram = audio_utils.get_specgram("audio_autoencoder-main/wavs/audio.wav")
predicted_specgram = ae.predict(specgram)
audio_utils.save_specgram(predicted_specgram)