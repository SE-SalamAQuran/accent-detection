import librosa
import IPython.display as ipd
#display waveform
import matplotlib.pyplot as plt
import librosa.display
audio_path = 'data/Jerusalem/jerusalem_train041.wav'

x , sr = librosa.load(audio_path)
ipd.Audio(audio_path)
ipd.Audio.autoplay_attr(audio_path)
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
#display Spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
#If to pring log of frequencies
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
