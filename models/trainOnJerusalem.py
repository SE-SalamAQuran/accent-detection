import warnings
import nltk
import numpy as np
from scipy.io import wavfile
import os

warnings.filterwarnings("ignore")

directory = 'data/Jerusalem'
rates = {}
r = list(rates)
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        samplerate, data = wavfile.read(directory + '/' + filename)
        r.append(samplerate)
    else:
        continue
print(r)
print(data)