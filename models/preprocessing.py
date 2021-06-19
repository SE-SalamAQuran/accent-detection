import sklearn.preprocessing
from scipy.io import wavfile
from scipy import signal
import numpy as np
import requests
import scipy.signal._peak_finding
import warnings
import pandas as pd
import scipy.ndimage.filters as fi
import stft
import matplotlib.pyplot as plt
import noisereduce as nr
import librosa, librosa.display
import os
import math

warnings.filterwarnings('ignore')
#

class Preprocessor():
    def __init__(self, path):
        rate, audio = wavfile.read(path)
        peaks, props = scipy.signal.find_peaks(audio)



#Step1: Normalization
    def normalize(x, axis=0):
        return sklearn.preprocessing.minmax_scale(x, axis)

#Step2: Pre-Emphasis
    def pre_emphasis(self, path):
        y, sr = librosa.load(path)
        y_filt = librosa.effects.preemphasis(y)
# and plot the results for comparison
        S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        S_preemph = librosa.amplitude_to_db(np.abs(librosa.stft(y_filt)), ref=np.max)
        return S_preemph




#Feature Extraction

#Step1: Zero-Crosssing Rate
    def z_cross_rate(self, data, n0, n1):
        #Should return zero crossings found in the audio signal between n0, n1
        return librosa.zero_crossings(data[n0:n1], pad=False)

#Step2: Spectral roll-off

    def roll_off(self, data, samp_rate):
        # Approximate minimum frequencies with roll_percent=0.1

        return librosa.feature.spectral_rolloff(y=data, sr=samp_rate, roll_percent=0.1)



#Step3: MFCC (Mel-frequency cepstral coefficients)
    def Mfcc(self, data, samp_rate):
        return librosa.feature.mfcc(data, sr=samp_rate)

#Step4: Chroma Frequencies
    def chroma_freq(self, data, samp_rate, hop_length = 512):
    # returns normalized energy for each chroma bin at each frame.
        return librosa.feature.chroma_stft(data, sr=samp_rate, hop_length=hop_length)


    def plot_signal(self, title, x_label, y_label, data):
        plt.figure()
        plt.plot(data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

    def logPow(self, data):
        return [math.log(math.pow(x,2)) for x in data]
