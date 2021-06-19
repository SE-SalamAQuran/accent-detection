import statistics
from sklearn.mixture import GaussianMixture as GMM
from scipy.io import wavfile
from scipy.signal import butter, lfilter, freqz
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import os
from python_speech_features import mfcc, ssc, logfbank, fbank
import librosa

warnings.filterwarnings('ignore')
# directory = 'data/Jerusalem'
# for filename in os.listdir(directory):
#     if filename.endswith(".wav"):
#          print(os.path.join(directory, filename))
#     else:
#         continue

path_j = 'data/Jerusalem/jerusalem_train041.wav'
path_h = 'data/Hebron/hebron_test021.wav'
path_n = 'data/Nablus/nablus_test023.wav'
path_r = 'data/Ramallah_Reef/ramallah-reef_test024.wav'

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def preprocessing(path):
    #Read audio data from path
    rate, data = wavfile.read(path)


    #Feature1: MFCC

    mel_fcc = mfcc(data, rate, nfft=4096)
    # print("Unfiltered MFCC", np.round(np.average(mel_fcc), 3))
    #
    X =  (np.mean(mel_fcc)) / (np.median(mel_fcc))
    print("MFCC: " ,X)

    # #Feature2: LOG Filter Banks Energies

    lfp = logfbank(data, rate, nfft=4096)
    Y =  (np.mean(lfp)) / (np.median(lfp))
    #
    print("LOG: ",Y)
    #
    # #Feature3: Spectral Sub-band Centroids
    spec = ssc(signal=data, samplerate=rate, nfft=4096)
    # #
    print(np.median(spec))
    #


print("Jerusalem")
preprocessing(path_j)
print('Hebron')
preprocessing(path_h)
print('Nablus')
preprocessing(path_n)
print('Ramallah')
preprocessing(path_r)