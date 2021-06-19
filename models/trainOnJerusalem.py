import sklearn.preprocessing
from scipy.io import wavfile
from scipy.signal import butter, lfilter, freqz
import numpy as np
import scipy.signal._peak_finding
import warnings
import pandas as pd
import scipy.ndimage.filters as fi
import matplotlib.pyplot as plt
import os
from python_speech_features import mfcc, ssc, logfbank
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
path_r = 'data/Ramallah_Reef/ramallah-reef_train045.wav'
path_n = 'data/Nablus/nablus_test023.wav'

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
    #Removing the noise
    #Step1: Apply Butter LPF to the signal
    order = 6
    cutoff = 50
    b, a = butter_lowpass(cutoff, rate, order)
    w, h = freqz(b, a, worN=8000)


    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    T = len(data) / float(rate)  # seconds

    n = int(T * rate)  # total number of samples
    t = np.linspace(0, T, n, endpoint=False)
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    data = np.sin(1.2 * 2 * np.pi * t) + 1.5 * np.cos(9 * 2 * np.pi * t) + 0.5 * np.sin(12.0 * 2 * np.pi * t)

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(data, cutoff, rate, order)



    #Feature1


    mel_fcc = mfcc(data, rate, nfft=4096)
    print("Unfiltered MFCC",np.average(mel_fcc))

    filtered_mfcc = mfcc(y, rate, nfft=4096)
    print("Filtered MFCC", np.average(filtered_mfcc))

    #Feature2

    l = logfbank(data, rate, 4096)
    print("Unfiltered Log", np.average(l))

    log = logfbank(y, samplerate=rate, nfft=4096)

    print("Filtered Log", np.average(log))
    #Feature3
    # spec = ssc(signal=data, samplerate=rate, nfft=4096)
    #
    # print(np.average(spec))
    #
    # #Feature4
    #
    # n0 = 0
    # n1 = 10
    # print(np.average(librosa.zero_crossings(y[n0:n1], pad=False)))

print("Jerusalem")
preprocessing(path_j)
print('Hebron')
preprocessing(path_h)
print('Nablus')
preprocessing(path_n)
print('Ramallah')
preprocessing(path_r)