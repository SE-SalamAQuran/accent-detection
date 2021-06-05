#This is called the spectral gating algorithm which removes noises from an audio signal
import IPython
from scipy.io import wavfile
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import librosa
import time
from datetime import timedelta as td

rate, data = wavfile.read('data/Hebron/hebron_train042.wav')

IPython.display.Audio(data, rate)