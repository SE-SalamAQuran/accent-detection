from models.preprocessing import Preprocessor
import librosa
from scipy.io import wavfile

path = 'data/Hebron/hebron_train041.wav'
rate = librosa.core.get_samplerate(path)
data = librosa.load(path)
p = Preprocessor(path)
print(p.Mfcc(rate, data))