import numpy as np
import scipy
import matplotlib.pyplot as plt
from utils import *
import librosa

# setup parameter
# parameter for STOI
fs_stoi     = 10000     # sample rate (Hz)
n_frame     = 256       # length of frame (samples)
k           = 512       # length of FFT (samples)
j           = 15        # number of 1/3 oct. bands
mn          = 150       # center freq. of first 1/3 oct. bands (Hz)
n           = 30        # number of frames
dyn_range   = 40        # speech dynamic range (dB)

# parameter for EC-parameter speech
tau_min     = -0.001    # minimum interaural delay compensation (s)
tau_max     = 0.001     # maximum interaural delay compensation (s)
ntaus       = 100       # number of tau values
gamma_min   = -20       # minimum interaural level compensation (dB)
gamma_max   = 20        # maximum interaural level compensation (dB)
ngammas     = 40        # number of gamma values

# parameter for jitter
sigma_delta_0   = 65e-6     # ITD compensation standard deviation (s)
sigma_epsilon_0 = 1.5       # ITD compensation standard deviation
alpha_0_dB      = 13        # constant for level shift deviation (dB)
tau_0           = 1.6e-3    # constant for time shift deviation (s)
p               = 1.6       # constant for level shift deviation

# prepare evaluation signals
# import all audio
fs, x = scipy.io.wavfile.read("output.wav")
fs, y = scipy.io.wavfile.read("output-noised.wav")

xl = x[:,0]
xr = x[:,1]
yl = y[:,0]
yr = y[:,1]

# check the data type of all audio
if xl.dtype != np.float32:
    xl = xl.astype(np.float32, order='C') / 32768.0
if xr.dtype != np.float32:
    xr = xr.astype(np.float32, order='C') / 32768.0
if yl.dtype != np.float32:
    yl = yl.astype(np.float32, order='C') / 32768.0
if yr.dtype != np.float32:
    yr = yr.astype(np.float32, order='C') / 32768.0

# check the frequency sampling for processing the audio data
if fs != fs_stoi:
    xl = librosa.resample(xl, fs, fs_stoi)
    xr = librosa.resample(xr, fs, fs_stoi)
    yl = librosa.resample(yl, fs, fs_stoi)
    yr = librosa.resample(yr, fs, fs_stoi)

# remove silent frame of signal
xl, xr, yl, yr  = remove_silent_frame(xl, xr, yl, yr, dyn_range, n_frame, n_frame/2)

# handle case when signals is zeros
if abs(np.log10(np.linalg.norm(xl, 2)/np.linalg.norm(yl, 2))) > 5.0 or abs(np.log10(np.linalg.norm(xr, 2)/np.linalg.norm(yr, 2))) > 5.0:
    sii = 0
