import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

# 1. Load the recorded audio file
file_path = 'test_breath.wav'
audio, sr = librosa.load(file_path, sr=None)

# 2. Apply a Bandpass Filter ONLY (100 Hz - 2500 Hz)
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

filtered_audio = bandpass_filter(audio, lowcut=100.0, highcut=2500.0, fs=sr)

# 3. Normalize the filtered audio (boost volume without clipping)
normalized_audio = filtered_audio / np.max(np.abs(filtered_audio))

# 4. Save the processed audio
output_path = 'clean_breath.wav'
sf.write(output_path, normalized_audio, sr)
print("File successfully bandpass-filtered and saved.")