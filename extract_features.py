import librosa
import librosa.display
import matplotlib.pyplot as plt

# 1. Load the clean audio
file_path = 'clean_breath.wav'
audio, sr = librosa.load(file_path, sr=None)

# 2. Extract MFCCs (This is the actual data the ML model will learn from)
# We extract 13 key features which is standard for human sounds
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

# 3. Plot the MFCC matrix
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=sr, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC Features - How the AI sees your breath')
plt.xlabel('Time')
plt.ylabel('MFCC Coefficients')
plt.tight_layout()
plt.show()