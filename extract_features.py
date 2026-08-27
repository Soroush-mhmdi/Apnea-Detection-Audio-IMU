import librosa
import librosa.display
import matplotlib.pyplot as plt

# 1. Load the clean audio
file_path = 'clean_breath.wav'
audio, sr = librosa.load(file_path, sr=None)

# --- THE FIX: Trim the first 0.5 seconds to remove the microphone click! ---
audio = audio[int(0.5 * sr):]

# 2. Extract standard MFCCs
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

# 3. Calculate Delta MFCCs (Derivative)
delta_mfccs = librosa.feature.delta(mfccs)

# 4. Plot both side-by-side
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
librosa.display.specshow(mfccs, x_axis='time', sr=sr, cmap='viridis')
plt.title('Standard MFCC (First 0.5s trimmed)')
plt.colorbar(format='%+2.0f')

plt.subplot(2, 1, 2)
librosa.display.specshow(delta_mfccs, x_axis='time', sr=sr, cmap='magma')
plt.title('Delta MFCC (Without the initial mic shock)')
plt.colorbar(format='%+2.0f')
plt.xlabel('Time (seconds)')

plt.tight_layout()
plt.show()