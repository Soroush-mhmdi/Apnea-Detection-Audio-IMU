import librosa
import numpy as np

# 1. Load the clean audio
file_path = 'clean_breath.wav'
audio, sr = librosa.load(file_path, sr=None)

# 2. Trim the first 0.5 seconds to remove the mic click
audio = audio[int(0.5 * sr):]

# 3. Extract MFCCs and Delta MFCCs
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
delta_mfccs = librosa.feature.delta(mfccs)

# 4. Combine them into one powerful feature matrix
# We stack them vertically (13 MFCCs + 13 Deltas = 26 features per time frame)
combined_features = np.vstack([mfccs, delta_mfccs])

# 5. Save the data for the Machine Learning model
dataset_name = 'breath_features.npy'
np.save(dataset_name, combined_features)

print("Dataset successfully built!")
print(f"File saved as: {dataset_name}")
print(f"Shape of the data (Features, Time frames): {combined_features.shape}")