import librosa
import numpy as np

def process_audio(file_path, label):
    print(f"Processing {file_path}...")
    # Load audio
    audio, sr = librosa.load(file_path, sr=None)
    
    # Trim the first 0.5 seconds to remove mic click
    audio = audio[int(0.5 * sr):]
    
    # Extract features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs)
    
    # Combine (Shape: 26, Time_Frames)
    combined_features = np.vstack([mfccs, delta_mfccs])
    
    # Transpose so rows are time frames and columns are 26 features -> (Time_Frames, 26)
    features = combined_features.T
    
    # Create an array of labels for all these time frames
    labels = np.full(features.shape[0], label)
    
    return features, labels

# Process all 4 files with their corresponding labels
X_normal, y_normal = process_audio('normal_breath.wav', 0)
X_deep, y_deep = process_audio('deep_breath.wav', 0)
X_snore, y_snore = process_audio('snoring_sim.wav', 0)
X_apnea, y_apnea = process_audio('apnea_sim.wav', 1)

# Combine everything into one giant dataset
X_all = np.vstack([X_normal, X_deep, X_snore, X_apnea])
y_all = np.concatenate([y_normal, y_deep, y_snore, y_apnea])

# Save the final dataset arrays
np.save('X_data.npy', X_all)
np.save('y_labels.npy', y_all)

print("\nDataset successfully built and labeled!")
print(f"Total Features Matrix Shape (X): {X_all.shape}")
print(f"Total Labels Vector Shape (y): {y_all.shape}")