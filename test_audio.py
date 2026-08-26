import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

# Recording configuration
fs = 44100  # Sampling rate
seconds = 5 # Recording duration in seconds

print("Please breathe into the microphone for 5 seconds...")

# Start recording from the default microphone
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16)
sd.wait()  # Wait until the recording is finished

print("Recording finished!")

# Save the recorded audio as a WAV file
write('test_breath.wav', fs, myrecording)
print("File successfully saved as 'test_breath.wav'.")