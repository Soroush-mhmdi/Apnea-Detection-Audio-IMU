import sounddevice as sd
from scipy.io.wavfile import write
import time

fs = 44100  # Sample rate

def record_phase(duration, phase_name, filename):
    print(f"\n--- Get ready for: {phase_name} ---")
    print(f"This will take {duration} seconds.")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
        
    print(">>> RECORDING NOW! <<<")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, myrecording)
    print(f"Done! Saved successfully as {filename}")
    time.sleep(2)

print("Welcome to the Apnea Audio Data Collector")
print("Make sure your laptop is about 30-50 cm away from you.")
input("Press Enter when you are in position and ready to start...")

# Phase 1: Normal Breathing
record_phase(40, "Normal Breathing (Breathe naturally, quietly)", "normal_breath.wav")

# Phase 2: Deep Breathing
record_phase(40, "Deep Breathing (Take slow, deep breaths)", "deep_breath.wav")

# Phase 3: Snoring Simulation
record_phase(40, "Snoring Simulation (Simulate snoring sounds)", "snoring_sim.wav")

# Phase 4: Apnea Simulation
record_phase(40, "Apnea Simulation (Hold your breath completely)", "apnea_sim.wav")

print("\nAll 4 phases recorded successfully!")
print("You now have the complete audio dataset ready for feature extraction.")