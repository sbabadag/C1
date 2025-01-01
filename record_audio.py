import pyaudio
import wave
import os

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 2  # Duration of each recording

# Commands to record
commands = ["hey selo", "aç", "kapat"]
samples_per_command = 10  # Number of samples to collect for each command

# Create a directory to save the recordings
if not os.path.exists("recordings"):
    os.makedirs("recordings")

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Function to record audio
def record_audio(filename):
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print(f"Recording {filename}...")

    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print(f"Finished recording {filename}")

    stream.stop_stream()
    stream.close()

    # Save the recording to a file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Record samples for each command
for command in commands:
    for i in range(samples_per_command):
        filename = f"recordings/{command}_{i+1}.wav"
        record_audio(filename)

# Terminate PyAudio
audio.terminate()