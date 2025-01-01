import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Constants
RATE = 16000  # Sample rate of the audio files
NUM_MFCC = 13  # Number of MFCC coefficients to extract
COMMANDS = ["hey selo", "aç", "kapat"]
SAMPLES_PER_COMMAND = 10

# Function to compute MFCCs
def compute_mfcc(audio, sample_rate, num_mfcc=13):
    # Compute the Short-Time Fourier Transform (STFT)
    stft = tf.signal.stft(audio, frame_length=256, frame_step=128, fft_length=256)
    spectrogram = tf.abs(stft)

    # Warp the linear-scale spectrogram to the mel-scale
    num_spectrogram_bins = stft.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz
    )
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute the log-mel spectrogram
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # Compute MFCCs using the Discrete Cosine Transform (DCT)
    mfccs = tf.signal.dct(log_mel_spectrogram, type=2, norm='ortho')[:, :num_mfcc]
    return mfccs

# Function to load and preprocess audio files
def load_and_preprocess_data(directory):
    X = []  # Features (MFCCs)
    y = []  # Labels

    for label_idx, command in enumerate(COMMANDS):
        for i in range(1, SAMPLES_PER_COMMAND + 1):
            filename = os.path.join(directory, f"{command}_{i}.wav")

            # Load the audio file
            audio_binary = tf.io.read_file(filename)
            audio, sample_rate = tf.audio.decode_wav(audio_binary)
            audio = tf.squeeze(audio, axis=-1)  # Convert to mono if necessary

            # Ensure the audio is at the correct sample rate
            if sample_rate.numpy() != RATE:
                raise ValueError(f"Sample rate of {filename} is not {RATE}")

            # Compute MFCC features
            mfcc_features = compute_mfcc(audio, sample_rate, num_mfcc=NUM_MFCC)
            mfcc_features = tf.reduce_mean(mfcc_features, axis=0)  # Average over time

            X.append(mfcc_features.numpy())
            y.append(label_idx)

    return np.array(X), np.array(y)

# Load and preprocess the data
X, y = load_and_preprocess_data("recordings")

# Manually split the data into training and testing sets
def manual_train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.2)

# Define the model
model = models.Sequential([
    layers.Input(shape=(NUM_MFCC,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(len(COMMANDS), activation='softmax')  # Output layer for 3 commands
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save the model
model.save("speech_command_model.h5")

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open("model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

# Function to convert binary file to C-style array
def binary_to_c_array(binary_file, output_file, array_name="model_data"):
    with open(binary_file, "rb") as f:
        data = f.read()
    with open(output_file, "w") as f:
        f.write(f"const unsigned char {array_name}[] = {{\n")
        for i, byte in enumerate(data):
            f.write(f"0x{byte:02x},")
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write("\n};\n")
        f.write(f"const unsigned int {array_name}_len = {len(data)};\n")

# Convert the .tflite model to a .h file
binary_to_c_array("model_quantized.tflite", "model.h")