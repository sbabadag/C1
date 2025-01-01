#include <TensorFlowLite_ESP32.h>  // Include the TensorFlow Lite ESP32 library
#include "model.h"  // Include the converted model
#include <Audio.h>  // Include the audio library for ESP32-S3 Sense

// Define the pins for the built-in LED
#define LED_BUILTIN 21

// TensorFlow Lite interpreter
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = tflite::GetModel(g_model);
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Audio buffer
constexpr int kAudioBufferSize = 16000;  // 1 second of audio at 16kHz
int16_t audio_buffer[kAudioBufferSize];

// Audio setup for ESP32-S3 Sense
Audio audio;

void setup() {
  // Initialize the LED pin
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  // Initialize the audio library
  audio.begin();

  // Set up TensorFlow Lite
  static tflite::MicroInterpreter static_interpreter(
      model, tflite::MicroOpResolver<6>(*error_reporter), error_reporter);
  interpreter = &static_interpreter;

  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Record audio into the buffer
  if (audio.available()) {
    audio.read(audio_buffer, kAudioBufferSize);
  }

  // Preprocess the audio (e.g., convert to MFCCs)
  // preprocessAudio(audio_buffer, input->data.f);

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Get the output
  float* output_data = output->data.f;
  int predicted_class = std::distance(output_data, std::max_element(output_data, output_data + 3));

  // Control the LED based on the predicted class
  if (predicted_class == 1) {  // "aç"
    digitalWrite(LED_BUILTIN, HIGH);
  } else if (predicted_class == 2) {  // "kapat"
    digitalWrite(LED_BUILTIN, LOW);
  }

  // Add a delay to avoid rapid toggling
  delay(1000);
}