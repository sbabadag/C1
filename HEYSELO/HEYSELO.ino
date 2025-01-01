#include <TensorFlowLite_ESP32.h>  // Include the TensorFlow Lite ESP32 library
#include "model.h"  // Include the converted model

// Define the pins for the built-in LED
#define LED_BUILTIN 21

// TensorFlow Lite interpreter
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = tflite::GetModel(g_model);
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Use MicroMutableOpResolver to resolve operations
tflite::MicroMutableOpResolver<6> micro_op_resolver;

// Tensor arena size (adjust based on your model's memory requirements)
constexpr int kTensorArenaSize = 10 * 1024;  // 10 KB
uint8_t tensor_arena[kTensorArenaSize];

// Input buffer (replace with your audio input data)
constexpr int kInputSize = 16000;  // Example input size (adjust based on your model)
float input_data[kInputSize];

void setup() {
  // Initialize the LED pin
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  // Set up TensorFlow Lite
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Fill the input buffer with your audio data (replace this with your actual input logic)
  for (int i = 0; i < kInputSize; i++) {
    input_data[i] = 0.0f;  // Replace with actual audio data
  }

  // Copy input data to the TensorFlow Lite input tensor
  for (int i = 0; i < kInputSize; i++) {
    input->data.f[i] = input_data[i];
  }

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