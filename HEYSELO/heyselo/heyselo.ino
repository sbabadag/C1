#include <TensorFlowLite_ESP32.h>
/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"  // Include the converted model data
#include <Arduino.h>
#include "ESP_I2S.h"


// Audio settings
#define SAMPLE_RATE 16000U
#define SAMPLE_BITS 16
#define I2S_BUFFER_SIZE 1024

I2SClass I2S;  // Create I2S instance

// TensorFlow Lite interpreter
tflite::MicroInterpreter* interpreter = nullptr;

// Input and output tensors
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Touch Switch variables
int threshold = 1500;  // Adjust if not responding properly
bool touch1detected = false;

// Callback function for touch switch
void gotTouch1() {
  touch1detected = true;
}

void setup() {
  // Start Serial Monitor, wait until port is ready
  Serial.begin(115200);
  while (!Serial);

  // Configure I2S with PDM microphone
  I2S.setPinsPdmRx(42, 41);  // CLK, DATA pins
  
  if (!I2S.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("Failed to initialize I2S!");
    while (1);
  }

  // Attach touch switch to interrupt handler
  touchAttachInterrupt(T1, gotTouch1, threshold);

  // Initialize TensorFlow Lite
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    return;
  }

  // Set up the resolver
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();

  // Initialize the interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, nullptr);
  interpreter = &static_interpreter;

  // Allocate tensors
  interpreter->AllocateTensors();

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Setup complete!");
}

void loop() {
  // Check if the touch switch has been pressed
  if (touch1detected) {
    Serial.println("Touch detected, starting wake word detection...");

    // Capture audio data using ESP_I2S
    int16_t audio_buffer[I2S_BUFFER_SIZE];
    size_t bytes_read = I2S.readBytes((char*)audio_buffer, sizeof(audio_buffer));
    
    if (bytes_read == 0) {
      Serial.println("Failed to read I2S data");
      return;
    }

    // Preprocess the audio data (e.g., normalize, convert to spectrogram, etc.)
    for (int i = 0; i < I2S_BUFFER_SIZE; i++) {
      input->data.int8[i] = audio_buffer[i] / 32768.0 * 128;  // Normalize to [-128, 127]
    }

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Invoke failed!");
      return;
    }

    // Get the output
    int8_t* output_data = output->data.int8;

    // Check for the wake word "hey selo"
    if (output_data[0] > 128) {  // Assuming the first output corresponds to "hey selo"
      Serial.println("Wake word detected!");

      // Wait for the command
      // Capture more audio data and run inference again
      bytes_read = I2S.readBytes((char*)audio_buffer, sizeof(audio_buffer));
      if (bytes_read == 0) {
        Serial.println("Failed to read I2S data");
        return;
      }
      for (int i = 0; i < I2S_BUFFER_SIZE; i++) {
        input->data.int8[i] = audio_buffer[i] / 32768.0 * 128;  // Normalize to [-128, 127]
      }
      invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return;
      }

      // Check for the command "aç" or "kapat"
      if (output_data[1] > 128) {  // Assuming the second output corresponds to "aç"
        Serial.println("Command: aç");
      } else if (output_data[2] > 128) {  // Assuming the third output corresponds to "kapat"
        Serial.println("Command: kapat");
      }
    }

    // Reset the touch variable
    touch1detected = false;
  }

  // Add a delay to avoid continuous inference
  delay(100);
}