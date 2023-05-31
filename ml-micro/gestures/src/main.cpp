#include <Arduino.h>
#include "leds.hpp"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"

#include "utils.h"
#include "diode_calibration.hpp"
#include "lstm_model.h"
#include "lstm_model_quantized.h"

namespace {
  const tflite::Model* model = nullptr;
  LightIntensityRegulator* regulator;
  using LstmOpResolver = tflite::MicroMutableOpResolver<6>;

  TfLiteStatus RegisterOps(LstmOpResolver& op_resolver) {
    TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
    TF_LITE_ENSURE_STATUS(op_resolver.AddUnidirectionalSequenceLSTM());
    TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
    TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
    // TF_LITE_ENSURE_STATUS(op_resolver.AddRelu());
    TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
    // TF_LITE_ENSURE_STATUS(op_resolver.AddShape());
    // TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
    // TF_LITE_ENSURE_STATUS(op_resolver.AddPack());
    // TF_LITE_ENSURE_STATUS(op_resolver.AddFill());
    TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());

    return kTfLiteOk;
  }
} 

void callback(const char* s) {
  // serialprintf(s);
  Serial.print(s);
}

TfLiteStatus loadModel() {
  RegisterDebugLogCallback(callback);

  model = ::tflite::GetModel(lstm_model_quantized_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  MicroPrintf("Model is loaded, version: %d\n", model->version());

  delay(1000);

  LstmOpResolver resolver;
  if (RegisterOps(resolver) != kTfLiteOk) {
    MicroPrintf("Something went wrong while registering operations.\n");
    return kTfLiteError;
  }

  constexpr int kTensorArenaSize = 30 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];


  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("Allocate tensors failed.\n");
    MicroPrintf("Initialization status is %d\n", interpreter.initialization_status());
    return kTfLiteError;
  }

  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect
  if (input == nullptr) {
    serialprintf("Input tensor is null.\n");
    return kTfLiteError;
  }

  // Get the input quantization parameters
  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;

  // Obtain a pointer to the output tensor.
  TfLiteTensor* output = interpreter.output(0);

  MicroPrintf("We got input_scale: %f and input_zero_point: %d\n", input_scale, input_zero_point);

  return kTfLiteOk;
}

const int AMOUNT_PDS = 3;
const uint16_t SAMPLE_SIZE = 100;
const uint16_t SAMPLE_RATE = 100;
const uint32_t SAMPLE_RATE_DELAY_MICROS = 1000000 / SAMPLE_RATE;

const size_t RESHAPE_X = 20;
const size_t RESHAPE_Y = 15;

uint16_t buffer[AMOUNT_PDS][SAMPLE_SIZE];
float normalized_buffer[AMOUNT_PDS][SAMPLE_SIZE];
float reshaped_buffer[RESHAPE_X][RESHAPE_Y];


// Values for the detection of input
uint16_t activation_calc_tick = 0;
const uint16_t ACTIVATION_CALC_TICK_TRIGGER = 200;    // Defines how often the controller should recalibrate the ambient light threshold
uint16_t activation_threshold = INT16_MAX;

// Booleans to control the state of the program
int16_t inference_primed = -1;
const uint16_t INFERENCE_PRIME_TICKS = SAMPLE_SIZE / 2;            // Defines how long it takes before we run inference after detecting an initial gesture input

void getMinMax(uint16_t* min, uint16_t* max, uint16_t* source, size_t length) {
  uint16_t calcMin = INT16_MAX;
  uint16_t calcMax = 0;

  for (uint16_t i=0; i < length; i++) {
    if (source[i] < calcMin) {
      calcMin = source[i];
    }

    if (source[i] > calcMax) {
      calcMax = source[i];
    }
  }

  *min = calcMin;
  *max = calcMax;
}

// Calculates the average of a 2D array
uint16_t getAverage(uint16_t* buff, const size_t length_a, const size_t length_b) {
  uint32_t accumulator = 0;
  for (uint16_t i = 0; i < length_a; i++) {
    for (uint16_t j = 0; j < length_b; j++) {
      accumulator += buff[i * length_b + j];
    }
  }

  // serialprintf("Accumulated value in getAverage(): %d ")

  return accumulator / (length_a * length_b);
}

// Reshapes a 2D array into the desired shape
template <typename T> void reshapeBuffer(T* dest, T* source, const size_t length_a, const size_t length_b, const size_t target_reshape_x, const size_t target_reshape_y) {
  uint16_t x = 0;
  uint16_t y = 0;
  for (uint16 i=0; i < length_a; i++) {
    for (uint16 j=0; j < length_b; j++) {
      dest[x * target_reshape_x + y] = source[i * length_b + j];
      x++;
      if (x == target_reshape_x) {
        x = 0;
        y++;
      }
    }
  }
}

// Normalizes a 2D array
void normalizeBuffer(float* dest, uint16_t* source, const size_t length_a, const size_t length_b) {
  uint16_t min, max;

  for (uint16_t i=0; i < length_a; i++) {
    getMinMax(&min, &max, &source[i * length_b], length_b);
    // serialprintf("Min: %d Max: %d\n", min, max);

    for (uint16_t j=0; j < length_b; j++) {
      // source[i][j] == source[i * length_b + j]

      // serialprintf("Source: %d\n", source[i * length_b + j]);

      float std = (float) (source[i * length_b + j] - min) / (float) (max - min);
      // serialprintf("Std: %f\n", std);
      
      dest[i * length_b + j] = std;
    }
  }
}

// Rolling buffer window
void shiftWindow(uint16_t* buff, const size_t length_a, const size_t window_size) {
  for (uint16_t i = 0; i < length_a; i++) {
    for (uint16_t j = 0; j < window_size; j++) {
      buff[window_size * i + j] = buff[window_size * i + j + 1];
    }
  }
}

// Calculate if the last samples of a 2 dimensional array have been below a threshold value
bool calculateBelowThreshold(uint16_t* buff, const size_t length_a, const size_t length_b, const size_t window, const uint16_t threshold) {
  for (uint16_t i = 0; i < length_a; i++) {
    for (uint16_t j = length_b - window; j < length_b; j++) {
      if (buff[i * length_b + j] > threshold) {
        return false;
      }
    }
  }

  return true;
}



// Buffer the current photoDiode values
void bufferPhotoDiodes(uint16_t* dest, const size_t buf_size) {
  const unsigned long start = micros();
  dest[1 * buf_size - 1] = analogRead(A0);
  dest[2 * buf_size - 1] = analogRead(A1);
  dest[3 * buf_size - 1] = analogRead(A2);

  const unsigned long diff = micros() - start + 4; // Add offset to compensate if statement
  if (diff < SAMPLE_RATE_DELAY_MICROS) {
      delayMicroseconds(SAMPLE_RATE_DELAY_MICROS - diff);
  }
}


void mainLoop() {
  shiftWindow((uint16_t*) buffer, AMOUNT_PDS, SAMPLE_SIZE);
  bufferPhotoDiodes((uint16_t*) buffer, SAMPLE_SIZE);
  
  // for (uint16_t i = 0; i < AMOUNT_PDS; i++) {
  //   Serial.println(buffer[i][SAMPLE_SIZE - 1]);
  //   // for (uint16_t j = 0; j < SAMPLE_SIZE; j++) {
  //   //   Serial.print(buffer[i][j]);
  //   //   Serial.print(", ");
  //   // }
  //   // Serial.println();
  // }

  // Normalize values in the buffer per photodiode from 0-1023 to 0.00-1.00 (necessary?)
  //normalizeBuffer((float*) normalized_buffer, (uint16_t*) buffer, AMOUNT_PDS, SAMPLE_SIZE);
  
  // We need to reshape the buffer from (100, 3) to (20, 15)
  //reshapeBuffer((float*) reshaped_buffer, (float*) normalized_buffer, AMOUNT_PDS, SAMPLE_SIZE, RESHAPE_X, RESHAPE_Y);

  // for (int i=0; i < (int) RESHAPE_X; i++) {
  //   for (int j=0; j < (int) RESHAPE_Y; j++) {
  //     Serial.print(reshaped_buffer[i][j]);
  //     Serial.print(", ");
  //   }
  //   Serial.println();
  // }

  if (inference_primed == -1 && calculateBelowThreshold((uint16_t*) buffer, AMOUNT_PDS, SAMPLE_SIZE, 10, activation_threshold)) {
    serialprintf("Priming the model to run in %d ticks...\n", INFERENCE_PRIME_TICKS);
    inference_primed = INFERENCE_PRIME_TICKS;
  }

  if (inference_primed >= 0) {
    inference_primed--;
  }

  if (inference_primed == 0) {
    serialprintf("Running inference!\n");
  }

  activation_calc_tick++;
  if (activation_calc_tick >= ACTIVATION_CALC_TICK_TRIGGER) {
    activation_threshold = getAverage((uint16_t*) buffer, AMOUNT_PDS, SAMPLE_SIZE);
    activation_calc_tick = 0;
    serialprintf("Calculating current ambient average: %d\n", activation_threshold);
  }

  delay(10);
}

void readPhotoDiodes() {
  uint16_t r0 = (uint16_t) analogRead(A0);
  uint16_t r1 = (uint16_t) analogRead(A1);
  uint16_t r2 = (uint16_t) analogRead(A2);

  Serial.print(r0);
  Serial.print(", ");
  Serial.print(r1);
  Serial.print(", ");
  Serial.println(r2);
}

void setup() {
  Serial.begin(9600);

  setupLeds();

  // Allow the serial monitor to attach
  delay(1000);
  tflite::InitializeTarget();
  loadModel();
  MicroPrintf("Initializing model finished!\n");
  regulator = new LightIntensityRegulator();

  setLedBlue();
  delay(500);
  setLedRed();

  activation_threshold = getAverage((uint16_t*) buffer, AMOUNT_PDS, SAMPLE_SIZE);
}

void loop() {
  mainLoop();
}