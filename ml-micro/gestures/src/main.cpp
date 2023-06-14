#include <Arduino.h>
#include "leds.hpp"
#include "float.h"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"

#include "utils.h"
#include "diode_calibration.hpp"
#include "lstm_model.h"
#include "lstm_model_quantized.h"
#include "sine_model_quantized.h"

namespace {
  tflite::MicroInterpreter* interpreter;
  constexpr int kTensorArenaSize = 15 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  LightIntensityRegulator* regulator;
  using LstmOpResolver = tflite::MicroMutableOpResolver<6>;

  TfLiteStatus RegisterOps(LstmOpResolver& op_resolver) {
    TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
    TF_LITE_ENSURE_STATUS(op_resolver.AddUnidirectionalSequenceLSTM());
    // TF_LITE_ENSURE_STATUS(op_resolver.AddL2Normalization());
    TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
    TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
    TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
    TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());

    // Below needed for non-quantized version:
    // TF_LITE_ENSURE_STATUS(op_resolver.AddShape());
    // TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
    // TF_LITE_ENSURE_STATUS(op_resolver.AddPack());
    // TF_LITE_ENSURE_STATUS(op_resolver.AddFill());

    return kTfLiteOk;
  }
} 

#if defined(ARDUINO)
  #define PD_ONE A0
  #define PD_TWO A1
  #define PD_THREE A2
#else
  #define PD_ONE 12
  #define PD_TWO 13
  #define PD_THREE 15
#endif

const int AMOUNT_PDS = 3;
const uint16_t SAMPLE_SIZE = 100;
const uint16_t SAMPLE_RATE = 100;
const uint32_t SAMPLE_RATE_DELAY_MICROS = 1000000 / SAMPLE_RATE;

const size_t RESHAPE_X = 20;
const size_t RESHAPE_Y = 15;

uint16_t buffer[AMOUNT_PDS][SAMPLE_SIZE];
uint16_t dummy_buffer[AMOUNT_PDS][SAMPLE_SIZE];
float normalized_buffer[AMOUNT_PDS][SAMPLE_SIZE];
float reshaped_buffer[RESHAPE_X][RESHAPE_Y];

const float test_buffer[SAMPLE_SIZE * AMOUNT_PDS] = {
  0.98, 0.96, 0.83, 0.98, 0.95, 0.83, 0.98, 0.95, 0.82, 0.98, 0.95, 0.83, 0.98, 0.95, 0.83, 0.97, 0.95, 0.83, 0.98, 0.95, 0.83, 0.98, 0.96, 0.82, 0.97, 0.94, 0.83, 0.97, 0.95, 0.83, 0.96, 0.93, 0.83, 0.97, 0.95, 0.83, 0.97, 0.94, 0.83, 0.97, 0.94, 0.83, 0.96, 0.94, 0.83, 0.97, 0.94, 0.82, 0.97, 0.94, 0.83, 0.96, 0.94, 0.82, 0.97, 0.94, 0.83, 0.97, 0.94, 0.83, 0.96, 0.94, 0.83, 0.96, 0.94, 0.83, 0.96, 0.94, 0.83, 0.98, 0.94, 0.82, 0.97, 0.94, 0.82, 0.98, 0.95, 0.83, 0.96, 0.95, 0.83, 0.98, 0.94, 0.82, 0.96, 0.95, 0.82, 0.97, 0.94, 0.83, 0.96, 0.94, 0.82, 0.97, 0.94, 0.83, 0.97, 0.94, 0.83, 0.98, 0.94, 0.82, 0.97, 0.94, 0.81, 0.97, 0.95, 0.82, 0.98, 0.94, 0.82, 0.97, 0.94, 0.82, 0.98, 0.95, 0.82, 0.97, 0.95, 0.81, 1.00, 0.94, 0.82, 0.98, 0.95, 0.82, 0.97, 0.95, 0.81, 0.96, 0.95, 0.81, 0.95, 0.94, 0.80, 0.95, 0.92, 0.80, 0.94, 0.93, 0.76, 0.93, 0.92, 0.73, 0.90, 0.88, 0.63, 0.86, 0.86, 0.52, 0.77, 0.80, 0.40, 0.65, 0.69, 0.21, 0.50, 0.57, 0.10, 0.28, 0.40, 0.04, 0.14, 0.17, 0.01, 0.06, 0.08, 0.00, 0.04, 0.04, 0.06, 0.09, 0.01, 0.20, 0.25, 0.00, 0.40, 0.48, 0.03, 0.57, 0.66, 0.17, 0.67, 0.75, 0.38, 0.71, 0.81, 0.58, 0.74, 0.84, 0.73, 0.76, 0.87, 0.77, 0.75, 0.89, 0.81, 0.78, 0.90, 0.81, 0.77, 0.91, 0.85, 0.78, 0.93, 0.87, 0.80, 0.93, 0.88, 0.79, 0.94, 0.88, 0.80, 0.94, 0.89, 0.80, 0.94, 0.89, 0.81, 0.94, 0.89, 0.80, 0.94, 0.88, 0.80, 0.95, 0.90, 0.81, 0.95, 0.88, 0.80, 0.95, 0.91, 0.80, 0.94, 0.90, 0.81, 0.95, 0.92, 0.81, 0.96, 0.88, 0.81, 0.95, 0.92, 0.81, 0.96, 0.92, 0.81, 0.95, 0.92, 0.81, 0.95, 0.92, 0.81, 0.96, 0.92, 0.81, 0.95, 0.92, 0.80, 0.96, 0.90, 0.81, 0.95, 0.93, 0.81, 0.96, 0.90, 0.81, 0.95, 0.90, 0.81, 0.97, 0.89, 0.81, 0.95, 0.92, 0.81, 0.96, 0.92, 0.81, 0.96, 0.92, 0.81, 0.96, 0.90, 0.81, 0.96, 0.91, 0.80, 0.95, 0.92, 0.82, 0.96, 0.91, 0.82, 0.96, 0.93, 0.81
};


// Values for the detection of input
const uint16_t ACTIVATION_CALC_TICK_TRIGGER = 2000;    // Defines how often the controller should recalibrate the ambient light threshold
uint16_t activation_calc_tick = ACTIVATION_CALC_TICK_TRIGGER - SAMPLE_SIZE;
uint16_t activation_threshold = UINT16_MAX;

// Booleans to control the state of the program
int16_t inference_primed = -1;
const uint16_t INFERENCE_PRIME_TICKS = SAMPLE_SIZE / 2;            // Defines how long it takes before we run inference after detecting an initial gesture input

// void callback(const char* s) {
//   // serialprintf(s);
//   Serial.print(s);
// }

void printInterpreterDetails() {
  // Obtain a pointer to the model's input tensor
  MicroPrintf("Interpreter input size: %d", interpreter->inputs_size());
  MicroPrintf("Interpreter output size: %d", interpreter->outputs_size());

  MicroPrintf("Interpreter arena_used_bytes: %d", interpreter->arena_used_bytes());
  MicroPrintf("Interpreter initialization_status: %d\n", interpreter->initialization_status());

  // MicroPrintf("Interpreter input name: %s", interpreter.input(0)->name);
  MicroPrintf("Interpreter input allocation_type: %d", interpreter->input(0)->allocation_type);
  MicroPrintf("Interpreter input bytes: %d", interpreter->input(0)->bytes);
  MicroPrintf("Interpreter input type: %d", interpreter->input(0)->type);

  // MicroPrintf("Interpreter output name: %s", interpreter.output(0)->name);
  MicroPrintf("Interpreter output allocation_type: %d", interpreter->output(0)->allocation_type);
  MicroPrintf("Interpreter output bytes: %d", interpreter->output(0)->bytes);
  MicroPrintf("Interpreter output type: %s", TfLiteTypeGetName(interpreter->output(0)->type));

  MicroPrintf("We got input->dims->size: %d", interpreter->input(0)->dims->size);
  for (uint16_t i = 0; i < interpreter->input(0)->dims->size; i++) {
    MicroPrintf("input->dims->data[%d]: %d", i, interpreter->input(0)->dims->data[i]);
  }

  MicroPrintf("Input type is: %s\n", TfLiteTypeGetName(interpreter->input(0)->type));

  MicroPrintf("We got output->dims->size: %d", interpreter->output(0)->dims->size);
  MicroPrintf("output->dims->data[0]: %d", interpreter->output(0)->dims->data[0]);
  MicroPrintf("output->dims->data[1]: %d", interpreter->output(0)->dims->data[1]);
  MicroPrintf("Output type is: %s", TfLiteTypeGetName(interpreter->output(0)->type));
}

void getMinMax(uint16_t* min, uint16_t* max, uint16_t* source, size_t length) {
  uint16_t calcMin = UINT16_MAX;
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
  for (uint16_t i = 0; i < length_a * length_b; i++) {
    accumulator += buff[i];
  }

  // serialprintf("Accumulated value in getAverage(): %d ")

  return accumulator / (length_a * length_b);
}

// Reshapes a 2D array into the desired shape
template <typename T> void reshapeBuffer(T* dest, T* source, const size_t length_a, const size_t length_b, const size_t target_reshape_x, const size_t target_reshape_y) {
  uint16_t x = 0;
  uint16_t y = 0;
  for (uint16_t i=0; i < length_a; i++) {
    for (uint16_t j=0; j < length_b; j++) {
      dest[x * target_reshape_x + y] = source[i * length_b + j];
      x++;
      if (x == target_reshape_x) {
        x = 0;
        y++;
      }
    }
  }
}

// Normalizes a 1D array
void normalizeBuffer(float* dest, uint16_t* source, const size_t buf_size) {
  uint16_t min, max;
  getMinMax(&min, &max, source, buf_size);
  for (uint16_t i=0; i < buf_size; i++) {
    float std = (float) (source[i] - min) / (float) (max - min);
    dest[i] = std;
  }
}

// Rolling buffer window that shifts the buffer in pairs of [window_size]
void shiftWindow(uint16_t* buff, const size_t window_size, const size_t sample_size) {
  for (uint16_t i = 0; i < (window_size * sample_size - window_size); i++) {
    buff[i] = buff[i + window_size];
  }
}

// Calculates if the last [window] values were at least [gate] amount below the [dynamic_threshold]
bool calculateBelowThreshold(uint16_t* buff, const size_t buf_size, const size_t window, const uint16_t gate, const uint16_t dynamic_threshold) {
  for (uint16_t i = buf_size - window; i < buf_size; i++) {
    if (buff[i] > (dynamic_threshold - gate)) {
      return false;
    }
  }

  return true;
}

// Buffer the current photoDiode values
void bufferPhotoDiodes(uint16_t* dest, const size_t buf_size) {
  const unsigned long start = micros();
  dest[buf_size - 3] = analogRead(PD_ONE);
  dest[buf_size - 2] = analogRead(PD_TWO);
  dest[buf_size - 1] = analogRead(PD_THREE);


  // Debugging purposes:
  ((uint16_t*) dummy_buffer)[buf_size - 3] = 1;
  ((uint16_t*) dummy_buffer)[buf_size - 2] = 2;
  ((uint16_t*) dummy_buffer)[buf_size - 1] = 3;

  const unsigned long diff = micros() - start + 4; // Add offset to compensate if statement
  if (diff < SAMPLE_RATE_DELAY_MICROS) {
      delayMicroseconds(SAMPLE_RATE_DELAY_MICROS - diff);
  }
}

// Find the maximum of a one dimensional buffer
template <typename T> int getMaxIndex(T* buff, const size_t length) {
  T maxValue = -1;
  int maxIndex = -1;
  for (uint16_t i = 0; i < length; i++) {
    serialprintf("%.4f\n", buff[i]);
    // serialprintf("%d\n", buff[i]);
    if (buff[i] > maxValue) {
      maxValue = buff[i];
      maxIndex = i;
    }
  }

  return maxIndex;
}

void floatArrayToIntArray(float* float_array, int8_t* int_array, const size_t length) {
  for (uint16_t i = 0; i < length; i++) {
    int_array[i] = (int8_t) (float_array[i] * 255);
  }
}

void setupModel() {
  const tflite::Model* model = tflite::GetModel(lstm_model_quantized_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  MicroPrintf("Model is loaded, version: %d\n", model->version());

  static LstmOpResolver resolver;
  if (RegisterOps(resolver) != kTfLiteOk) {
    MicroPrintf("Something went wrong while registering operations.\n");
    return;
  }

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("Allocate tensors failed.\n");
    MicroPrintf("Initialization status is %d\n", interpreter->initialization_status());
    return;
  }
  
  printf("Initializing model finished!\n");
  printf("Initialization status is %d\n", interpreter->initialization_status());
  printf("MicroInterpreter location: %p\n", interpreter);
}

void invokeModel() {
  MicroPrintf("Running inference!");
  TfLiteTensor* input = interpreter->input(0);
  TfLiteTensor* output = interpreter->output(0);

  // for (uint16_t i = 0; i < AMOUNT_PDS * SAMPLE_SIZE; i++) {
  //   Serial.print(((uint16_t*) buffer)[i]);
  //   Serial.print(", ");
  // }
  // Serial.println();
  
  // Normalize values in the buffer per photodiode from 0-1023 to 0.00-1.00 (necessary?)
  MicroPrintf("Normalizing buffer...");
  normalizeBuffer((float*) normalized_buffer, (uint16_t*) buffer, AMOUNT_PDS * SAMPLE_SIZE);

  // We need to reshape the buffer from (100, 3) to (20, 15)
  //MicroPrintf("Reshaping buffer...");
  //reshapeBuffer((float*) reshaped_buffer, (float*) normalized_buffer, AMOUNT_PDS, SAMPLE_SIZE, RESHAPE_X, RESHAPE_Y);

  // MicroPrintf("Copying reshaped buffer to tensor...");
  // memcpy(input->data.f, &reshaped_buffer, RESHAPE_Y * RESHAPE_X * sizeof(float));

  // floatArrayToIntArray((float*) normalized_buffer, (int8_t*) int_buffer, AMOUNT_PDS * SAMPLE_SIZE);

  MicroPrintf("Copying normalized buffer to tensor...");
  memcpy(input->data.f, &test_buffer, AMOUNT_PDS * SAMPLE_SIZE * sizeof(float));

  for (uint16_t i = 0; i < AMOUNT_PDS * SAMPLE_SIZE; i++) {
    printf("%0.2f", ((float*) input->data.f)[i]);
    printf(", ");
  }
  printf("\n");

  printInterpreterDetails();
  printf("Resetting TFLite interpreter\n");
  interpreter->Reset();
  printf("Invoking TFLite interpreter\n");

  const unsigned long start = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  const unsigned long end = micros();

  serialprintf("Model invocation took: %.1f ms\n", (float) (end - start) / 1000);

  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }
  
  int res = getMaxIndex(output->data.f, output->dims->data[1]);
  MicroPrintf("Final model output: %d", res);

  // After invocation recalc the ambient light 50 ticks later
  activation_calc_tick = ACTIVATION_CALC_TICK_TRIGGER - SAMPLE_SIZE / 2;
  setLedOff();
}


void mainLoop() {
  shiftWindow((uint16_t*) buffer, AMOUNT_PDS, SAMPLE_SIZE);
  bufferPhotoDiodes((uint16_t*) buffer, AMOUNT_PDS * SAMPLE_SIZE);

  // for (uint16_t i = 0; i < AMOUNT_PDS * SAMPLE_SIZE; i++) {
  //   Serial.print(((uint16_t*) buffer)[i]);
  //   Serial.print(", ");
  // }
  // Serial.println();
    

  if (inference_primed == -1 && calculateBelowThreshold((uint16_t*) buffer, AMOUNT_PDS * SAMPLE_SIZE, 5, 10, activation_threshold)) {
    MicroPrintf("Priming the model to run in %d ticks...", INFERENCE_PRIME_TICKS);
    setLedGreen();
    inference_primed = INFERENCE_PRIME_TICKS;
  }

  if (inference_primed >= 0) {
    inference_primed--;
  }

  if (inference_primed == 0) {
    invokeModel();
  }

  activation_calc_tick++;
  if (activation_calc_tick >= ACTIVATION_CALC_TICK_TRIGGER) {
    // regulator->recalibrate();
    setLedOrange();
    activation_threshold = getAverage((uint16_t*) buffer, AMOUNT_PDS, SAMPLE_SIZE);
    activation_calc_tick = 0;
    serialprintf("Calculating current ambient average: %d\n", activation_threshold);
    setLedOff();
  }
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
  // RegisterDebugLogCallback(callback);
  delay(1000);

  #ifdef TF_LITE_STATIC_MEMORY
    MicroPrintf("TFLite Static Memory is enabled");
  #else
    MicroPrintf("TFLite Static Memory is disabled");
  #endif

  regulator = new LightIntensityRegulator();
  
  tflite::InitializeTarget();
  setupModel();

  setLedBlue();
  delay(500);
  setLedRed();

  invokeModel();
  activation_threshold = getAverage((uint16_t*) buffer, AMOUNT_PDS, SAMPLE_SIZE);
}

void loop() {
  mainLoop();
  delay(10);
}