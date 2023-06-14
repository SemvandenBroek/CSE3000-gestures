#include <Arduino.h>
#include "float.h"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "leds.hpp"
#include "utils.hpp"
#include "constants.h"
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

// Buffers to store sensor data
uint16_t buffer[AMOUNT_PDS][SAMPLE_SIZE];
uint16_t dummy_buffer[AMOUNT_PDS][SAMPLE_SIZE];
float normalized_buffer[AMOUNT_PDS][SAMPLE_SIZE];
float reshaped_buffer[RESHAPE_X][RESHAPE_Y];

// Values for the detection of input
uint16_t activation_calc_tick = ACTIVATION_CALC_TICK_TRIGGER - SAMPLE_SIZE;
uint16_t activation_threshold = UINT16_MAX;

// Tick counters to control the state of the program
int16_t inference_primed = -1;

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
}  // namespace

// Buffer the current photoDiode values
void bufferPhotoDiodes(uint16_t* dest, const size_t buf_size) {
  const unsigned long start = micros();
  dest[buf_size - 3] = analogRead(PD_ONE);
  dest[buf_size - 2] = analogRead(PD_TWO);
  dest[buf_size - 1] = analogRead(PD_THREE);

  // Debugging purposes:
  ((uint16_t*)dummy_buffer)[buf_size - 3] = 1;
  ((uint16_t*)dummy_buffer)[buf_size - 2] = 2;
  ((uint16_t*)dummy_buffer)[buf_size - 1] = 3;

  const unsigned long diff = micros() - start + 4;  // Add offset to compensate if statement
  if (diff < SAMPLE_RATE_DELAY_MICROS) {
    delayMicroseconds(SAMPLE_RATE_DELAY_MICROS - diff);
  }
}

void setupModel() {
  const tflite::Model* model = tflite::GetModel(lstm_model_quantized_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  printf("Model is loaded, version: %d\n", model->version());

  static LstmOpResolver resolver;
  if (RegisterOps(resolver) != kTfLiteOk) {
    printf("Something went wrong while registering operations.\n");
    return;
  }

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    printf("Allocate tensors failed.\n");
    printf("Initialization status is %d\n", interpreter->initialization_status());
    return;
  }

  printf("Initializing model finished!\n");
  printf("Initialization status is %d\n", interpreter->initialization_status());
  printf("MicroInterpreter location: %p\n", interpreter);
}

void invokeModel() {
  printf("Running inference!");
  TfLiteTensor* input = interpreter->input(0);
  TfLiteTensor* output = interpreter->output(0);

  // for (uint16_t i = 0; i < AMOUNT_PDS * SAMPLE_SIZE; i++) {
  //   Serial.print(((uint16_t*) buffer)[i]);
  //   Serial.print(", ");
  // }
  // Serial.println();

  // Normalize values in the buffer per photodiode from 0-1023 to 0.00-1.00 (necessary?)
  printf("Normalizing buffer...");
  mlutils::normalizeBuffer((float*)normalized_buffer, (uint16_t*)buffer, AMOUNT_PDS * SAMPLE_SIZE);

  // We need to reshape the buffer from (100, 3) to (20, 15)
  // printf("Reshaping buffer...");
  // reshapeBuffer((float*) reshaped_buffer, (float*) normalized_buffer, AMOUNT_PDS, SAMPLE_SIZE, RESHAPE_X, RESHAPE_Y);

  // printf("Copying reshaped buffer to tensor...");
  // memcpy(input->data.f, &reshaped_buffer, RESHAPE_Y * RESHAPE_X * sizeof(float));

  // floatArrayToIntArray((float*) normalized_buffer, (int8_t*) int_buffer, AMOUNT_PDS * SAMPLE_SIZE);

  printf("Copying normalized buffer to tensor...");
  memcpy(input->data.f, &test_buffer, AMOUNT_PDS * SAMPLE_SIZE * sizeof(float));

  for (uint16_t i = 0; i < AMOUNT_PDS * SAMPLE_SIZE; i++) {
    mlutils::serialprintf("%0.2f", ((float*)input->data.f)[i]);
    mlutils::serialprintf(", ");
  }
  mlutils::serialprintf("\n");

  mlutils::printInterpreterDetails(interpreter);
  printf("Resetting TFLite interpreter\n");
  interpreter->Reset();
  printf("Invoking TFLite interpreter\n");

  const unsigned long start = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  const unsigned long end = micros();

  mlutils::serialprintf("Model invocation took: %.1f ms\n", (float)(end - start) / 1000);

  if (invoke_status != kTfLiteOk) {
    printf("Invoke failed");
    return;
  }

  int res = mlutils::getMaxIndex(output->data.f, output->dims->data[1]);
  printf("Final model output: %d", res);

  // After invocation recalc the ambient light 50 ticks later
  activation_calc_tick = ACTIVATION_CALC_TICK_TRIGGER - SAMPLE_SIZE / 2;
  setLedOff();
}

void mainLoop() {
  mlutils::shiftWindow((uint16_t*)buffer, AMOUNT_PDS, SAMPLE_SIZE);
  bufferPhotoDiodes((uint16_t*)buffer, AMOUNT_PDS * SAMPLE_SIZE);

  // for (uint16_t i = 0; i < AMOUNT_PDS * SAMPLE_SIZE; i++) {
  //   Serial.print(((uint16_t*) buffer)[i]);
  //   Serial.print(", ");
  // }
  // Serial.println();

  if (inference_primed == -1 && mlutils::calculateBelowThreshold((uint16_t*)buffer, AMOUNT_PDS * SAMPLE_SIZE, 5, 10, activation_threshold)) {
    printf("Priming the model to run in %d ticks...", INFERENCE_PRIME_TICKS);
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
    activation_threshold = mlutils::getAverage((uint16_t*)buffer, AMOUNT_PDS, SAMPLE_SIZE);
    activation_calc_tick = 0;
    mlutils::serialprintf("Calculating current ambient average: %d\n", activation_threshold);
    setLedOff();
  }
}

void setup() {
  Serial.begin(9600);

  setupLeds();

  // Allow the serial monitor to attach
  delay(1000);

#ifdef TF_LITE_STATIC_MEMORY
  printf("TFLite Static Memory is enabled");
#else
  printf("TFLite Static Memory is disabled");
#endif

  regulator = new LightIntensityRegulator();

  tflite::InitializeTarget();
  setupModel();

  setLedBlue();
  delay(500);
  setLedRed();

  invokeModel();
  activation_threshold = mlutils::getAverage((uint16_t*)buffer, AMOUNT_PDS, SAMPLE_SIZE);
}

void loop() {
  mainLoop();
  delay(10);
}