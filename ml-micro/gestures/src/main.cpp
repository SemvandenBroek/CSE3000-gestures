#include <Arduino.h>
#include "leds.hpp"
#include "float.h"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"

#include "utils.h"
#include "diode_calibration.hpp"
#include "lstm_model.h"
#include "lstm_model_quantized.h"
#include "sine_model_quantized.h"

namespace {
  // const tflite::Model* model = nullptr;
  // tflite::MicroInterpreter* interpreter;
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

void callback(const char* s) {
  // serialprintf(s);
  Serial.print(s);
}

void printInterpreterDetails(tflite::MicroInterpreter interpreter) {
  // Obtain a pointer to the model's input tensor
  MicroPrintf("Interpreter input size: %d", interpreter.inputs_size());
  MicroPrintf("Interpreter output size: %d", interpreter.outputs_size());

  MicroPrintf("Interpreter arena_used_bytes: %d", interpreter.arena_used_bytes());
  MicroPrintf("Interpreter initialization_status: %d\n", interpreter.initialization_status());

  // MicroPrintf("Interpreter input name: %s", interpreter.input(0)->name);
  MicroPrintf("Interpreter input allocation_type: %d", interpreter.input(0)->allocation_type);
  MicroPrintf("Interpreter input bytes: %d", interpreter.input(0)->bytes);
  MicroPrintf("Interpreter input type: %d", interpreter.input(0)->type);

  // MicroPrintf("Interpreter output name: %s", interpreter.output(0)->name);
  MicroPrintf("Interpreter output allocation_type: %d", interpreter.output(0)->allocation_type);
  MicroPrintf("Interpreter output bytes: %d", interpreter.output(0)->bytes);
  MicroPrintf("Interpreter output type: %s", TfLiteTypeGetName(interpreter.output(0)->type));

  MicroPrintf("We got input->dims->size: %d", interpreter.input(0)->dims->size);
  for (uint16_t i = 0; i < interpreter.input(0)->dims->size; i++) {
    MicroPrintf("input->dims->data[%d]: %d", i, interpreter.input(0)->dims->data[i]);
  }

  MicroPrintf("Input type is: %s\n", TfLiteTypeGetName(interpreter.input(0)->type));

  MicroPrintf("We got output->dims->size: %d", interpreter.output(0)->dims->size);
  MicroPrintf("output->dims->data[0]: %d", interpreter.output(0)->dims->data[0]);
  MicroPrintf("output->dims->data[1]: %d", interpreter.output(0)->dims->data[1]);
  MicroPrintf("Output type is: %s", TfLiteTypeGetName(interpreter.output(0)->type));
}

// TfLiteStatus loadModel(const tflite::Model* model, tflite::MicroInterpreter* interpreter) {
//   #ifdef TF_LITE_STATIC_MEMORY
//     MicroPrintf("TFLite Static Memory is enabled");
//   #else
//     MicroPrintf("TFLite Static Memory is disabled");
//   #endif

//   if (model->version() != TFLITE_SCHEMA_VERSION) {
//     MicroPrintf("Model provided is schema version %d not equal "
//         "to supported version %d.\n",
//         model->version(), TFLITE_SCHEMA_VERSION);
//   }

//   MicroPrintf("Model is loaded, version: %d\n", model->version());

//   LstmOpResolver resolver;
//   if (RegisterOps(resolver) != kTfLiteOk) {
//     MicroPrintf("Something went wrong while registering operations.\n");
//     return kTfLiteError;
//   }

//   constexpr int kTensorArenaSize = 10 * 1024;
//   uint8_t tensor_arena[kTensorArenaSize];

//   interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize);

//   // if (interpreter.AllocateTensors() != kTfLiteOk) {
//   //   MicroPrintf("Allocate tensors failed.\n");
//   //   MicroPrintf("Initialization status is %d\n", interpreter.initialization_status());
//   //   return kTfLiteError;
//   // }

//   TfLiteTensor* input = interpreter.input(0);

//   // Make sure the input has the properties we expect
//   if (input == nullptr) {
//     MicroPrintf("Input tensor is null.\n");
//     return kTfLiteError;
//   }

//   // Obtain a pointer to the output tensor.
//   TfLiteTensor* output = interpreter.output(0);

//   // Make sure the output has the properties we expect
//   if (output == nullptr) {
//     MicroPrintf("Output tensor is null.\n");
//     return kTfLiteError;
//   }

//   printInterpreterDetails(interpreter);

//   return kTfLiteOk;
// }

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

// Normalizes a 2D array
void normalizeBuffer(float* dest, uint16_t* source, const size_t length_a, const size_t length_b) {
  uint16_t min, max;

  // We calculate the min/max of the entire sample
  // uint16_t concat_buffer[length_a * length_b];
  // memcpy(source, )
  getMinMax(&min, &max, source, length_a * length_b);
  serialprintf("Min: %d Max: %d\n", min, max);

  for (uint16_t i=0; i < length_a; i++) {
    // Doing minmax here calculates the min/max per photodiode
    // getMinMax(&min, &max, &source[i * length_b], length_b);
    // serialprintf("Min: %d Max: %d\n", min, max);

    for (uint16_t j=0; j < length_b; j++) {
      // source[i][j] == source[i * length_b + j]

      //serialprintf("Source: %d\n", source[i * length_b + j]);

      float std = (float) (source[i * length_b + j] - min) / (float) (max - min);
      //serialprintf("Std: %f\n", std);
      
      dest[i * length_b + j] = std;
    }
  }
}

// Normalizes a 1D array
void normalizeBufferOneD(float* dest, uint16_t* source, const size_t buf_size) {
  uint16_t min, max;
  getMinMax(&min, &max, source, buf_size);
  for (uint16_t i=0; i < buf_size; i++) {
    float std = (float) (source[i] - min) / (float) (max - min);
    dest[i] = std;
  }
}

// Rolling buffer window that shifts the buffer in pairs of 100
void shiftWindowOld(uint16_t* buff, const size_t length_a, const size_t window_size) {
  for (uint16_t i = 0; i < length_a; i++) {
    for (uint16_t j = 0; j < window_size; j++) {
      buff[window_size * i + j] = buff[window_size * i + j + 1];
    }
  }
}

// Rolling buffer window that shifts the buffer in pairs of [window_size]
void shiftWindowNew(uint16_t* buff, const size_t window_size, const size_t sample_size) {
  for (uint16_t i = 0; i < (window_size * sample_size - window_size); i++) {
    buff[i] = buff[i + window_size];
  }
}

// Calculate if the last samples of a 2 dimensional array have been below a threshold value
bool calculateBelowThreshold(uint16_t* buff, const size_t length_a, const size_t length_b, const size_t window, const uint16_t gate, const uint16_t dynamic_threshold) {
  for (uint16_t i = 0; i < length_a; i++) {
    for (uint16_t j = length_b - window; j < length_b; j++) {
      if (buff[i * length_b + j] > (dynamic_threshold - gate)) {
        // serialprintf("Not below threshold!\n");
        return false;
      }
    }
  }

  serialprintf("Below threshold!\n");
  return true;
}

// Calculates if the last [window] values were at least [gate] amount below the [dynamic_threshold]
bool calculateBelowThresholdOneD(uint16_t* buff, const size_t buf_size, const size_t window, const uint16_t gate, const uint16_t dynamic_threshold) {
  for (uint16_t i = buf_size - window; i < buf_size; i++) {
    if (buff[i] > (dynamic_threshold - gate)) {
      return false;
    }
  }

  return true;
}

// Buffer the current photoDiode values
void bufferPhotoDiodesNew(uint16_t* dest, const size_t buf_size) {
  const unsigned long start = micros();
  dest[buf_size - 3] = analogRead(A0);
  dest[buf_size - 2] = analogRead(A1);
  dest[buf_size - 1] = analogRead(A2);


  // Debugging purposes:
  ((uint16_t*) dummy_buffer)[buf_size - 3] = 1;
  ((uint16_t*) dummy_buffer)[buf_size - 2] = 2;
  ((uint16_t*) dummy_buffer)[buf_size - 1] = 3;

  const unsigned long diff = micros() - start + 4; // Add offset to compensate if statement
  if (diff < SAMPLE_RATE_DELAY_MICROS) {
      delayMicroseconds(SAMPLE_RATE_DELAY_MICROS - diff);
  }
}

void bufferPhotoDiodesOld(uint16_t* dest, const size_t sample_size) {
  const unsigned long start = micros();
  dest[1 * sample_size - 1] = analogRead(A0);
  dest[2 * sample_size - 1] = analogRead(A1);
  dest[3 * sample_size - 1] = analogRead(A2);

  const unsigned long diff = micros() - start + 4; // Add offset to compensate if statement
  if (diff < SAMPLE_RATE_DELAY_MICROS) {
      delayMicroseconds(SAMPLE_RATE_DELAY_MICROS - diff);
  }
}

// Find the maximum of a one dimensional buffer
template <typename T> int getMax(T* buff, const size_t length) {
  T maxValue = -1;
  int maxIndex = -1;
  for (uint16_t i = 0; i < length; i++) {
    // MicroPrintf("%.4f", buff[i]);
    serialprintf("%.4f\n", buff[i]);
    if (buff[i] > maxValue) {
      maxValue = buff[i];
      maxIndex = i;
    }
  }

  return maxIndex;
}


void mainLoop(tflite::MicroInterpreter interpreter) {
  // shiftWindowOld((uint16_t*) buffer, AMOUNT_PDS, SAMPLE_SIZE);
  shiftWindowNew((uint16_t*) buffer, AMOUNT_PDS, SAMPLE_SIZE);
  bufferPhotoDiodesNew((uint16_t*) buffer, AMOUNT_PDS * SAMPLE_SIZE);
  // bufferPhotoDiodesOld((uint16_t*) buffer, SAMPLE_SIZE);

  // for (uint16_t i = 0; i < AMOUNT_PDS * SAMPLE_SIZE; i++) {
  //   Serial.print(((uint16_t*) buffer)[i]);
  //   Serial.print(", ");
  // }
  // Serial.println();
    

  if (inference_primed == -1 && calculateBelowThresholdOneD((uint16_t*) buffer, AMOUNT_PDS * SAMPLE_SIZE, 5, 10, activation_threshold)) {
    MicroPrintf("Priming the model to run in %d ticks...", INFERENCE_PRIME_TICKS);
    setLedGreen();
    inference_primed = INFERENCE_PRIME_TICKS;
  }

  if (inference_primed >= 0) {
    inference_primed--;
  }

  if (inference_primed == 0) {
    MicroPrintf("Running inference!");
    TfLiteTensor* input = interpreter.input(0);

    // for (uint16_t i = 0; i < AMOUNT_PDS * SAMPLE_SIZE; i++) {
    //   Serial.print(((uint16_t*) buffer)[i]);
    //   Serial.print(", ");
    // }
    // Serial.println();
    
    // Normalize values in the buffer per photodiode from 0-1023 to 0.00-1.00 (necessary?)
    MicroPrintf("Normalizing buffer...");
    normalizeBufferOneD((float*) normalized_buffer, (uint16_t*) buffer, AMOUNT_PDS * SAMPLE_SIZE);
  
    // We need to reshape the buffer from (100, 3) to (20, 15)
    //MicroPrintf("Reshaping buffer...");
    //reshapeBuffer((float*) reshaped_buffer, (float*) normalized_buffer, AMOUNT_PDS, SAMPLE_SIZE, RESHAPE_X, RESHAPE_Y);

    // MicroPrintf("Copying reshaped buffer to tensor...");
    // memcpy(input->data.f, &reshaped_buffer, RESHAPE_Y * RESHAPE_X * sizeof(float));

    MicroPrintf("Copying normalized buffer to tensor...");
    memcpy(input->data.f, &test_buffer, AMOUNT_PDS * SAMPLE_SIZE * sizeof(float));

    for (uint16_t i = 0; i < AMOUNT_PDS * SAMPLE_SIZE; i++) {
      Serial.print(((float*) input->data.f)[i]);
      Serial.print(", ");
    }
    Serial.println();

    // for (uint16_t i = 0; i < AMOUNT_PDS; i++) {
    //   //Serial.println(buffer[i][SAMPLE_SIZE - 1]);
    //   for (uint16_t j = 0; j < SAMPLE_SIZE; j++) {
    //     Serial.print(((float*) input->data.f)[i * SAMPLE_SIZE + j]);
    //     Serial.print(", ");
    //   }
    //   Serial.println();
    // }

    // MicroPrintf("==================== Printing reshaped data ========================");
    // for (uint16_t i=0; i < RESHAPE_X; i++) {
    //   for (uint16_t j=0; j < RESHAPE_Y; j++) {
    //     Serial.print(((float*) reshaped_buffer)[i * RESHAPE_Y + j]);
    //     Serial.print(", ");
    //   }
    //   Serial.println();
    // }
    // MicroPrintf("====================================================================");


    // MicroPrintf("==================== Printing reshaped data from input tensor ========================");
    // for (uint16_t i=0; i < RESHAPE_X; i++) {
    //   for (uint16_t j=0; j < RESHAPE_Y; j++) {
    //     Serial.print(((float*) input->data.f)[i * RESHAPE_Y + j]);
    //     Serial.print(", ");
    //   }
    //   Serial.println();
    // }
    // MicroPrintf("====================================================================");
    

    printInterpreterDetails(interpreter);
    MicroPrintf("Calling Invoke()!");
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Invoke failed");
      return;
    }

    TfLiteTensor* output = interpreter.output(0);
    int res = getMax(output->data.f, output->dims->data[1]);
    MicroPrintf("Final model output: %d", res);

    // After invocation recalc the ambient light 50 ticks later
    activation_calc_tick = ACTIVATION_CALC_TICK_TRIGGER - SAMPLE_SIZE / 2;
    setLedOff();
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
  RegisterDebugLogCallback(callback);
  delay(1000);

  tflite::InitializeTarget();
  regulator = new LightIntensityRegulator();

  setLedBlue();
  delay(500);
  setLedRed();
  
  activation_threshold = getAverage((uint16_t*) buffer, AMOUNT_PDS, SAMPLE_SIZE);
}

void loop() {
  #ifdef TF_LITE_STATIC_MEMORY
    MicroPrintf("TFLite Static Memory is enabled");
  #else
    MicroPrintf("TFLite Static Memory is disabled");
  #endif

  const tflite::Model* model = tflite::GetModel(lstm_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  MicroPrintf("Model is loaded, version: %d\n", model->version());

  LstmOpResolver resolver;
  if (RegisterOps(resolver) != kTfLiteOk) {
    MicroPrintf("Something went wrong while registering operations.\n");
    return;
  }

  constexpr int kTensorArenaSize = 10 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("Allocate tensors failed.\n");
    MicroPrintf("Initialization status is %d\n", interpreter.initialization_status());
    return;
  }
  
  MicroPrintf("Initializing model finished!\n");
  setLedGreen();
  delay(1000);
  setLedOff();
  while (true) {
    mainLoop(interpreter);
  }
}