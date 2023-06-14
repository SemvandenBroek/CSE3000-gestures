#include <Arduino.h>
#include "constants.h"
#include "diode_calibration.hpp"
#include "float.h"
#include "leds.hpp"
#include "lstm_model.h"
#include "lstm_model_quantized.h"
#include "sine_model_quantized.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "utils.h"

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
}  // namespace

uint16_t buffer[AMOUNT_PDS][SAMPLE_SIZE];
uint16_t dummy_buffer[AMOUNT_PDS][SAMPLE_SIZE];
float normalized_buffer[AMOUNT_PDS][SAMPLE_SIZE];
float reshaped_buffer[RESHAPE_X][RESHAPE_Y];

const float test_buffer[SAMPLE_SIZE * AMOUNT_PDS] = {
    0.91, 0.98, 0.78, 0.92, 0.98, 0.77, 0.91, 0.99, 0.77, 0.92, 0.99, 0.77, 0.91, 0.98, 0.77, 0.92, 0.98, 0.77, 0.91, 1.00, 0.77, 0.91, 0.98, 0.77, 0.91, 0.99, 0.78, 0.93,
    0.99, 0.78, 0.91, 0.98, 0.78, 0.92, 0.99, 0.77, 0.93, 0.98, 0.77, 0.93, 0.99, 0.77, 0.92, 1.00, 0.77, 0.92, 0.98, 0.77, 0.91, 0.98, 0.77, 0.90, 1.00, 0.75, 0.91, 0.99,
    0.77, 0.92, 0.99, 0.77, 0.91, 0.99, 0.77, 0.92, 0.99, 0.78, 0.91, 0.99, 0.77, 0.91, 0.99, 0.77, 0.92, 0.99, 0.77, 0.92, 0.99, 0.77, 0.91, 0.99, 0.78, 0.90, 0.99, 0.77,
    0.90, 0.98, 0.77, 0.90, 0.99, 0.77, 0.90, 0.98, 0.77, 0.89, 0.98, 0.77, 0.88, 0.98, 0.76, 0.86, 0.98, 0.76, 0.85, 0.96, 0.76, 0.83, 0.96, 0.75, 0.81, 0.96, 0.75, 0.81,
    0.95, 0.73, 0.78, 0.94, 0.73, 0.74, 0.94, 0.73, 0.72, 0.92, 0.72, 0.69, 0.91, 0.71, 0.64, 0.89, 0.69, 0.56, 0.88, 0.66, 0.47, 0.84, 0.65, 0.36, 0.83, 0.62, 0.30, 0.77,
    0.59, 0.23, 0.72, 0.56, 0.19, 0.64, 0.52, 0.15, 0.53, 0.40, 0.14, 0.43, 0.33, 0.09, 0.33, 0.27, 0.07, 0.27, 0.23, 0.04, 0.22, 0.20, 0.05, 0.19, 0.17, 0.04, 0.16, 0.15,
    0.02, 0.14, 0.12, 0.02, 0.12, 0.11, 0.02, 0.12, 0.08, 0.01, 0.06, 0.06, 0.05, 0.07, 0.06, 0.17, 0.05, 0.06, 0.27, 0.03, 0.05, 0.41, 0.02, 0.09, 0.52, 0.03, 0.18, 0.62,
    0.02, 0.27, 0.68, 0.02, 0.35, 0.73, 0.00, 0.44, 0.77, 0.02, 0.51, 0.78, 0.14, 0.56, 0.81, 0.26, 0.60, 0.81, 0.38, 0.65, 0.82, 0.46, 0.67, 0.83, 0.54, 0.69, 0.85, 0.59,
    0.67, 0.88, 0.65, 0.72, 0.85, 0.69, 0.72, 0.86, 0.73, 0.72, 0.85, 0.74, 0.73, 0.85, 0.77, 0.73, 0.90, 0.80, 0.74, 0.86, 0.81, 0.73, 0.86, 0.84, 0.74, 0.88, 0.86, 0.75,
    0.89, 0.89, 0.74, 0.89, 0.91, 0.74, 0.89, 0.90, 0.75, 0.89, 0.95, 0.75, 0.88, 0.94, 0.75, 0.92, 0.94, 0.77, 0.90, 0.93, 0.73, 0.88, 0.94, 0.74, 0.90, 0.94, 0.75, 0.90,
    0.95, 0.75, 0.90, 0.90, 0.75, 0.90, 0.95, 0.75, 0.84, 0.94, 0.77, 0.83, 0.95, 0.76, 0.90, 0.95, 0.75, 0.90, 0.95, 0.73};

// Values for the detection of input
uint16_t activation_calc_tick = ACTIVATION_CALC_TICK_TRIGGER - SAMPLE_SIZE;
uint16_t activation_threshold = UINT16_MAX;

// Tick to control the state of the program
int16_t inference_primed = -1;


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

// Find the maximum of a one dimensional buffer
template <typename T>
int getMaxIndex(T* buff, const size_t length) {
    T maxValue = -1;
    int maxIndex = -1;
    for (uint16_t i = 0; i < length; i++) {
        mlutils::serialprintf("%.4f\n", buff[i]);
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
        int_array[i] = (int8_t)(float_array[i] * 255);
    }
}

void setupModel() {
    const tflite::Model* model = tflite::GetModel(lstm_model_quantized_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf(
            "Model provided is schema version %d not equal "
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
    mlutils::normalizeBuffer((float*)normalized_buffer, (uint16_t*)buffer, AMOUNT_PDS * SAMPLE_SIZE);

    // We need to reshape the buffer from (100, 3) to (20, 15)
    // MicroPrintf("Reshaping buffer...");
    // reshapeBuffer((float*) reshaped_buffer, (float*) normalized_buffer, AMOUNT_PDS, SAMPLE_SIZE, RESHAPE_X, RESHAPE_Y);

    // MicroPrintf("Copying reshaped buffer to tensor...");
    // memcpy(input->data.f, &reshaped_buffer, RESHAPE_Y * RESHAPE_X * sizeof(float));

    // floatArrayToIntArray((float*) normalized_buffer, (int8_t*) int_buffer, AMOUNT_PDS * SAMPLE_SIZE);

    MicroPrintf("Copying normalized buffer to tensor...");
    memcpy(input->data.f, &test_buffer, AMOUNT_PDS * SAMPLE_SIZE * sizeof(float));

    for (uint16_t i = 0; i < AMOUNT_PDS * SAMPLE_SIZE; i++) {
        printf("%0.2f", ((float*)input->data.f)[i]);
        printf(", ");
    }
    printf("\n");

    mlutils::printInterpreterDetails(interpreter);
    printf("Resetting TFLite interpreter\n");
    interpreter->Reset();
    printf("Invoking TFLite interpreter\n");

    const unsigned long start = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    const unsigned long end = micros();

    mlutils::serialprintf("Model invocation took: %.1f ms\n", (float)(end - start) / 1000);

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
    mlutils::shiftWindow((uint16_t*)buffer, AMOUNT_PDS, SAMPLE_SIZE);
    bufferPhotoDiodes((uint16_t*)buffer, AMOUNT_PDS * SAMPLE_SIZE);

    for (uint16_t i = 0; i < AMOUNT_PDS * SAMPLE_SIZE; i++) {
        Serial.print(((uint16_t*)buffer)[i]);
        Serial.print(", ");
    }
    Serial.println();

    if (inference_primed == -1 && mlutils::calculateBelowThreshold((uint16_t*)buffer, AMOUNT_PDS * SAMPLE_SIZE, 5, 10, activation_threshold)) {
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
        activation_threshold = mlutils::getAverage((uint16_t*)buffer, AMOUNT_PDS, SAMPLE_SIZE);
        activation_calc_tick = 0;
        mlutils::serialprintf("Calculating current ambient average: %d\n", activation_threshold);
        setLedOff();
    }
}

void readPhotoDiodes() {
    uint16_t r0 = (uint16_t)analogRead(A0);
    uint16_t r1 = (uint16_t)analogRead(A1);
    uint16_t r2 = (uint16_t)analogRead(A2);

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
    activation_threshold = mlutils::getAverage((uint16_t*)buffer, AMOUNT_PDS, SAMPLE_SIZE);
}

void loop() {
    mainLoop();
    delay(100);
}