#include "utils.hpp"

#if defined(ARDUINO)
#include <Arduino.h>
#endif

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace mlutils {
void serialprintf(const char *fmt, ...) {
    char serial_buffer[512];
    va_list va;
    va_start(va, fmt);
    vsprintf(serial_buffer, fmt, va);
    Serial.print(serial_buffer);
}

void getMinMax(uint16_t *min, uint16_t *max, uint16_t *source, size_t length) {
    uint16_t calcMin = UINT16_MAX;
    uint16_t calcMax = 0;

    for (uint16_t i = 0; i < length; i++) {
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
uint16_t getAverage(uint16_t *buff, const size_t length_a, const size_t length_b) {
    uint32_t accumulator = 0;
    for (uint16_t i = 0; i < length_a * length_b; i++) {
        accumulator += buff[i];
    }

    // printf("Accumulated value in getAverage(): %d ")

    return accumulator / (length_a * length_b);
}

// Reshapes a 2D array into the desired shape
template <typename T>
void reshapeBuffer(T *dest, T *source, const size_t length_a, const size_t length_b, const size_t target_reshape_x, const size_t target_reshape_y) {
    uint16_t x = 0;
    uint16_t y = 0;
    for (uint16_t i = 0; i < length_a; i++) {
        for (uint16_t j = 0; j < length_b; j++) {
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
void normalizeBuffer(float *dest, uint16_t *source, const size_t buf_size) {
    uint16_t min, max;
    mlutils::getMinMax(&min, &max, source, buf_size);
    for (uint16_t i = 0; i < buf_size; i++) {
        float std = (float)(source[i] - min) / (float)(max - min);
        dest[i] = std;
    }
}

// Rolling buffer window that shifts the buffer in pairs of [window_size]
void shiftWindow(uint16_t *buff, const size_t window_size, const size_t sample_size) {
    for (uint16_t i = 0; i < (window_size * sample_size - window_size); i++) {
        buff[i] = buff[i + window_size];
    }
}

// Calculates if the last [window] values were at least [gate] amount below the
// [dynamic_threshold]
bool calculateBelowThreshold(uint16_t *buff, const size_t buf_size, const size_t window, const uint16_t gate, const uint16_t dynamic_threshold) {
    for (uint16_t i = buf_size - window; i < buf_size; i++) {
        if (buff[i] > (dynamic_threshold - gate)) {
            return false;
        }
    }

    return true;
}

void printInterpreterDetails(tflite::MicroInterpreter *interpreter) {
    // Obtain a pointer to the model's input tensor
    printf("Interpreter input size: %d", interpreter->inputs_size());
    printf("Interpreter output size: %d", interpreter->outputs_size());

    printf("Interpreter arena_used_bytes: %d", interpreter->arena_used_bytes());
    printf("Interpreter initialization_status: %d\n", interpreter->initialization_status());

    // printf("Interpreter input name: %s", interpreter.input(0)->name);
    printf("Interpreter input allocation_type: %d", interpreter->input(0)->allocation_type);
    printf("Interpreter input bytes: %d", interpreter->input(0)->bytes);
    printf("Interpreter input type: %d", interpreter->input(0)->type);

    // printf("Interpreter output name: %s", interpreter.output(0)->name);
    printf("Interpreter output allocation_type: %d", interpreter->output(0)->allocation_type);
    printf("Interpreter output bytes: %d", interpreter->output(0)->bytes);
    printf("Interpreter output type: %s", TfLiteTypeGetName(interpreter->output(0)->type));

    printf("We got input->dims->size: %d", interpreter->input(0)->dims->size);
    for (uint16_t i = 0; i < interpreter->input(0)->dims->size; i++) {
        printf("input->dims->data[%d]: %d", i, interpreter->input(0)->dims->data[i]);
    }

    printf("Input type is: %s\n", TfLiteTypeGetName(interpreter->input(0)->type));

    printf("We got output->dims->size: %d", interpreter->output(0)->dims->size);
    printf("output->dims->data[0]: %d", interpreter->output(0)->dims->data[0]);
    printf("output->dims->data[1]: %d", interpreter->output(0)->dims->data[1]);
    printf("Output type is: %s", TfLiteTypeGetName(interpreter->output(0)->type));
}

template int mlutils::getMaxIndex(float *buff, const size_t length);

// Find the maximum of a one dimensional buffer
template <typename T>
int getMaxIndex(T *buff, const size_t length) {
    T maxValue = -1;
    int maxIndex = -1;
    for (uint16_t i = 0; i < length; i++) {
        printf("%.4f\n", buff[i]);
        // printf("%d\n", buff[i]);
        if (buff[i] > maxValue) {
            maxValue = buff[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

void floatArrayToIntArray(float *float_array, int8_t *int_array, const size_t length) {
    for (uint16_t i = 0; i < length; i++) {
        int_array[i] = (int8_t)(float_array[i] * 255);
    }
}
}  // namespace mlutils