#ifndef ML_MICRO_UTILS_H
#define ML_MICRO_UTILS_H

#include "tensorflow/lite/micro/micro_interpreter.h"

#if defined(ARDUINO)
#include <Arduino.h>
#endif

namespace mlutils {
void serialprintf(const char* fmt, ...);
void getMinMax(uint16_t* min, uint16_t* max, uint16_t* source, size_t length);
uint16_t getAverage(uint16_t* buff, const size_t length_a, const size_t length_b);
template <typename T>
void reshapeBuffer(T* dest, T* source, const size_t length_a, const size_t length_b, const size_t target_reshape_x, const size_t target_reshape_y);
void normalizeBuffer(float* dest, uint16_t* source, const size_t buf_size);
void shiftWindow(uint16_t* buff, const size_t window_size, const size_t sample_size);
bool calculateBelowThreshold(uint16_t* buff, const size_t buf_size, const size_t window, const uint16_t gate, const uint16_t dynamic_threshold);
void printInterpreterDetails(tflite::MicroInterpreter* interpreter);
}  // namespace mlutils

#endif