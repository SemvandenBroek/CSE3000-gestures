#ifndef ML_MICRO_CONSTANTS_H
#define ML_MICRO_CONSTANTS_H

#include "float.h"

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

const uint16_t ACTIVATION_CALC_TICK_TRIGGER = 2000;      // Defines how often the controller should recalibrate the ambient light threshold
const uint16_t INFERENCE_PRIME_TICKS = SAMPLE_SIZE / 2;  // Defines how long it takes before we run inference after detecting an initial gesture input

#endif