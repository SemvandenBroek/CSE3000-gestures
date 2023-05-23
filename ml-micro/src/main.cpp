#include <Arduino.h>
#include "leds.hpp"

void setup() {
  setupLeds();
}

void loop() {
  // put your main code here, to run repeatedly:
  while (true) {
    setLedBlue();
    delay(1000);
    setLedWhite();
    delay(1000);
  }
}