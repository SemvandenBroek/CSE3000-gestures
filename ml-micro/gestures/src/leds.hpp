#include "Arduino.h"

// Onboard LED off
void setLedOff() {
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
}

// Onboard LED to red
void setLedRed() {
  digitalWrite(LEDR, LOW);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
}

// Onboard LED to green
void setLedGreen() {
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, LOW);
  digitalWrite(LEDB, HIGH);
}

void setLedBlue() {
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, LOW);
}

void setLedOrange() {
  digitalWrite(LEDR, LOW);
  digitalWrite(LEDG, LOW);
  digitalWrite(LEDB, HIGH);
}

void setLedYellow() {
  digitalWrite(LEDR, LOW);
  digitalWrite(LEDG, LOW);
  digitalWrite(LEDB, HIGH);
}

void setLedPurple() {
  digitalWrite(LEDR, LOW);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, LOW);
}

void setLedWhite() {
  digitalWrite(LEDR, LOW);
  digitalWrite(LEDG, LOW);
  digitalWrite(LEDB, LOW);
}

void setupLeds() {
  //Set the LED pins as outputs
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  setLedOff();
}

enum LedColor {
  RED,
  GREEN,
  BLUE,
  ORANGE,
  YELLOW,
  PURPLE,
  WHITE,
  OFF
};

void setLed(LedColor color) {
  switch (color) {
    case RED:
      setLedRed();
      break;
    case GREEN:
      setLedGreen();
      break;
    case BLUE:
      setLedBlue();
      break;
    case ORANGE:
      setLedOrange();
      break;
    case YELLOW:
      setLedYellow();
      break;
    case PURPLE:
      setLedPurple();
      break;
    case WHITE:
      setLedWhite();
      break;
    case OFF:
      setLedOff();
      break;
  }
}