#include <Arduino.h>

void serialprintf(const char *fmt, ...) {
  char serial_buffer[512];
  va_list va;
  va_start(va, fmt);
  vsprintf(serial_buffer, fmt, va);
  Serial.print(serial_buffer);
}