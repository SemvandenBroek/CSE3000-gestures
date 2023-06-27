# ML-Micro folder
This folder contains all the code used to deploy a model to the Arduino Nano 33 BLE using Platform.io. (An effort was also made to port this codebase to ESP32 but that code is not published as of now).

The `./gestures/src/` folder contains the most important part and the custom code to run the detection system.

The `./lib/` folder contains some git submodules however they should no longer be needed as the TensorFlow library is now added through `./gestures/platformio.ini`