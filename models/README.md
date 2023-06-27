# Models folder
This folder contains all the analyzed models that are trained on the data. The following file formats can be found:

- `tf/` The `tf/` folder contains the model in a format that can be loaded with `tf.keras.models.load_model(...)`
- `.tflite` contains both original and quantized versions of the model in the `.tflite` format
- `.cc` is a C byte-array of the `.tflite` file to be used to deploy models in a C deployment
- `debug.csv` contains quantization debug values, which are currently not used