# ML-Dev folder
This is the main folder in which neural network training with TensorFlow has been performed.

# How to use
- Initialize a Python virtualenv and make sure that `../requirements.txt` is installed with `pip`. 
- Run the `help` command `python run_analysis.py -h` to get more information about input parameters for the different files

# Files
`run_analysis.py` contains code that trains multiple sets of models with the parameters found in `constants.py`, 
this might take a while and output results are stored in the `../results/` folder as `.pickle` files. This is useful for
analyzing K-Fold cross-validation accuracy or to render Confusion matrices.

`analyze_results.ipynb` is used to analyze the results from `run_analysis.py`.

`create_tflite_models.py` is able to train models with the full available dataset (so no train/test split) for use in
production environments. Using the `-r` flag retrains all models and subsequently converts them to `.tflite` and `.cc`
files. (Warning, a Unix-like terminal is required for the tool `xxd` in order to convert the `.tflite` files to byte
arrays)

`embedded_analyzer.ipynb` is used to compare outputs of the microcontroller to outputs that the Python variant of TFLite
produces in order to find discrepancies, by pasting in a fixed sample returned from the microcontroller.
Also used to test the difference between quantized and non-quantized tflite models.

`fft_test.ipynb` was a short analysis ran with multiple samples in order to see if an fft filter would provide benefits for
filtering out a flickering lighting environment.

`sandbox.ipynb` was my main experimental area in which most of all code was firstly tried out before migrating to separate
python files.

