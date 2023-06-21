import argparse
import os

from utils import load_all_gestures, preprocess_pipeline, Gestures, compile_model, extract_features, split_new, compile_tflite
from models import build_lstm_stateless_model
from constants import EPOCHS_LSTM, LSTM_UNITS


def compile_lstm(units: int, retrain=True):
    model = None

    model_name = f"lstm_{units}"
    if not os.path.exists(os.path.join("../models", model_name)):
        retrain = True

    gestures = load_all_gestures('../data/gestures')
    shape = (100, 3)
    preprocess_pipeline(gestures, reshape_shape=shape)

    (train, _) = split_new(gestures, 1.0, skip_candidates=['default', 'b2'])
    (_, x, y) = extract_features(train, num_classes=len(Gestures), expected_input_shape=shape)

    if retrain:
        model = build_lstm_stateless_model(shape, lstm_units=units, num_classes=len(Gestures))
        compile_model(model)

        model.fit(x, y, batch_size=32, epochs=EPOCHS_LSTM)

    compile_tflite(model, shape=[100, 3], save_dir="../models", name=model_name, representative_dataset=x)


def compile_all_lstm(retrain):
    for u in LSTM_UNITS:
        compile_lstm(u, retrain)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-r", dest='retrain', action="store_true", help="Append -r to retrain the models")
    parsed = arg_parser.parse_args()

    compile_all_lstm(parsed.retrain)