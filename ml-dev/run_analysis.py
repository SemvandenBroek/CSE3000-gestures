import argparse
import logging
import os
import pickle

import tensorflow as tf
from utils import load_all_gestures, preprocess_pipeline, split_new, extract_features, Gestures, Fold, \
    kfold_cross_validation
from models import build_conv_lstm_model, build_lstm_stateless_model
from constants import LSTM_UNITS, FOLD_ITERATIONS, EPOCHS_CONV_LSTM, EPOCHS_LSTM

logging.basicConfig(level=logging.INFO,
                    force=True)
log = logging.getLogger("CSE3000/analysis")


def save_dict(dictionary, name):
    base_path = '../results/'
    save_path = os.path.join(base_path, name)

    if not os.path.exists(base_path):
        os.mkdir(base_path)

    with open(save_path, 'wb') as file:
        pickle.dump(dictionary, file)


def analyze_conv_lstm(gpu):
    log.info("Analyzing convolutional LSTM")
    shape = (100, 3)
    gestures = load_all_gestures('../data/gestures')
    preprocess_pipeline(gestures, reshape_shape=shape)

    device = "/device:GPU:0" if gpu else "/device:CPU:0"

    # Init empty dict
    fold_data = {}

    with tf.device(device):
        # Run five-fold and ten-fold
        for f in filter(lambda x: x is not Fold.FIVE, Fold):
            log.info("Running %d-fold..." % f.value)

            for u in LSTM_UNITS:
                key = "ConvLSTM %d" % u
                log.info("Currently running %s for convolutional LSTM" % key)
                model = build_conv_lstm_model(input_shape=shape, num_classes=len(Gestures), lstm_units=u)
                fold_data.setdefault(key, {'iterations': FOLD_ITERATIONS, 'folds': {}})
                fold_data.get(key).get('folds').setdefault(f, [])
                for i in range(FOLD_ITERATIONS):
                    log.info("Iteration %d of %d" % (i + 1, FOLD_ITERATIONS))
                    acc_per_fold, loss_per_fold, confusion_per_fold = kfold_cross_validation(model, gestures,
                                                                                             num_folds=f.value,
                                                                                             epochs=EPOCHS_CONV_LSTM,
                                                                                             early_stop=False,
                                                                                             expected_input_shape=shape)

                    fold_data.get(key).get('folds').get(f).append({
                        'acc_per_fold': acc_per_fold,
                        'loss_per_fold': loss_per_fold,
                        'confusion_per_fold': confusion_per_fold
                    })

                    save_dict(fold_data, 'conv_lstm.pickle')


def analyze_lstm(gpu):
    log.info("Analyzing normal LSTM")
    shape = (100, 3)
    gestures = load_all_gestures('../data/gestures')
    preprocess_pipeline(gestures, reshape_shape=shape)

    device = "/device:GPU:0" if gpu else "/device:CPU:0"

    # Init empty dict
    fold_data = {}

    with tf.device(device):
        # Run five-fold and ten-fold
        for f in Fold:
            log.info("Running %d-fold..." % f.value)

            for u in LSTM_UNITS:
                key = "LSTM %d" % u
                log.info("Currently running %s for normal LSTM" % key)
                model = build_lstm_stateless_model(input_shape=shape, lstm_units=u, dense_1=False, dropout=True, bidirectional=False, num_classes=len(Gestures))
                fold_data.setdefault(key, {'iterations': FOLD_ITERATIONS, 'folds': {}})
                fold_data.get(key).get('folds').setdefault(f, [])
                for i in range(FOLD_ITERATIONS):
                    log.info("Iteration %d of %d" % (i + 1, FOLD_ITERATIONS))
                    acc_per_fold, loss_per_fold, confusion_per_fold = kfold_cross_validation(model, gestures,
                                                                                             num_folds=f.value,
                                                                                             epochs=EPOCHS_LSTM,
                                                                                             early_stop=False,
                                                                                             expected_input_shape=shape)

                    fold_data.get(key).get('folds').get(f).append({
                        'acc_per_fold': acc_per_fold,
                        'loss_per_fold': loss_per_fold,
                        'confusion_per_fold': confusion_per_fold
                    })

                    save_dict(fold_data, 'lstm.pickle')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('model', type=str, choices=['conv_lstm', 'lstm', 'all'])
    arg_parser.add_argument("--gpu", dest='gpu', action="store_true", help="Append --gpu to run with the GPU")
    parsed = arg_parser.parse_args()

    match parsed.model:
        case 'conv_lstm':
            analyze_conv_lstm(parsed.gpu)
        case 'lstm':
            analyze_lstm(parsed.gpu)
        case 'all':
            analyze_lstm(parsed.gpu)
            analyze_conv_lstm(parsed.gpu)
