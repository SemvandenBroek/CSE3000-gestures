import random

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import logging
from enum import Enum
import itertools
import os
import pickle
import re

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

log = logging.getLogger("CSE3000/loader")
# Utility functions cell
RANDOM_SEED = 1000

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class Fold(Enum):
    FIVE = 5
    TEN = 10


class Hand(Enum):
    right = "right_hand"
    left = "left_hand"


class GestureException(Exception):
    pass


class Gestures(Enum):
    SWIPE_LEFT = 0, 'swipe_left'
    SWIPE_RIGHT = 1, 'swipe_right'
    SWIPE_UP = 2, 'swipe_up'
    SWIPE_DOWN = 3, 'swipe_down'
    ROT_CW = 4, 'clockwise'
    ROT_CCW = 5, 'counter_clockwise'
    TAP = 6, 'tap'
    DOUBLE_TAP = 7, 'double_tap'
    ZOOM_IN = 8, 'zoom_in'
    ZOOM_OUT = 9, 'zoom_out'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value

    def __str__(self):
        return self.fullname

    @staticmethod
    def from_name(name: str):
        try:
            return next(g for g in Gestures if g.fullname == name)
        except StopIteration:
            raise GestureException("No gesture with name '%s' found..." % name)


class LoadGestureException(Exception):
    pass


def load_gesture_samples(gesture: Gestures, hand: Hand = Hand.right, base_path='gestures_data/gestures', skip_old_data: bool = True):
    result = []
    base_path = os.path.join(base_path, gesture.fullname, hand.value)
    log.debug("Loading gestures from base path: %s" % base_path)
    folder_items = os.listdir(base_path)

    # Filter on the .pickle extension
    filtered_ext = list(filter(lambda x: re.search(r'\.pickle$', x) is not None, folder_items))

    if len(filtered_ext) == 0:
        raise LoadGestureException("No gestures found in folder: %s" % base_path)

    for item in filtered_ext:
        r_match = re.match(r'candidate_(\w+).pickle$', item)
        if r_match is None:
            raise LoadGestureException("Incorrectly formatted data file name: %s" % item)

        candidate_id = r_match.group(1)
        with open(os.path.join(base_path, item), 'rb') as f:
            while True:
                try:
                    data_contents = pickle.load(f)

                    if isinstance(data_contents, dict):
                        if 'target_gesture' in data_contents:
                            # Data v3
                            # print(data_contents)
                            data_contents['gesture'] = Gestures.from_name(data_contents['target_gesture'])
                            # data_contents['all_data'] = data_contents['data']
                            # print(type(data_contents['data']))
                            # data_contents['data'] = list(map(lambda x: x['data'], data_contents['data']))
                            result.append(data_contents)
                        else:
                            # Data v2
                            data_contents['gesture'] = Gestures.from_name(data_contents['gesture'])
                            if not skip_old_data:
                                result.append(data_contents)
                    else:
                        # Data loader v1
                        data = {
                            'data': data_contents,
                            'gesture': gesture,
                            'candidate': candidate_id
                        }
                        if not skip_old_data:
                            result.append(data)
                except EOFError:
                    break

    return result


def split_data_between_participants(data, ratio=0.7, expected_input_shape=(100, 3), dtype=np.int16):
    lb_candidate = lambda x: x['candidate']

    # For itertools.groupby to work we need to sort the data first
    data.sort(key=lambda x: x['candidate'])

    participants = set(map(lb_candidate, data))
    amount_measurements = len(data)
    amount_participants = len(participants)

    log.debug("Participants: %s" % participants)
    log.info("Got dataset for %d participants with %d measurements total" % (amount_participants, amount_measurements))

    amount_train = int(amount_measurements * ratio)
    amount_test = amount_measurements - amount_train
    log.info("Estimating %d measurements for training and %d measurements for test (ratio: %0.1f)" % (
        amount_train, amount_test, ratio))

    train_data = []
    train_data_outcomes = []

    test_data = []
    test_data_outcomes = []

    train_candidates = set()
    test_candidates = set()

    # Group the data per participant as that is the recommended method for training models
    for participant, d in itertools.groupby(data, lb_candidate):
        d_list = list(d)

        # if participant == 'A3':
        #     continue

        for data_point in d_list:
            try:
                if len(train_data) < amount_train:
                    assert data_point['data'].shape == expected_input_shape
                    train_candidates.add(participant)
                    train_data.append(data_point['data'])
                    train_data_outcomes.append(data_point['gesture'])
                else:
                    test_candidates.add(participant)
                    test_data.append(data_point['data'])
                    test_data_outcomes.append(data_point['gesture'])
                    # test_data.extend([p['data'] for p in d_list])
                    # test_data_outcomes.extend([p['gesture'].value for p in d_list])
            except AssertionError as e:
                log.error("Could not load gesture %s of participant %s (expected shape: %s but got %s)" % (
                    data_point['gesture'], participant, expected_input_shape, data_point['data'].shape))

    log.info("Train candidates: %s\tTest candidates: %s" % (train_candidates, test_candidates))

    return (np.array(train_data, dtype=dtype), np.array(train_data_outcomes, dtype=dtype)), (
        np.array(test_data, dtype=dtype), np.array(test_data_outcomes, dtype=dtype))


def split_new(dataset, ratio: float = 0.7, skip_candidates: list = None):
    if skip_candidates is None:
        skip_candidates = []

    dataset.sort(key=lambda x: x['candidate'])

    candidate_map = {}
    for c, data in itertools.groupby(dataset, lambda x: x['candidate']):
        if c not in skip_candidates:
            candidate_map.setdefault(c, list(data))

    split_amount = int(len(candidate_map) * ratio)

    train = dict(itertools.islice(candidate_map.items(), split_amount))
    test = dict(itertools.islice(candidate_map.items(), split_amount, len(candidate_map)))

    log.info("Train candidates: %s \t Test candidates: %s" % (list(train.keys()), list(test.keys())))

    return train, test


def extract_features(data: dict, num_classes=len(Gestures), expected_input_shape=(100, 3), dtype=np.float32):
    candidates = []
    x_data = []
    y_data = []

    for candidate, measurements in data.items():
        for measurement in measurements:
            try:
                assert measurement['data'].shape == expected_input_shape
                candidates.append(candidate)
                x_data.append(measurement['data'])
                y_data.append(measurement['gesture'])
            except AssertionError:
                log.error("Could not load gesture %s of participant %s (expected shape: %s but got %s)" % (
                    measurement['gesture'], candidate, expected_input_shape, measurement['data'].shape))

    return candidates, np.array(x_data, dtype=dtype), to_categorical(y_data, num_classes)


def preprocess_pipeline(data: list, expected_input_shape=(100, 3), dtype=np.int16, reshape_shape=(20, 15)):
    for measurement in data:
        try:
            assert measurement['data'].shape == expected_input_shape

            measurement['data'] = normalize_all_photodiode_dataset(measurement['data'])
            # measurement['data'] = measurement['data'].reshape(reshape_shape)
        except AssertionError:
            data.remove(measurement)
            log.error(
                "Could not preprocess gesture %s of participant %s (expected shape: %s but got %s), removed from dataset" % (
                    measurement['gesture'], measurement['candidate'], expected_input_shape, measurement['data'].shape))


def normalize_per_photodiode_dataset(data, dtype=np.float32):
    """Watch out this function might not normalize the data as expected, further research required"""
    normalized = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    for graph in data:
        pd_data = []
        for pd in graph.reshape(graph.shape[1], graph.shape[0]):
            pd_data.append(scaler.fit_transform(pd.reshape(-1, 1)).reshape(pd.shape))
        normalized.append(np.array(pd_data).reshape(graph.shape))
    return np.array(normalized, dtype=dtype)


def normalize_all_photodiode_dataset(data):
    """Watch out this function might not normalize the data as expected, further research required"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    return scaled


def normalize_test(data):
    normalized = []
    for measurement in data:
        mean = measurement.mean()
        std = measurement.std()
        normalized_measurement = (measurement - mean) / std

        normalized.append(normalized_measurement)
    return normalized


def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    mae = history.history['mae']
    val_mae = history.history['val_mae']

    epochs = range(1, len(loss) + 1)

    fig, axs = plt.subplots(1, 3, figsize=(25, 5))

    axs[0].plot(epochs, loss, 'g.', label='Training Loss')
    axs[0].plot(epochs, val_loss, 'c.', label='Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(epochs, mae, 'g.', label='Training MAE')
    axs[1].plot(epochs, val_mae, 'c.', label='Validation MAE')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('MAE')
    axs[1].legend()

    axs[2].plot(epochs, acc, 'g.', label='Training Accuracy')
    axs[2].plot(epochs, val_acc, 'c.', label='Validation Accuracy')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Accuracy')
    axs[2].legend()

    fig.savefig('output_figures/history_plot.svg')
    fig.show()


def compile_model(model: keras.Model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mae', 'categorical_accuracy'])


def kfold_cross_validation(model: keras.Model, full_dataset: dict, num_folds: int = 5, epochs=30, early_stop=False,
                           batch_size=32, expected_input_shape=(20, 15)):
    kfold = KFold(num_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_num = 1
    acc_per_fold = []
    loss_per_fold = []
    confusion_per_fold = []

    # (dataset, _) = split_new(full_dataset, 1)
    full_dataset.sort(key=lambda x: x['candidate'])

    dataset = []
    candidates = []
    for c, data in itertools.groupby(full_dataset, lambda x: x['candidate']):
        candidates.append(c)
        dataset.append(list(data))

    # (candidates, x_data, y_data) = extract_features(dataset, expected_input_shape=(20, 15))
    # print(candidates)
    # print(len(x_data))
    # print(len(candidates))
    # print(x_data.shape)
    # print(y_data.shape)

    # We need to input candidates instead of features
    for train_index, test_index in kfold.split(candidates):
        fold_model = keras.models.clone_model(model)
        log.info("Fold No. %d" % fold_num)
        compile_model(fold_model)

        print(train_index)

        candidate_train_data = []
        candidate_train_labels = []

        candidate_test_data = []
        candidate_test_labels = []

        for i in train_index:
            for measurement in dataset[i]:
                try:
                    assert measurement['data'].shape == expected_input_shape
                    candidate_train_data.append(measurement['data'])
                    candidate_train_labels.append(to_categorical(measurement['gesture'], len(Gestures)))
                except AssertionError:
                    log.error(
                        "Could not perform k-fold on gesture %s of participant %s (expected shape: %s but got %s), removed from dataset" % (
                            measurement['gesture'], measurement['candidate'], expected_input_shape,
                            measurement['data'].shape))

        for i in test_index:
            for measurement in dataset[i]:
                try:
                    assert measurement['data'].shape == expected_input_shape
                    candidate_test_data.append(measurement['data'])
                    candidate_test_labels.append(to_categorical(measurement['gesture'], len(Gestures)))
                except AssertionError:
                    log.error(
                        "Could not perform k-fold on gesture %s of participant %s (expected shape: %s but got %s), removed from dataset" % (
                            measurement['gesture'], measurement['candidate'], expected_input_shape,
                            measurement['data'].shape))

        early_stop_cb = EarlyStopping(
            monitor='loss',
            mode='min',
            patience=10
        )

        candidate_train_data = np.array(candidate_train_data)
        candidate_train_labels = np.array(candidate_train_labels)
        candidate_test_data = np.array(candidate_test_data)
        candidate_test_labels = np.array(candidate_test_labels)

        if early_stop:
            fold_model.fit(candidate_train_data, candidate_train_labels, callbacks=[early_stop_cb],
                           batch_size=batch_size, epochs=epochs, verbose=0)
        else:
            fold_model.fit(candidate_train_data, candidate_train_labels, batch_size=batch_size, epochs=epochs,
                           verbose=0)
        scores = fold_model.evaluate(candidate_test_data, candidate_test_labels, verbose=2)
        predictions = np.argmax(fold_model.predict(candidate_test_data), axis=1)
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        confusion_per_fold.append(confusion_matrix(np.argmax(candidate_test_labels, axis=1), predictions))

        fold_num += 1

    return acc_per_fold, loss_per_fold, confusion_per_fold


def load_all_gestures(base_path, filtered_list=Gestures, shuffle=True):
    combined = []

    right_samples_count = 0
    left_samples_count = 0
    for g in filtered_list:
        right_samples = load_gesture_samples(g, hand=Hand.right, base_path=base_path)
        left_samples = load_gesture_samples(g, hand=Hand.left, base_path=base_path)
        right_samples_count += len(right_samples)
        left_samples_count += len(left_samples)
        combined.extend(right_samples)
        combined.extend(left_samples)

    log.info("Got %d Right hand measurements and %d Left hand measurements" % (right_samples_count, left_samples_count))
    if shuffle:
        random.Random(4).shuffle(combined)

    return combined


def compile_tflite(model: keras.Sequential | None, shape: list, save_dir: str, name: str, representative_dataset):
    save_path = os.path.join(save_dir, name)
    tf_save_path = os.path.join(save_path, "tf")
    tflite_save_path = os.path.join(save_path, f"{name}.tflite")
    tflite_quantized_save_path = os.path.join(save_path, f"{name}_quantized.tflite")

    if model is not None:
        run_model = tf.function(lambda x: model(x))
        # This is important, let's fix the input size.
        batch_size = 1
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec((1, 100, 3), model.inputs[0].dtype))

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        model.save(tf_save_path, save_format="tf", signatures=concrete_func)

    if not os.path.exists(save_dir):
        raise Exception("Could not load model, set retrain=True")

    # Convert to tflite model
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_save_path)
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    open(tflite_save_path, "wb").write(tflite_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True

    def representative_dataset_generator():
        for value in representative_dataset[0:500]:
            value = np.expand_dims(value, axis=0)
            yield [np.array(value, dtype=np.float32, ndmin=3)]

    converter.representative_dataset = representative_dataset_generator
    tflite_model_quantized = converter.convert()

    debugger = tf.lite.experimental.QuantizationDebugger(
        converter=converter, debug_dataset=representative_dataset_generator)
    debugger.run()
    with open(os.path.join(save_path, 'debug.csv'), 'w') as f:
        debugger.layer_statistics_dump(f)

    open(tflite_quantized_save_path, "wb").write(tflite_model_quantized)

    model_size = os.path.getsize(tflite_save_path)
    quantized_size = os.path.getsize(tflite_quantized_save_path)
    print("Normal model size: %d bytes (%.2f Kb)" % (model_size, float(model_size / 1000)))
    print("Quantized model size: %d bytes (%.2f Kb)" % (quantized_size, float(quantized_size / 1000)))
    print("Difference: %d bytes" % (model_size - quantized_size))

    # Temporarily switch path so that xxd uses a nicer identifier in the .cc file
    old_cwd = os.getcwd()
    os.chdir(save_path)
    os.system(f"xxd -i {name}.tflite > {name}.cc")
    os.system(f"xxd -i {name}_quantized.tflite > {name}_quantized.cc")
    os.chdir(old_cwd)
