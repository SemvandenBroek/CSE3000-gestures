import os
import pickle
import re
from enum import Enum


class Hand(Enum):
    right = "right_hand"
    left = "left_hand"


class GestureNames(Enum):
    swipe_left = "swipe_left"
    swipe_right = "swipe_right"


class Gestures(Enum):
    SWIPE_LEFT = 0
    SWIPE_RIGHT = 1

    @staticmethod
    def from_name(name: str):
        if name == GestureNames.swipe_right.value:
            return Gestures.SWIPE_RIGHT
        elif name == GestureNames.swipe_left.value:
            return Gestures.SWIPE_LEFT
        raise Exception("No matching gesture found for '%s'" % name)


class LoadGestureException(Exception):
    pass


def load_gesture_samples(gesture_name: GestureNames, hand: Hand = Hand.right):
    result = []
    base_path = f"gestures_data/gestures/{gesture_name.value}/{hand.value}"
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
                        result.append(data_contents)
                    else:
                        # Old data loader
                        data = {
                            'data': data_contents,
                            'gesture': gesture_name.value,
                            'candidate': candidate_id
                        }
                        result.append(data)
                except EOFError:
                    break

    return result