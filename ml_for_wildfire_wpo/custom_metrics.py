"""Custom metrics."""

import os
import sys
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking


def max_prediction_anywhere(function_name=None, test_mode=False):
    """Creates metric to return max prediction anywhere.

    "Anywhere" = at masked or unmasked grid cell

    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_boolean(test_mode)
    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (max prediction anywhere).

        E = number of examples
        M = number of grid rows
        N = number of grid columns
        S = ensemble size

        :param target_tensor: E-by-M-by-N-by-2 tensor, where
            target_tensor[..., 0] contains the actual target values and
            target_tensor[..., 1] contains weights.
        :param prediction_tensor: E-by-M-by-N-by-S tensor of predicted values.
        :return: metric: Max prediction.
        """

        return K.max(prediction_tensor)

    if function_name is not None:
        metric.__name__ = function_name

    return metric


def max_prediction_unmasked(function_name=None, test_mode=False):
    """Creates metric to return max unmasked prediction.

    "Unmasked" = at grid cell with weight >= 0.05

    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_boolean(test_mode)
    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (max unmasked prediction).

        E = number of examples
        M = number of grid rows
        N = number of grid columns
        S = ensemble size

        :param target_tensor: E-by-M-by-N-by-2 tensor, where
            target_tensor[..., 0] contains the actual target values and
            target_tensor[..., 1] contains weights.
        :param prediction_tensor: E-by-M-by-N-by-S tensor of predicted values.
        :return: metric: Max prediction.
        """

        weight_tensor = target_tensor[..., 1]
        mask_tensor = (weight_tensor >= 0.05).astype(int)
        return K.max(prediction_tensor * K.expand_dims(mask_tensor, axis=-1))

    if function_name is not None:
        metric.__name__ = function_name

    return metric
