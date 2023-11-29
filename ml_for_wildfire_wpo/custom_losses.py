"""Custom loss functions."""

import os
import sys
import tensorflow
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking


def _log2(input_tensor):
    """Computes logarithm in base 2.

    :param input_tensor: Keras tensor.
    :return: logarithm_tensor: Keras tensor with the same shape as
        `input_tensor`.
    """

    return (
        K.log(K.maximum(input_tensor, 1e-6)) /
        K.log(tensorflow.Variable(2., dtype=tensorflow.float64))
    )


def mean_squared_error(function_name, test_mode=False):
    """Creates mean squared error (MSE) loss function.

    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (mean squared error).

        E = number of examples
        M = number of grid rows
        N = number of grid columns
        T = number of target variables (channels)
        S = ensemble size

        :param target_tensor: E-by-M-by-N-by-(T + 1) tensor, where
            target_tensor[..., :-1] contains the actual target values and
            target_tensor[..., -1] contains weights.
        :param prediction_tensor: E-by-M-by-N-by-T-by-S tensor of predicted
            values.
        :return: loss: Mean squared error.
        """

        # squared_error_tensor = K.mean(
        #     (target_tensor[..., :-1] - prediction_tensor) ** 2, axis=-1
        # )

        squared_error_tensor = (
            target_tensor[..., :-1] -
            K.mean(prediction_tensor, axis=-1)
        ) ** 2

        weight_tensor = target_tensor[..., -1]
        return K.mean(weight_tensor * squared_error_tensor)

    loss.__name__ = function_name
    return loss


def dual_weighted_mse(channel_weights, function_name, test_mode=False):
    """Creates dual-weighted mean squared error (DWMSE) loss function.

    K = number of output channels (target variables)

    :param channel_weights: length-K numpy array of channel weights.
    :param function_name: See doc for `mean_squared_error`.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: Mean squared error.
        """

        # squared_error_tensor = K.mean(
        #     (target_tensor[..., :-1] - prediction_tensor) ** 2, axis=-1
        # )

        ensemble_mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)
        dual_weight_tensor = K.maximum(
            K.abs(target_tensor[..., :-1]),
            K.abs(ensemble_mean_prediction_tensor)
        )

        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

        print(channel_weight_tensor)
        print(dual_weight_tensor)
        print(target_tensor[..., :-1] - ensemble_mean_prediction_tensor)

        error_tensor = (
            channel_weight_tensor * dual_weight_tensor *
            (target_tensor[..., :-1] - ensemble_mean_prediction_tensor) ** 2
        )

        mask_weight_tensor = target_tensor[..., -1]
        return K.mean(mask_weight_tensor * error_tensor)

    loss.__name__ = function_name
    return loss
