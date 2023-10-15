"""Custom loss functions."""

import tensorflow
from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import error_checking


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


def mean_squared_error(function_name=None, test_mode=False):
    """Creates mean squared error (MSE) loss function.

    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_boolean(test_mode)
    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (mean squared error).

        E = number of examples
        M = number of grid rows
        N = number of grid columns
        S = ensemble size

        :param target_tensor: E-by-M-by-N-by-2 tensor, where
            target_tensor[..., 0] contains the actual target values and
            target_tensor[..., 1] contains weights.
        :param prediction_tensor: E-by-M-by-N-by-S tensor of predicted values.
        :return: loss: Mean squared error.
        """

        # squared_error_tensor = K.mean(
        #     (target_tensor[..., [0]] - prediction_tensor) ** 2, axis=-1
        # )
        squared_error_tensor = (
            (target_tensor[..., 0] - K.mean(prediction_tensor, axis=-1)) ** 2
        )

        weight_tensor = target_tensor[..., 1]
        return K.mean(weight_tensor * squared_error_tensor)

    if function_name is not None:
        loss.__name__ = function_name

    return loss


def dual_weighted_mse(function_name=None, test_mode=False):
    """Creates dual-weighted mean squared error (DWMSE) loss function.

    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_boolean(test_mode)
    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE).

        E = number of examples
        M = number of grid rows
        N = number of grid columns
        S = ensemble size

        :param target_tensor: E-by-M-by-N-by-2 tensor, where
            target_tensor[..., 0] contains the actual target values and
            target_tensor[..., 1] contains weights.
        :param prediction_tensor: E-by-M-by-N-by-S tensor of predicted values.
        :return: loss: Mean squared error.
        """

        # squared_error_tensor = K.mean(
        #     (target_tensor[..., [0]] - prediction_tensor) ** 2, axis=-1
        # )

        ensemble_mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)
        dual_weight_tensor = K.maximum(
            K.abs(target_tensor[..., 0]),
            K.abs(ensemble_mean_prediction_tensor)
        )
        error_tensor = (
            dual_weight_tensor *
            (target_tensor[..., 0] - ensemble_mean_prediction_tensor) ** 2
        )

        mask_weight_tensor = target_tensor[..., 1]
        return K.mean(mask_weight_tensor * error_tensor)

    if function_name is not None:
        loss.__name__ = function_name

    return loss


def cross_entropy(function_name=None, test_mode=False):
    """Creates cross-entropy loss function.

    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_boolean(test_mode)
    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (cross-entropy).

        E = number of examples
        M = number of grid rows
        N = number of grid columns
        K = number of classes
        S = ensemble size

        :param target_tensor: E-by-M-by-N-by-(K + 1) tensor, where
            target_tensor[..., -1] contains weights and the first K slices of
            the last axis are a one-hot encoding of class membership.
        :param prediction_tensor: E-by-M-by-N-by-K-by-S tensor of predicted
            class probabilities.
        :return: loss: Cross-entropy.
        """

        # product_tensor = K.mean(
        #     K.expand_dims(target_tensor[..., :-1], axis=-1) *
        #     _log2(prediction_tensor),
        #     axis=-1
        # )
        product_tensor = (
            target_tensor[..., :-1] *
            _log2(K.mean(prediction_tensor, axis=-1))
        )

        weight_tensor = target_tensor[..., -1]
        product_tensor_summed_over_classes = (
            weight_tensor * K.sum(product_tensor, axis=-1)
        )

        return -K.mean(product_tensor_summed_over_classes)

    if function_name is not None:
        loss.__name__ = function_name

    return loss
