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


def mean_squared_error(function_name, expect_ensemble=True, test_mode=False):
    """Creates mean squared error (MSE) loss function.

    :param function_name: Function name (string).
    :param expect_ensemble: Boolean flag.  If True, will expect
        prediction_tensor to have dimensions E x M x N x T x S.  If False, will
        expect prediction_tensor to have dimensions E x M x N x T.
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
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
        :param prediction_tensor: Tensor of predicted values.  If
            expect_ensemble == True, will expect dimensions E x M x N x T x S.
            Otherwise, will expect E x M x N x T.
        :return: loss: Mean squared error.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                target_tensor[..., :-1], axis=-1
            )
            relevant_prediction_tensor = prediction_tensor
            weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)
        else:
            relevant_target_tensor = target_tensor[..., :-1]
            relevant_prediction_tensor = prediction_tensor
            weight_tensor = target_tensor[..., -1]

        squared_error_tensor = (
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )
        weight_tensor = K.expand_dims(weight_tensor, axis=-1)

        # return K.mean(weight_tensor * squared_error_tensor)
        return (
            K.sum(weight_tensor * squared_error_tensor) /
            K.sum(weight_tensor * K.ones_like(squared_error_tensor))
        )

    loss.__name__ = function_name
    return loss


def dual_weighted_mse(channel_weights, function_name, expect_ensemble=True,
                      test_mode=False):
    """Creates dual-weighted mean squared error (DWMSE) loss function.

    K = number of output channels (target variables)

    :param channel_weights: length-K numpy array of channel weights.
    :param function_name: See doc for `mean_squared_error`.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: Mean squared error.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                target_tensor[..., :-1], axis=-1
            )
            relevant_prediction_tensor = prediction_tensor
            mask_weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)
        else:
            relevant_target_tensor = target_tensor[..., :-1]
            relevant_prediction_tensor = prediction_tensor
            mask_weight_tensor = target_tensor[..., -1]

        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )

        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

        if expect_ensemble:
            channel_weight_tensor = K.expand_dims(
                channel_weight_tensor, axis=-1
            )

        error_tensor = (
            channel_weight_tensor * dual_weight_tensor *
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )
        mask_weight_tensor = K.expand_dims(mask_weight_tensor, axis=-1)

        # return K.mean(mask_weight_tensor * error_tensor)
        return (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

    loss.__name__ = function_name
    return loss


def dual_weighted_mse_1channel(channel_weight, channel_index, function_name,
                               expect_ensemble=True, test_mode=False):
    """Creates DWMSE loss function for one channel (target variable).

    :param channel_weight: Channel weight.
    :param channel_index: Channel index.
    :param function_name: See doc for `mean_squared_error`.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_greater(channel_weight, 0.)
    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (one-channel DWMSE).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: One-channel DWMSE.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                target_tensor[..., channel_index], axis=-1
            )
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index, :]
            )
            mask_weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)
        else:
            relevant_target_tensor = target_tensor[..., channel_index]
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index]
            )
            mask_weight_tensor = target_tensor[..., -1]

        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )
        error_tensor = (
            channel_weight * dual_weight_tensor *
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )

        # return K.mean(mask_weight_tensor * error_tensor)
        return (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

    loss.__name__ = function_name
    return loss
