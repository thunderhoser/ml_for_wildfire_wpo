"""Custom loss functions."""

import os
import sys
import numpy
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras import ops as K_ops

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


def dual_weighted_mse(
        channel_weights, function_name, max_dual_weight_by_channel=None,
        expect_ensemble=True, test_mode=False):
    """Creates dual-weighted mean squared error (DWMSE) loss function.

    K = number of output channels (target variables)

    :param channel_weights: length-K numpy array of channel weights.
    :param function_name: See doc for `mean_squared_error`.
    :param max_dual_weight_by_channel: length-K numpy array of maximum dual
        weights.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    if max_dual_weight_by_channel is None:
        max_dual_weight_by_channel = numpy.full(len(channel_weights), 1e12)

    error_checking.assert_is_numpy_array(
        channel_weights,
        exact_dimensions=numpy.array([len(channel_weights)], dtype=int)
    )
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)

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

        max_dual_weight_tensor = K.cast(
            K.constant(max_dual_weight_by_channel), dual_weight_tensor.dtype
        )
        for _ in range(3):
            max_dual_weight_tensor = K.expand_dims(
                max_dual_weight_tensor, axis=0
            )
        if expect_ensemble:
            max_dual_weight_tensor = K.expand_dims(
                max_dual_weight_tensor, axis=-1
            )

        dual_weight_tensor = K.minimum(
            dual_weight_tensor, max_dual_weight_tensor
        )

        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        for _ in range(3):
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


def dual_weighted_mse_constrained_dsr(
        channel_weights, fwi_index, function_name,
        max_dual_weight_by_channel=None, expect_ensemble=True, test_mode=False):
    """Creates DWMSE loss function with constrained DSR.

    "Constrained DSR" means that daily severity rating is computed directly from
    fire-weather index (FWI).  This method assumes that the last elements of
    the arrays `channel_weights` and `max_dual_weight_by_channel` pertain to
    DSR.

    K = number of output channels (target variables), not including DSR

    :param channel_weights: length-(K + 1) numpy array of channel weights.
    :param fwi_index: Array index for FWI.  This tells the method that FWI
        predictions and targets can be found in
        target_tensor[:, :, :, fwi_index, ...] and
        prediction_tensor[:, :, :, fwi_index, ...], respectively.
    :param function_name: See doc for `mean_squared_error`.
    :param max_dual_weight_by_channel: length-(K + 1) numpy array of maximum
        dual weights.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_integer(fwi_index)
    error_checking.assert_is_geq(fwi_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    if max_dual_weight_by_channel is None:
        max_dual_weight_by_channel = numpy.full(len(channel_weights), 1e12)

    error_checking.assert_is_numpy_array(
        channel_weights,
        exact_dimensions=numpy.array([len(channel_weights)], dtype=int)
    )
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: Mean squared error.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        target_dsr_tensor = 0.0272 * K.pow(target_tensor[..., fwi_index], 1.77)
        target_tensor = K.concatenate([
            target_tensor[..., :-1],
            K.expand_dims(target_dsr_tensor, axis=-1),
            K.expand_dims(target_tensor[..., -1], axis=-1)
        ], axis=-1)

        if expect_ensemble:
            predicted_dsr_tensor = 0.0272 * K.pow(
                prediction_tensor[..., fwi_index, :], 1.77
            )
            prediction_tensor = K.concatenate([
                prediction_tensor,
                K.expand_dims(predicted_dsr_tensor, axis=-2)
            ], axis=-2)

            relevant_target_tensor = K.expand_dims(
                target_tensor[..., :-1], axis=-1
            )
            relevant_prediction_tensor = prediction_tensor
            mask_weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)
        else:
            predicted_dsr_tensor = 0.0272 * K.pow(
                prediction_tensor[..., fwi_index], 1.77
            )
            prediction_tensor = K.concatenate([
                prediction_tensor,
                K.expand_dims(predicted_dsr_tensor, axis=-1)
            ], axis=-1)

            relevant_target_tensor = target_tensor[..., :-1]
            relevant_prediction_tensor = prediction_tensor
            mask_weight_tensor = target_tensor[..., -1]

        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )

        max_dual_weight_tensor = K.cast(
            K.constant(max_dual_weight_by_channel), dual_weight_tensor.dtype
        )
        for _ in range(3):
            max_dual_weight_tensor = K.expand_dims(
                max_dual_weight_tensor, axis=0
            )
        if expect_ensemble:
            max_dual_weight_tensor = K.expand_dims(
                max_dual_weight_tensor, axis=-1
            )

        dual_weight_tensor = K.minimum(
            dual_weight_tensor, max_dual_weight_tensor
        )

        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        for _ in range(3):
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


def dual_weighted_mse_1channel(
        channel_weight, channel_index, function_name, max_dual_weight=1e12,
        expect_ensemble=True, test_mode=False):
    """Creates DWMSE loss function for one channel (target variable).

    :param channel_weight: Channel weight.
    :param channel_index: Channel index.
    :param function_name: See doc for `mean_squared_error`.
    :param max_dual_weight: Max dual weight.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_greater(channel_weight, 0.)
    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_greater(max_dual_weight, 0.)
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
        dual_weight_tensor = K.minimum(dual_weight_tensor, max_dual_weight)

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


def dual_weighted_crps_constrained_dsr(
        channel_weights, fwi_index, function_name,
        max_dual_weight_by_channel=None, test_mode=False):
    """Creates dual-weighted CRPS loss function with constrained DSR.

    :param channel_weights: See doc for `dual_weighted_mse_constrained_dsr`.
    :param fwi_index: Same.
    :param function_name: Same.
    :param max_dual_weight_by_channel: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_integer(fwi_index)
    error_checking.assert_is_geq(fwi_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    if max_dual_weight_by_channel is None:
        max_dual_weight_by_channel = numpy.full(len(channel_weights), 1e12)

    error_checking.assert_is_numpy_array(
        channel_weights,
        exact_dimensions=numpy.array([len(channel_weights)], dtype=int)
    )
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (dual-weighted CRPS).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: Dual-weighted CRPS.
        """

        # Add DSR to target tensor.
        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        target_dsr_tensor = 0.0272 * K.pow(target_tensor[..., fwi_index], 1.77)
        target_tensor = K.concatenate([
            target_tensor[..., :-1],
            K.expand_dims(target_dsr_tensor, axis=-1),
            K.expand_dims(target_tensor[..., -1], axis=-1)
        ], axis=-1)

        # Add DSR to prediction tensor.
        predicted_dsr_tensor = 0.0272 * K.pow(
            prediction_tensor[..., fwi_index, :], 1.77
        )
        prediction_tensor = K.concatenate([
            prediction_tensor,
            K.expand_dims(predicted_dsr_tensor, axis=-2)
        ], axis=-2)

        # Ensure compatible tensor shapes.
        relevant_target_tensor = K.expand_dims(
            target_tensor[..., :-1], axis=-1
        )
        relevant_prediction_tensor = prediction_tensor
        mask_weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)

        # Create dual-weight tensor.
        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )

        max_dual_weight_tensor = K.cast(
            K.constant(max_dual_weight_by_channel), dual_weight_tensor.dtype
        )
        for _ in range(3):
            max_dual_weight_tensor = K.expand_dims(
                max_dual_weight_tensor, axis=0
            )
        max_dual_weight_tensor = K.expand_dims(max_dual_weight_tensor, axis=-1)

        dual_weight_tensor = K.minimum(
            dual_weight_tensor, max_dual_weight_tensor
        )

        # Create channel-weight tensor.
        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        for _ in range(3):
            channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

        # Compute dual-weighted CRPS.
        absolute_error_tensor = K.abs(
            relevant_prediction_tensor - relevant_target_tensor
        )
        mean_prediction_error_tensor = K.mean(
            dual_weight_tensor * absolute_error_tensor, axis=-1
        )

        relevant_prediction_tensor = K_ops.swapaxes(relevant_prediction_tensor, 0, 1)
        censored_relevant_prediction_tensor = K.minimum(
            relevant_prediction_tensor, max_dual_weight_tensor
        )
        print('relevant_prediction_tensor.shape = {0:s}'.format(
            str(relevant_prediction_tensor.shape)
        ))
        print('censored_relevant_prediction_tensor.shape = {0:s}'.format(
            str(censored_relevant_prediction_tensor.shape)
        ))

        # mean_prediction_diff_tensor = K.map_fn(
        #     fn=lambda p: K.mean(
        #         K.maximum(
        #             K.abs(K.expand_dims(p[1], axis=-1)),
        #             K.abs(K.expand_dims(p[1], axis=-2))
        #         ) *
        #         K.abs(
        #             K.expand_dims(p[0], axis=-1) -
        #             K.expand_dims(p[0], axis=-2)
        #         ),
        #         axis=(-2, -1)
        #     ),
        #     elems=(relevant_prediction_tensor, censored_relevant_prediction_tensor)
        # )

        # mean_prediction_diff_tensor = K.map_fn(
        #     fn=lambda p: K.mean(
        #         K.maximum(
        #             K.abs(K.expand_dims(p, axis=-1)),
        #             K.abs(K.expand_dims(p, axis=-2))
        #         ) *
        #         K.abs(
        #             K.expand_dims(p, axis=-1) -
        #             K.expand_dims(p, axis=-2)
        #         ),
        #         axis=(-2, -1)
        #     ),
        #     elems=relevant_prediction_tensor
        # )

        # mean_prediction_diff_tensor = K.stack([
        #     K.mean(
        #         K.maximum(
        #             K.abs(K.expand_dims(p, axis=-1)),
        #             K.abs(K.expand_dims(p, axis=-2))
        #         ) *
        #         K.abs(
        #             K.expand_dims(p, axis=-1) -
        #             K.expand_dims(p, axis=-2)
        #         ),
        #         axis=(-2, -1)
        #     )
        #     for p in relevant_prediction_tensor
        # ], axis=0)

        mean_prediction_diff_tensor = tensorflow.vectorized_map(
            fn=lambda p: K.mean(
                K.maximum(
                    K.abs(K.expand_dims(p, axis=-1)),
                    K.abs(K.expand_dims(p, axis=-2))
                ) *
                K.abs(
                    K.expand_dims(p, axis=-1) -
                    K.expand_dims(p, axis=-2)
                ),
                axis=(-2, -1)
            ),
            elems=relevant_prediction_tensor
        )

        print('First mean_prediction_diff_tensor.shape = {0:s}'.format(
            str(mean_prediction_diff_tensor.shape)
        ))

        # mean_prediction_diff_tensor = K_ops.swapaxes(mean_prediction_diff_tensor[:, 0, ...], 0, 1)
        mean_prediction_diff_tensor = K_ops.swapaxes(mean_prediction_diff_tensor, 0, 1)
        print('Second mean_prediction_diff_tensor.shape = {0:s}'.format(
            str(mean_prediction_diff_tensor.shape)
        ))

        error_tensor = channel_weight_tensor * (
            mean_prediction_error_tensor -
            0.5 * mean_prediction_diff_tensor
        )

        return (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

    loss.__name__ = function_name
    return loss
