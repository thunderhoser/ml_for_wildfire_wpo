"""Custom loss functions."""

import os
import sys
import numpy
import tensorflow
from tensorflow.math import lgamma as log_gamma
from tensorflow.keras import backend as K
from tensorflow.keras import ops as tf_ops

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking

MIN_EVIDENCE = 1e-12  # Prevents division by zero.


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


def _get_evidence(input_tensor):
    """Converts input values to evidence.

    :param input_tensor: Input tensor, containing raw values.
    :return: output_tensor: Output tensor -- with same shape as input tensor,
        but containing proper evidence values.
    """

    return K.maximum(tf_ops.softplus(input_tensor), MIN_EVIDENCE)


def __evidential_nll_get_error_tensor(target_tensor, prediction_tensor):
    """Returns error tensor, with evidential NLL for every atomic data sample.

    NLL = negative log likelihood
    "Atomic data sample" = one time step, one grid point, one variable

    :param target_tensor: See doc for `mean_squared_error`.
    :param prediction_tensor: Same.
    :return: error_tensor: Tensor of NLL values with the same shape as both
        input tensors.
    """

    converted_pred_tensor = convert_evidential_outputs(prediction_tensor)

    mu_tensor = converted_pred_tensor[..., 0]
    v_tensor = converted_pred_tensor[..., 1]
    alpha_tensor = converted_pred_tensor[..., 2]
    beta_tensor = converted_pred_tensor[..., 3]
    omega_tensor = 2 * beta_tensor * (1 + v_tensor)

    first_term_tensor = 0.5 * K.log(numpy.pi / v_tensor)
    second_term_tensor = -alpha_tensor * K.log(omega_tensor)
    third_term_tensor = (alpha_tensor + 0.5) * K.log(
        v_tensor * (target_tensor[..., :-1] - mu_tensor) ** 2
        + omega_tensor
    )
    fourth_term_tensor = log_gamma(alpha_tensor) - log_gamma(alpha_tensor + 0.5)

    return (
        first_term_tensor + second_term_tensor +
        third_term_tensor + fourth_term_tensor
    )


def _evidential_negative_log_likelihood(target_tensor, prediction_tensor):
    """Computes negative-log-likelihood part of loss function for evidential NN.

    :param target_tensor: See doc for `mean_squared_error`.
    :param prediction_tensor: Same.
    :return: evidential_nll: Scalar.
    """

    error_tensor = __evidential_nll_get_error_tensor(
        target_tensor=target_tensor,
        prediction_tensor=prediction_tensor
    )

    weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)
    return (
        K.sum(weight_tensor * error_tensor) /
        K.sum(weight_tensor * K.ones_like(error_tensor))
    )


def _dual_weighted_evidential_nll(target_tensor, prediction_tensor,
                                  channel_weights, max_dual_weight_by_channel):
    """Computes dual-weighted version of NLL for evidential NN.

    NLL = negative log likelihood

    :param target_tensor: See doc for `dual_weighted_mse_constrained_dsr`.
    :param prediction_tensor: Same.
    :param channel_weights: Same.
    :param max_dual_weight_by_channel: Same.
    """

    # Create dual-weight tensor.  Inputs to K.maximum below should both have
    # dimensions of E x M x N x T.
    dual_weight_tensor = K.maximum(
        K.abs(target_tensor[..., :-1]),
        K.abs(prediction_tensor[..., 0])
    )

    max_dual_weight_tensor = K.cast(
        K.constant(max_dual_weight_by_channel), dual_weight_tensor.dtype
    )
    for _ in range(3):
        max_dual_weight_tensor = K.expand_dims(
            max_dual_weight_tensor, axis=0
        )

    dual_weight_tensor = K.minimum(dual_weight_tensor, max_dual_weight_tensor)

    # Create channel-weight tensor.
    channel_weight_tensor = K.cast(
        K.constant(channel_weights), dual_weight_tensor.dtype
    )
    for _ in range(3):
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

    # Do the rest.
    error_tensor = __evidential_nll_get_error_tensor(
        target_tensor=target_tensor,
        prediction_tensor=prediction_tensor
    )
    error_tensor = error_tensor * dual_weight_tensor * channel_weight_tensor

    mask_weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)
    return (
        K.sum(mask_weight_tensor * error_tensor) /
        K.sum(mask_weight_tensor * K.ones_like(error_tensor))
    )


def _dual_weighted_evidential_reg_term(
        target_tensor, prediction_tensor, channel_weights,
        max_dual_weight_by_channel):
    """Computes regularization term in loss function for evidential NN.

    :param target_tensor: See doc for `dual_weighted_mse_constrained_dsr`.
    :param prediction_tensor: Same.
    :param channel_weights: Same.
    :param max_dual_weight_by_channel: Same.
    """

    # Create dual-weight tensor.  Inputs to K.maximum below should both have
    # dimensions of E x M x N x T.
    dual_weight_tensor = K.maximum(
        K.abs(target_tensor[..., :-1]),
        K.abs(prediction_tensor[..., 0])
    )

    max_dual_weight_tensor = K.cast(
        K.constant(max_dual_weight_by_channel), dual_weight_tensor.dtype
    )
    for _ in range(3):
        max_dual_weight_tensor = K.expand_dims(
            max_dual_weight_tensor, axis=0
        )

    dual_weight_tensor = K.minimum(dual_weight_tensor, max_dual_weight_tensor)

    # Create channel-weight tensor.
    channel_weight_tensor = K.cast(
        K.constant(channel_weights), dual_weight_tensor.dtype
    )
    for _ in range(3):
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

    # Compute unweighted error tensor.
    converted_pred_tensor = convert_evidential_outputs(prediction_tensor)
    mu_tensor = converted_pred_tensor[..., 0]
    v_tensor = converted_pred_tensor[..., 1]
    alpha_tensor = converted_pred_tensor[..., 2]

    error_tensor = (
        K.abs(target_tensor[..., :-1] - mu_tensor) *
        (2 * v_tensor + alpha_tensor)
    )

    # Finalize.
    error_tensor = error_tensor * dual_weight_tensor * channel_weight_tensor
    mask_weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)
    return (
        K.sum(mask_weight_tensor * error_tensor) /
        K.sum(mask_weight_tensor * K.ones_like(error_tensor))
    )


def convert_evidential_outputs(prediction_tensor):
    """Converts evidential outputs to the desired parameters.

    For definitions of the four parameters -- mu, v, alpha, beta --
    see Schreck et al. (2023) here: https://arxiv.org/abs/2309.13207

    :param prediction_tensor: Tensor of evidential outputs, with dimensions
        E x M x N x T x 4.  prediction_tensor[..., 0] contains values of mu,
        the mean prediction; prediction_tensor[..., 1] contains values of ln(v);
        prediction_tensor[..., 2] contains values of ln(alpha);
        and prediction_tensor[..., 3] contains values of ln(beta).
    :return: prediction_tensor: Same as input, except that
        prediction_tensor[..., 1] contains values of v rather than ln(v);
        prediction_tensor[..., 2] contains alpha rather than ln(alpha);
        and prediction_tensor[..., 3] contains beta rather than ln(beta).
    """

    mu_tensor = prediction_tensor[..., 0]
    v_tensor = _get_evidence(prediction_tensor[..., 1])
    alpha_tensor = _get_evidence(prediction_tensor[..., 2]) + 1.
    beta_tensor = _get_evidence(prediction_tensor[..., 3])

    return K.stack([mu_tensor, v_tensor, alpha_tensor, beta_tensor], axis=-1)


def mean_squared_error(function_name, expect_ensemble=True,
                       is_nn_evidential=False, test_mode=False):
    """Creates mean squared error (MSE) loss function.

    :param function_name: Function name (string).
    :param expect_ensemble: Boolean flag.  If True, will expect
        prediction_tensor to have dimensions E x M x N x T x S.  If False, will
        expect prediction_tensor to have dimensions E x M x N x T.
    :param is_nn_evidential: Boolean flag.  If True, will expect evidential NN,
        where prediction_tensor has dimensions E x M x N x T x 4.
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

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
            Otherwise, if is_nn_evidential == True, will expect dimensions
            E x M x N x T x 4.
            Otherwise, will expect E x M x N x T.
        :return: loss: Mean squared error.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if is_nn_evidential:
            relevant_target_tensor = target_tensor[..., :-1]
            relevant_prediction_tensor = prediction_tensor[..., 0]
            weight_tensor = target_tensor[..., -1]
        elif expect_ensemble:
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
        expect_ensemble=True, is_nn_evidential=False, test_mode=False):
    """Creates dual-weighted mean squared error (DWMSE) loss function.

    K = number of output channels (target variables)

    :param channel_weights: length-K numpy array of channel weights.
    :param function_name: See doc for `mean_squared_error`.
    :param max_dual_weight_by_channel: length-K numpy array of maximum dual
        weights.
    :param expect_ensemble: Same.
    :param is_nn_evidential: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

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

        if is_nn_evidential:
            relevant_target_tensor = target_tensor[..., :-1]
            relevant_prediction_tensor = prediction_tensor[..., 0]
            mask_weight_tensor = target_tensor[..., -1]
        elif expect_ensemble:
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
        max_dual_weight_by_channel=None,
        expect_ensemble=True, is_nn_evidential=False, test_mode=False):
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
    :param is_nn_evidential: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_integer(fwi_index)
    error_checking.assert_is_geq(fwi_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

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

        if is_nn_evidential:
            predicted_dsr_tensor = 0.0272 * K.pow(
                prediction_tensor[..., fwi_index, :], 1.77
            )
            prediction_tensor = K.concatenate([
                prediction_tensor,
                K.expand_dims(predicted_dsr_tensor, axis=-2)
            ], axis=-2)

            relevant_target_tensor = target_tensor[..., :-1]
            relevant_prediction_tensor = prediction_tensor[..., 0]
            mask_weight_tensor = target_tensor[..., -1]
        elif expect_ensemble:
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
        expect_ensemble=True, is_nn_evidential=False, test_mode=False):
    """Creates DWMSE loss function for one channel (target variable).

    :param channel_weight: Channel weight.
    :param channel_index: Channel index.
    :param function_name: See doc for `mean_squared_error`.
    :param max_dual_weight: Max dual weight.
    :param expect_ensemble: Same.
    :param is_nn_evidential: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_greater(channel_weight, 0.)
    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_greater(max_dual_weight, 0.)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (one-channel DWMSE).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: One-channel DWMSE.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if is_nn_evidential:
            relevant_target_tensor = target_tensor[..., channel_index]
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index, 0]
            )
            mask_weight_tensor = target_tensor[..., -1]
        elif expect_ensemble:
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

        relevant_prediction_tensor = tensorflow.transpose(
            relevant_prediction_tensor, perm=[1, 0, 2, 3, 4]
        )
        censored_relevant_prediction_tensor = K.minimum(
            K.abs(relevant_prediction_tensor), max_dual_weight_tensor
        )

        output_type = tensorflow.TensorSpec(
            shape=relevant_prediction_tensor.shape[1:-1],
            dtype=relevant_prediction_tensor.dtype
        )

        # TODO(thunderhoser): In a fresh Colab notebook (albeit one with
        # Keras 2), map_fn works as expected.  It generates the intermediate
        # tensors (where the last two axes have size S x S, S being the ensemble
        # size) individually for each slice along the first axis, i.e., for each
        # grid row.  After generating the intermediate tensors for one grid row,
        # map_fn throws out the intermediate tensors, thus conserving memory.
        # But when I run the code on Hera in Keras 3, map_fn generates the full
        # intermediate tensor at once -- with dimensions M x E x N x T x S x S
        # -- and crashes the memory.  I don't know what's causing this to happen
        # -- maybe Keras 3, maybe something else in the environment, maybe
        # something weird on Hera?  Anyways, this code works on Hera as long as
        # I keep the ensemble size down to ~25.
        mean_prediction_diff_tensor = tensorflow.map_fn(
            fn=lambda p: K.mean(
                K.maximum(
                    K.abs(K.expand_dims(p[1], axis=-1)),
                    K.abs(K.expand_dims(p[1], axis=-2))
                ) *
                K.abs(
                    K.expand_dims(p[0], axis=-1) -
                    K.expand_dims(p[0], axis=-2)
                ),
                axis=(-2, -1)
            ),
            elems=(relevant_prediction_tensor, censored_relevant_prediction_tensor),
            parallel_iterations=1,
            swap_memory=True,
            fn_output_signature=output_type
        )

        mean_prediction_diff_tensor = tensorflow.transpose(
            mean_prediction_diff_tensor, perm=[1, 0, 2, 3]
        )
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


def dual_weighted_evidential_loss(
        channel_weights, regularization_weight, function_name,
        max_dual_weight_by_channel=None, test_mode=False):
    """Creates dual-weighted evidential loss function.

    :param channel_weights: See doc for `dual_weighted_mse_constrained_dsr`.
    :param regularization_weight: Scalar weight for the regularization part of
        the loss function.  This is lambda in Equation 20 of Schreck et al.
    :param function_name: Same.
    :param max_dual_weight_by_channel: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_geq(regularization_weight, 0.)
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
        """Computes loss (dual-weighted evidential loss).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: Dual-weighted evidential loss.
        """

        first_term = _dual_weighted_evidential_nll(
            target_tensor=target_tensor,
            prediction_tensor=prediction_tensor,
            channel_weights=channel_weights,
            max_dual_weight_by_channel=max_dual_weight_by_channel
        )

        second_term = _dual_weighted_evidential_reg_term(
            target_tensor=target_tensor,
            prediction_tensor=prediction_tensor,
            channel_weights=channel_weights,
            max_dual_weight_by_channel=max_dual_weight_by_channel
        )

        return first_term + regularization_weight * second_term

    loss.__name__ = function_name
    return loss
