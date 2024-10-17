"""Custom loss functions."""

import numpy
import tensorflow
import tensorflow.math
from tensorflow.keras import backend as K
# from tensorflow.keras import ops as tf_ops
from gewittergefahr.gg_utils import error_checking


def __get_num_target_fields(prediction_tensor, expect_ensemble):
    """Determines number of target fields.

    :param prediction_tensor: See documentation for `mean_squared_error`.
    :param expect_ensemble: Same.
    :return: num_target_fields: Integer.
    """

    if expect_ensemble:
        return prediction_tensor.shape[-2]

    return prediction_tensor.shape[-1]


def _log2(input_tensor):
    """Computes logarithm in base 2.

    :param input_tensor: Keras tensor.
    :return: logarithm_tensor: Keras tensor with the same shape as
        `input_tensor`.
    """

    return (
        K.log(K.maximum(input_tensor, K.epsilon())) /
        K.log(tensorflow.Variable(2., dtype=tensorflow.float64))
    )


def _natural_log(input_tensor):
    """Computes natural logarithm.

    :param input_tensor: Keras tensor.
    :return: logarithm_tensor: Keras tensor with the same shape as
        `input_tensor`.
    """

    return K.log(K.maximum(input_tensor, K.epsilon()))


def _power(input_tensor, exponent):
    """Computes power.

    :param input_tensor: Keras tensor.
    :param exponent: Scalar exponent (float).
    :return: power_tensor: Keras tensor with the same shape as `input_tensor`.
    """

    power_tensor = K.pow(K.maximum(input_tensor, K.epsilon()), exponent)
    return K.maximum(power_tensor, K.epsilon())


def _exponential(input_tensor):
    """Computes exponential function.

    :param input_tensor: Keras tensor.
    :return: exp_tensor: Keras tensor with the same shape as `input_tensor`.
    """

    return K.maximum(K.exp(input_tensor), K.epsilon())


def _check_index_args(dmc_index, dc_index, isi_index):
    """Error-checks index arguments.

    :param dmc_index: See documentation for `dual_weighted_mse_all_constraints`.
    :param dc_index: Same.
    :param isi_index: Same.
    """

    error_checking.assert_is_integer(dmc_index)
    error_checking.assert_is_integer(dc_index)
    error_checking.assert_is_integer(isi_index)

    all_indices = numpy.array([dmc_index, dc_index, isi_index], dtype=int)
    error_checking.assert_is_geq_numpy_array(all_indices, 0)
    assert len(all_indices) == len(numpy.unique(all_indices))


def _add_bui_to_tensors(prediction_tensor, target_tensor_no_mask,
                        dmc_index, dc_index, expect_ensemble):
    """Adds build-up index (BUI) to each tensor.

    E = number of examples (time steps)
    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields, not including BUI
    S = ensemble size

    :param prediction_tensor: Tensor of predicted fire-weather values.  If
        `expect_ensemble == True`, this must have dimensions E x M x N x T x S;
        otherwise, must have dimensions E x M x N x T.
    :param target_tensor_no_mask: Tensor of actual fire-weather values.  This
        must have dimensions E x M x N x T.
    :param dmc_index: Array index for DMC.  This tells the method that DMC
        predictions and targets can be found at
        target_tensor_no_mask[:, :, :, dmc_index, ...] and
        prediction_tensor[:, :, :, dmc_index, ...], respectively.
    :param dc_index: Same as above but for DC.
    :param expect_ensemble: Boolean flag, indicating whether to expect
        ensemble or deterministic predictions.
    :return: prediction_tensor: Same as input but with extra BUI channel at the
        end.  This tensor has dimensions E x M x N x (T + 1) x S or
        E x M x N x (T + 1).
    :return: target_tensor_no_mask: Same as input but with extra BUI channel at
        the end.  This tensor has dimensions E x M x N x (T + 1).
    """

    if prediction_tensor is not None:
        target_tensor_no_mask = K.cast(
            target_tensor_no_mask, prediction_tensor.dtype
        )

    target_dmc_tensor = target_tensor_no_mask[..., dmc_index]
    target_dc_tensor = target_tensor_no_mask[..., dc_index]
    tgt_dmc = target_dmc_tensor
    tgt_dc = target_dc_tensor
    target_bui_tensor = tensorflow.where(
        tgt_dmc <= 0.4 * tgt_dc,
        (0.8 * tgt_dmc * tgt_dc) / (tgt_dmc + 0.4 * tgt_dc),
        tgt_dmc - (1. - 0.8 * tgt_dc / (tgt_dmc + 0.4 * tgt_dc)) / (0.92 + _power(0.0114 * tgt_dmc, 1.7))
    )

    target_bui_tensor = tensorflow.where(
        tensorflow.math.is_finite(target_bui_tensor),
        target_bui_tensor,
        tensorflow.zeros_like(target_bui_tensor)
    )
    target_bui_tensor = K.maximum(target_bui_tensor, 0.)
    target_tensor_no_mask = K.concatenate([
        target_tensor_no_mask,
        K.expand_dims(target_bui_tensor, axis=-1)
    ], axis=-1)

    if prediction_tensor is None:
        return prediction_tensor, target_tensor_no_mask

    if expect_ensemble:
        predicted_dmc_tensor = prediction_tensor[..., dmc_index, :]
        predicted_dc_tensor = prediction_tensor[..., dc_index, :]
    else:
        predicted_dmc_tensor = prediction_tensor[..., dmc_index]
        predicted_dc_tensor = prediction_tensor[..., dc_index]

    pred_dmc = predicted_dmc_tensor
    pred_dc = predicted_dc_tensor
    predicted_bui_tensor = tensorflow.where(
        pred_dmc <= 0.4 * pred_dc,
        (0.8 * pred_dmc * pred_dc) / (pred_dmc + 0.4 * pred_dc),
        pred_dmc - (1. - 0.8 * pred_dc / (pred_dmc + 0.4 * pred_dc)) / (0.92 + _power(0.0114 * pred_dmc, 1.7))
    )

    predicted_bui_tensor = tensorflow.where(
        tensorflow.math.is_finite(predicted_bui_tensor),
        predicted_bui_tensor,
        tensorflow.zeros_like(predicted_bui_tensor)
    )
    predicted_bui_tensor = K.maximum(predicted_bui_tensor, 0.)

    if expect_ensemble:
        prediction_tensor = K.concatenate([
            prediction_tensor,
            K.expand_dims(predicted_bui_tensor, axis=-2)
        ], axis=-2)
    else:
        prediction_tensor = K.concatenate([
            prediction_tensor,
            K.expand_dims(predicted_bui_tensor, axis=-1)
        ], axis=-1)

    return prediction_tensor, target_tensor_no_mask


# def _add_bui_to_tensors(prediction_tensor, target_tensor_no_mask,
#                         dmc_index, dc_index, expect_ensemble):
#     """Adds build-up index (BUI) to each tensor.
#
#     E = number of examples (time steps)
#     M = number of rows in grid
#     N = number of columns in grid
#     T = number of target fields, not including BUI
#     S = ensemble size
#
#     :param prediction_tensor: Tensor of predicted fire-weather values.  If
#         `expect_ensemble == True`, this must have dimensions E x M x N x T x S;
#         otherwise, must have dimensions E x M x N x T.
#     :param target_tensor_no_mask: Tensor of actual fire-weather values.  This
#         must have dimensions E x M x N x T.
#     :param dmc_index: Array index for DMC.  This tells the method that DMC
#         predictions and targets can be found at
#         target_tensor_no_mask[:, :, :, dmc_index, ...] and
#         prediction_tensor[:, :, :, dmc_index, ...], respectively.
#     :param dc_index: Same as above but for DC.
#     :param expect_ensemble: Boolean flag, indicating whether to expect
#         ensemble or deterministic predictions.
#     :return: prediction_tensor: Same as input but with extra BUI channel at the
#         end.  This tensor has dimensions E x M x N x (T + 1) x S or
#         E x M x N x (T + 1).
#     :return: target_tensor_no_mask: Same as input but with extra BUI channel at
#         the end.  This tensor has dimensions E x M x N x (T + 1).
#     """
#
#     if prediction_tensor is not None:
#         target_tensor_no_mask = K.cast(
#             target_tensor_no_mask, prediction_tensor.dtype
#         )
#
#     target_dmc_tensor = target_tensor_no_mask[..., dmc_index]
#     target_dc_tensor = target_tensor_no_mask[..., dc_index]
#     tgt_dmc = target_dmc_tensor
#     tgt_dc = target_dc_tensor
#
#     target_bui_tensor = (0.8 * tgt_dmc * tgt_dc) / (tgt_dmc + 0.4 * tgt_dc)
#     target_bui_tensor = tensorflow.where(
#         tensorflow.math.is_finite(target_bui_tensor),
#         target_bui_tensor,
#         tensorflow.zeros_like(target_bui_tensor)
#     )
#     target_bui_tensor = K.maximum(target_bui_tensor, 0.)
#
#     first_term = (tgt_dmc - target_bui_tensor) / tgt_dmc
#     first_term = tensorflow.where(
#         tensorflow.math.is_finite(first_term),
#         first_term,
#         tensorflow.zeros_like(first_term)
#     )
#     first_term = K.maximum(first_term, 0.)
#
#     second_term = 0.92 + _power(0.0114 * tgt_dmc, 1.7)
#     tgt_prelim_bui = tgt_dmc - first_term * second_term
#     tgt_prelim_bui = tensorflow.where(
#         tensorflow.math.is_finite(tgt_prelim_bui),
#         tgt_prelim_bui,
#         tensorflow.zeros_like(tgt_prelim_bui)
#     )
#     tgt_prelim_bui = K.maximum(tgt_prelim_bui, 0.)
#
#     target_bui_tensor = tensorflow.where(
#         target_bui_tensor < tgt_dmc,
#         tgt_prelim_bui,
#         target_bui_tensor
#     )
#     target_tensor_no_mask = K.concatenate([
#         target_tensor_no_mask,
#         K.expand_dims(target_bui_tensor, axis=-1)
#     ], axis=-1)
#
#     if prediction_tensor is None:
#         return prediction_tensor, target_tensor_no_mask
#
#     if expect_ensemble:
#         predicted_dmc_tensor = prediction_tensor[..., dmc_index, :]
#         predicted_dc_tensor = prediction_tensor[..., dc_index, :]
#     else:
#         predicted_dmc_tensor = prediction_tensor[..., dmc_index]
#         predicted_dc_tensor = prediction_tensor[..., dc_index]
#
#     pred_dmc = predicted_dmc_tensor
#     pred_dc = predicted_dc_tensor
#
#     predicted_bui_tensor = (0.8 * pred_dmc * pred_dc) / (pred_dmc + 0.4 * pred_dc)
#     predicted_bui_tensor = tensorflow.where(
#         tensorflow.math.is_finite(predicted_bui_tensor),
#         predicted_bui_tensor,
#         tensorflow.zeros_like(predicted_bui_tensor)
#     )
#     predicted_bui_tensor = K.maximum(predicted_bui_tensor, 0.)
#
#     first_term = (pred_dmc - predicted_bui_tensor) / pred_dmc
#     first_term = tensorflow.where(
#         tensorflow.math.is_finite(first_term),
#         first_term,
#         tensorflow.zeros_like(first_term)
#     )
#     first_term = K.maximum(first_term, 0.)
#
#     second_term = 0.92 + _power(0.0114 * pred_dmc, 1.7)
#     pred_prelim_bui = pred_dmc - first_term * second_term
#     pred_prelim_bui = tensorflow.where(
#         tensorflow.math.is_finite(pred_prelim_bui),
#         pred_prelim_bui,
#         tensorflow.zeros_like(pred_prelim_bui)
#     )
#     pred_prelim_bui = K.maximum(pred_prelim_bui, 0.)
#
#     predicted_bui_tensor = tensorflow.where(
#         predicted_bui_tensor < pred_dmc,
#         pred_prelim_bui,
#         predicted_bui_tensor
#     )
#
#     if expect_ensemble:
#         prediction_tensor = K.concatenate([
#             prediction_tensor,
#             K.expand_dims(predicted_bui_tensor, axis=-2)
#         ], axis=-2)
#     else:
#         prediction_tensor = K.concatenate([
#             prediction_tensor,
#             K.expand_dims(predicted_bui_tensor, axis=-1)
#         ], axis=-1)
#
#     return prediction_tensor, target_tensor_no_mask


def _add_fwi_to_tensors(prediction_tensor, target_tensor_no_mask,
                        isi_index, bui_index, expect_ensemble):
    """Adds fire-weather index (FWI) to each tensor.

    E = number of examples (time steps)
    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields, not including FWI
    S = ensemble size

    :param prediction_tensor: Tensor of predicted fire-weather values.  If
        `expect_ensemble == True`, this must have dimensions E x M x N x T x S;
        otherwise, must have dimensions E x M x N x T.
    :param target_tensor_no_mask: Tensor of actual fire-weather values.  This
        must have dimensions E x M x N x T.
    :param isi_index: Array index for ISI.  This tells the method that ISI
        predictions and targets can be found at
        target_tensor_no_mask[:, :, :, isi_index, ...] and
        prediction_tensor[:, :, :, isi_index, ...], respectively.
    :param bui_index: Same as above but for BUI.
    :param expect_ensemble: Boolean flag, indicating whether to expect
        ensemble or deterministic predictions.
    :return: prediction_tensor: Same as input but with extra FWI channel at the
        end.  This tensor has dimensions E x M x N x (T + 1) x S or
        E x M x N x (T + 1).
    :return: target_tensor_no_mask: Same as input but with extra FWI channel at
        the end.  This tensor has dimensions E x M x N x (T + 1).
    """

    if prediction_tensor is not None:
        target_tensor_no_mask = K.cast(
            target_tensor_no_mask, prediction_tensor.dtype
        )

    target_isi_tensor = target_tensor_no_mask[..., isi_index]
    target_bui_tensor = target_tensor_no_mask[..., bui_index]
    tgt_isi = target_isi_tensor
    tgt_bui = target_bui_tensor
    tgt_duff_moisture_func = tensorflow.where(
        tgt_bui <= 80.,
        0.626 * _power(tgt_bui, 0.809) + 2,
        1000. / (25 + 108.64 * _exponential(-0.023 * tgt_bui))
    )
    tgt_prelim_fwi = 0.1 * tgt_isi * tgt_duff_moisture_func
    target_fwi_tensor = tensorflow.where(
        tgt_prelim_fwi > 1.,
        _exponential(2.72 * _power(0.434 * _natural_log(tgt_prelim_fwi), 0.647)),
        tgt_prelim_fwi
    )

    target_fwi_tensor = tensorflow.where(
        tensorflow.math.is_finite(target_fwi_tensor),
        target_fwi_tensor,
        tensorflow.zeros_like(target_fwi_tensor)
    )
    target_fwi_tensor = K.maximum(target_fwi_tensor, 0.)
    target_tensor_no_mask = K.concatenate([
        target_tensor_no_mask,
        K.expand_dims(target_fwi_tensor, axis=-1)
    ], axis=-1)

    if prediction_tensor is None:
        return prediction_tensor, target_tensor_no_mask

    if expect_ensemble:
        predicted_isi_tensor = prediction_tensor[..., isi_index, :]
        predicted_bui_tensor = prediction_tensor[..., bui_index, :]
    else:
        predicted_isi_tensor = prediction_tensor[..., isi_index]
        predicted_bui_tensor = prediction_tensor[..., bui_index]

    pred_isi = predicted_isi_tensor
    pred_bui = predicted_bui_tensor

    pred_duff_moisture_func = tensorflow.where(
        pred_bui <= 80.,
        0.626 * _power(pred_bui, 0.809) + 2,
        1000. / (25 + 108.64 * _exponential(-0.023 * pred_bui))
    )
    pred_prelim_fwi = 0.1 * pred_isi * pred_duff_moisture_func
    predicted_fwi_tensor = tensorflow.where(
        pred_prelim_fwi > 1.,
        _exponential(2.72 * _power(0.434 * _natural_log(pred_prelim_fwi), 0.647)),
        pred_prelim_fwi
    )

    predicted_fwi_tensor = tensorflow.where(
        tensorflow.math.is_finite(predicted_fwi_tensor),
        predicted_fwi_tensor,
        tensorflow.zeros_like(predicted_fwi_tensor)
    )
    predicted_fwi_tensor = K.maximum(predicted_fwi_tensor, 0.)

    if expect_ensemble:
        prediction_tensor = K.concatenate([
            prediction_tensor,
            K.expand_dims(predicted_fwi_tensor, axis=-2)
        ], axis=-2)
    else:
        prediction_tensor = K.concatenate([
            prediction_tensor,
            K.expand_dims(predicted_fwi_tensor, axis=-1)
        ], axis=-1)

    return prediction_tensor, target_tensor_no_mask


def _add_dsr_to_tensors(prediction_tensor, target_tensor_no_mask,
                        fwi_index, expect_ensemble):
    """Adds daily severity rating (DSR) to each tensor.

    E = number of examples (time steps)
    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields, not including DSR
    S = ensemble size

    :param prediction_tensor: Tensor of predicted fire-weather values.  If
        `expect_ensemble == True`, this must have dimensions E x M x N x T x S;
        otherwise, must have dimensions E x M x N x T.
    :param target_tensor_no_mask: Tensor of actual fire-weather values.  This
        must have dimensions E x M x N x T.
    :param fwi_index: Array index for FWI.  This tells the method that FWI
        predictions and targets can be found at
        target_tensor_no_mask[:, :, :, fwi_index, ...] and
        prediction_tensor[:, :, :, fwi_index, ...], respectively.
    :param expect_ensemble: Boolean flag, indicating whether to expect
        ensemble or deterministic predictions.
    :return: prediction_tensor: Same as input but with extra DSR channel at the
        end.  This tensor has dimensions E x M x N x (T + 1) x S or
        E x M x N x (T + 1).
    :return: target_tensor_no_mask: Same as input but with extra DSR channel at
        the end.  This tensor has dimensions E x M x N x (T + 1).
    """

    if prediction_tensor is not None:
        target_tensor_no_mask = K.cast(
            target_tensor_no_mask, prediction_tensor.dtype
        )

    target_dsr_tensor = 0.0272 * _power(
        target_tensor_no_mask[..., fwi_index], 1.77
    )
    target_tensor_no_mask = K.concatenate([
        target_tensor_no_mask,
        K.expand_dims(target_dsr_tensor, axis=-1)
    ], axis=-1)

    if prediction_tensor is None:
        return prediction_tensor, target_tensor_no_mask

    if expect_ensemble:
        predicted_dsr_tensor = 0.0272 * _power(
            prediction_tensor[..., fwi_index, :], 1.77
        )
        prediction_tensor = K.concatenate([
            prediction_tensor,
            K.expand_dims(predicted_dsr_tensor, axis=-2)
        ], axis=-2)
    else:
        predicted_dsr_tensor = 0.0272 * _power(
            prediction_tensor[..., fwi_index], 1.77
        )
        prediction_tensor = K.concatenate([
            prediction_tensor,
            K.expand_dims(predicted_dsr_tensor, axis=-1)
        ], axis=-1)

    return prediction_tensor, target_tensor_no_mask


def mean_squared_error(function_name, expect_ensemble=True, test_mode=False):
    """Creates mean squared error (MSE) loss function.

    E = number of examples (time steps)
    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields
    S = ensemble size

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

        :param target_tensor: This could be an E-by-M-by-N-by-(T + 1) tensor,
            where target_tensor[..., :-1] contains the actual target values and
            target_tensor[..., -1] contains weights.

        This could also be an E-by-M-by-N-by-(2T + 1) tensor, where
            target_tensor[..., :T] contains the actual target values;
            target_tensor[..., T:-1] contains raw-GFS-forecast target values;
            and target_tensor[..., -1] contains weights.

        :param prediction_tensor: Tensor of predicted values.  If
            expect_ensemble == True, will expect dimensions E x M x N x T x S.
            Otherwise, will expect E x M x N x T.
        :return: loss: Mean squared error.
        """

        num_target_fields = __get_num_target_fields(
            prediction_tensor=prediction_tensor,
            expect_ensemble=expect_ensemble
        )

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                target_tensor[..., :num_target_fields], axis=-1
            )
            relevant_prediction_tensor = prediction_tensor
            weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)
        else:
            relevant_target_tensor = target_tensor[..., :num_target_fields]
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

    T = number of target fields

    :param channel_weights: length-T numpy array of channel weights.
    :param function_name: See doc for `mean_squared_error`.
    :param max_dual_weight_by_channel: length-T numpy array of maximum dual
        weights.
    :param expect_ensemble: See doc for `mean_squared_error`.
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

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: Mean squared error.
        """

        num_target_fields = __get_num_target_fields(
            prediction_tensor=prediction_tensor,
            expect_ensemble=expect_ensemble
        )

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                target_tensor[..., :num_target_fields], axis=-1
            )
            relevant_prediction_tensor = prediction_tensor
            mask_weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)
        else:
            relevant_target_tensor = target_tensor[..., :num_target_fields]
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


def dual_weighted_mse_all_constraints(
        channel_weights, dmc_index, dc_index, isi_index, function_name,
        max_dual_weight_by_channel=None, expect_ensemble=True, test_mode=False):
    """Creates DWMSE loss function with constrained BUI, FWI, and DSR.

    K = number of output channels (target variables) -- not including BUI, FWI,
        and DSR, which will be computed on the fly

    :param channel_weights: length-(K + 3) numpy array of channel weights.  The
        last three weights must correspond to BUI, FWI, and DSR -- in that
        order.
    :param dmc_index: Array index for DMC.  This tells the function that DMC
        predictions and targets can be found at
        target_tensor[:, :, :, dmc_index, ...] and
        prediction_tensor[:, :, :, dmc_index, ...], respectively.
    :param dc_index: Same as above but for DC.
    :param isi_index: Same as above but for ISI.
    :param function_name: Function name (string).
    :param max_dual_weight_by_channel: length-(K + 3) numpy array of maximum
        dual weights.  The last three values must correspond to BUI, FWI, and
        DSR -- in that order.
    :param expect_ensemble: Boolean flag.  If True, will expect
        prediction_tensor to have dimensions E x M x N x T x S.  If False, will
        expect prediction_tensor to have dimensions E x M x N x T.
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    _check_index_args(
        dmc_index=dmc_index, dc_index=dc_index, isi_index=isi_index
    )

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    if max_dual_weight_by_channel is None:
        max_dual_weight_by_channel = numpy.full(len(channel_weights), 1e12)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: DWMSE.
        """

        num_target_fields = __get_num_target_fields(
            prediction_tensor=prediction_tensor,
            expect_ensemble=expect_ensemble
        )

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        target_tensor_no_mask = target_tensor[..., :num_target_fields]
        mask_weight_tensor = target_tensor[..., -1]

        prediction_tensor, target_tensor_no_mask = _add_bui_to_tensors(
            prediction_tensor=prediction_tensor,
            target_tensor_no_mask=target_tensor_no_mask,
            dmc_index=dmc_index,
            dc_index=dc_index,
            expect_ensemble=expect_ensemble
        )
        prediction_tensor, target_tensor_no_mask = _add_fwi_to_tensors(
            prediction_tensor=prediction_tensor,
            target_tensor_no_mask=target_tensor_no_mask,
            isi_index=isi_index,
            bui_index=-1,
            expect_ensemble=expect_ensemble
        )
        prediction_tensor, target_tensor_no_mask = _add_dsr_to_tensors(
            prediction_tensor=prediction_tensor,
            target_tensor_no_mask=target_tensor_no_mask,
            fwi_index=-1,
            expect_ensemble=expect_ensemble
        )

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                target_tensor_no_mask, axis=-1
            )
            relevant_prediction_tensor = prediction_tensor
            mask_weight_tensor = K.expand_dims(mask_weight_tensor, axis=-1)
        else:
            relevant_target_tensor = target_tensor_no_mask
            relevant_prediction_tensor = prediction_tensor

        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )
        max_dual_weight_tensor = K.cast(
            K.constant(max_dual_weight_by_channel), dual_weight_tensor.dtype
        )
        mdwt = max_dual_weight_tensor
        for _ in range(3):
            mdwt = K.expand_dims(mdwt, axis=0)
        if expect_ensemble:
            mdwt = K.expand_dims(mdwt, axis=-1)
        dual_weight_tensor = K.minimum(dual_weight_tensor, mdwt)

        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        cwt = channel_weight_tensor
        for _ in range(3):
            cwt = K.expand_dims(cwt, axis=0)
        if expect_ensemble:
            cwt = K.expand_dims(cwt, axis=-1)

        error_tensor = (
            cwt * dual_weight_tensor *
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


def dual_weighted_crps_all_constraints(
        channel_weights, dmc_index, dc_index, isi_index, function_name,
        max_dual_weight_by_channel=None, test_mode=False):
    """Creates DWCRPS loss function with constrained BUI, FWI, and DSR.

    :param channel_weights: See documentation for
        `dual_weighted_mse_all_constraints`.
    :param dmc_index: Same.
    :param dc_index: Same.
    :param isi_index: Same.
    :param function_name: Same.
    :param max_dual_weight_by_channel: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    _check_index_args(
        dmc_index=dmc_index, dc_index=dc_index, isi_index=isi_index
    )

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    if max_dual_weight_by_channel is None:
        max_dual_weight_by_channel = numpy.full(len(channel_weights), 1e12)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWCRPS).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: DWCRPS.
        """

        num_target_fields = __get_num_target_fields(
            prediction_tensor=prediction_tensor,
            expect_ensemble=True
        )

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        target_tensor_no_mask = target_tensor[..., :num_target_fields]
        mask_weight_tensor = target_tensor[..., -1]

        prediction_tensor, target_tensor_no_mask = _add_bui_to_tensors(
            prediction_tensor=prediction_tensor,
            target_tensor_no_mask=target_tensor_no_mask,
            dmc_index=dmc_index,
            dc_index=dc_index,
            expect_ensemble=True
        )
        prediction_tensor, target_tensor_no_mask = _add_fwi_to_tensors(
            prediction_tensor=prediction_tensor,
            target_tensor_no_mask=target_tensor_no_mask,
            isi_index=isi_index,
            bui_index=-1,
            expect_ensemble=True
        )
        prediction_tensor, target_tensor_no_mask = _add_dsr_to_tensors(
            prediction_tensor=prediction_tensor,
            target_tensor_no_mask=target_tensor_no_mask,
            fwi_index=-1,
            expect_ensemble=True
        )

        relevant_target_tensor = K.expand_dims(target_tensor_no_mask, axis=-1)
        relevant_prediction_tensor = prediction_tensor
        mask_weight_tensor = K.expand_dims(mask_weight_tensor, axis=-1)

        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )
        max_dual_weight_tensor = K.cast(
            K.constant(max_dual_weight_by_channel), dual_weight_tensor.dtype
        )
        mdwt = max_dual_weight_tensor
        for _ in range(3):
            mdwt = K.expand_dims(mdwt, axis=0)
        mdwt = K.expand_dims(mdwt, axis=-1)
        max_dual_weight_tensor = mdwt
        dual_weight_tensor = K.minimum(dual_weight_tensor, mdwt)

        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        for _ in range(3):
            channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

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
            K.abs(relevant_prediction_tensor), mdwt
        )

        def compute_mapd_1row(
                prediction_tensor_1row, censored_prediction_tensor_1row):
            """Computes MAPD for one grid row.

            MAPD = mean absolute pairwise difference

            :param prediction_tensor_1row: E-by-N-by-T-by-S tensor of
                predictions.
            :param censored_prediction_tensor_1row: Same as
                `prediction_tensor_1row`, except that values above the max
                dual weight have been replaced with the max dual weight.
            :return: mapd_tensor_1row: E-by-N-by-T tensor of mean absolute
                pairwise differences.
            """

            pt1row = prediction_tensor_1row
            cpt1row = censored_prediction_tensor_1row

            return K.mean(
                K.maximum(
                    K.abs(K.expand_dims(cpt1row, axis=-1)),
                    K.abs(K.expand_dims(cpt1row, axis=-2))
                ) *
                K.abs(
                    K.expand_dims(pt1row, axis=-1) -
                    K.expand_dims(pt1row, axis=-2)
                ),
                axis=(-2, -1)
            )

        def loop_body(i, mapd_tensor):
            """Body of while-loop for computing MAPD.

            This method is run once for every iteration through the while-loop,
            i.e., once for every grid row.

            :param i: Index of current grid row.
            :param mapd_tensor: M-by-E-by-N-by-T tensor of MAPD values, which
                this method will update.
            :return: i_new: Index of next grid row.
            :return: mapd_tensor: Updated version of input.
            """

            this_mapd_tensor = compute_mapd_1row(
                prediction_tensor_1row=relevant_prediction_tensor[i, ...],
                censored_prediction_tensor_1row=
                censored_relevant_prediction_tensor[i, ...]
            )

            mapd_tensor = mapd_tensor.write(i, this_mapd_tensor)
            return i + 1, mapd_tensor

        mapd_tensor = tensorflow.TensorArray(
            size=relevant_prediction_tensor.shape[0],
            dtype=tensorflow.float32
        )
        i = tensorflow.constant(0)
        condition = lambda i, mapd_tensor: tensorflow.less(
            i, relevant_prediction_tensor.shape[0]
        )
        _, mapd_tensor = tensorflow.while_loop(
            cond=condition,
            body=loop_body,
            loop_vars=[i, mapd_tensor],
            maximum_iterations=relevant_prediction_tensor.shape[0],
            # parallel_iterations=1,
            # swap_memory=True
        )
        mapd_tensor = mapd_tensor.stack()

        mapd_tensor = tensorflow.transpose(
            mapd_tensor, perm=[1, 0, 2, 3]
        )
        error_tensor = channel_weight_tensor * (
            mean_prediction_error_tensor -
            0.5 * mapd_tensor
        )

        return (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

    loss.__name__ = function_name
    return loss


def dual_weighted_crpss_all_constraints(
        channel_weights, dmc_index, dc_index, isi_index, function_name,
        max_dual_weight_by_channel=None, test_mode=False):
    """Creates DWCRPSS loss function with constrained BUI, FWI, and DSR.

    :param channel_weights: See documentation for
        `dual_weighted_mse_all_constraints`.
    :param dmc_index: Same.
    :param dc_index: Same.
    :param isi_index: Same.
    :param function_name: Same.
    :param max_dual_weight_by_channel: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    _check_index_args(
        dmc_index=dmc_index, dc_index=dc_index, isi_index=isi_index
    )

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    if max_dual_weight_by_channel is None:
        max_dual_weight_by_channel = numpy.full(len(channel_weights), 1e12)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWCRPSS).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: DWCRPSS.
        """

        num_target_fields = __get_num_target_fields(
            prediction_tensor=prediction_tensor,
            expect_ensemble=True
        )

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        target_tensor_no_mask = target_tensor[..., :num_target_fields]
        gfs_prediction_tensor = target_tensor[..., num_target_fields:-1]
        mask_weight_tensor = target_tensor[..., -1]

        prediction_tensor, target_tensor_no_mask = _add_bui_to_tensors(
            prediction_tensor=prediction_tensor,
            target_tensor_no_mask=target_tensor_no_mask,
            dmc_index=dmc_index,
            dc_index=dc_index,
            expect_ensemble=True
        )
        prediction_tensor, target_tensor_no_mask = _add_fwi_to_tensors(
            prediction_tensor=prediction_tensor,
            target_tensor_no_mask=target_tensor_no_mask,
            isi_index=isi_index,
            bui_index=-1,
            expect_ensemble=True
        )
        prediction_tensor, target_tensor_no_mask = _add_dsr_to_tensors(
            prediction_tensor=prediction_tensor,
            target_tensor_no_mask=target_tensor_no_mask,
            fwi_index=-1,
            expect_ensemble=True
        )

        _, gfs_prediction_tensor = _add_bui_to_tensors(
            prediction_tensor=None,
            target_tensor_no_mask=gfs_prediction_tensor,
            dmc_index=dmc_index,
            dc_index=dc_index,
            expect_ensemble=True
        )
        _, gfs_prediction_tensor = _add_fwi_to_tensors(
            prediction_tensor=None,
            target_tensor_no_mask=gfs_prediction_tensor,
            isi_index=isi_index,
            bui_index=-1,
            expect_ensemble=True
        )
        _, gfs_prediction_tensor = _add_dsr_to_tensors(
            prediction_tensor=None,
            target_tensor_no_mask=gfs_prediction_tensor,
            fwi_index=-1,
            expect_ensemble=True
        )

        relevant_target_tensor = K.expand_dims(target_tensor_no_mask, axis=-1)
        relevant_gfs_prediction_tensor = K.expand_dims(
            gfs_prediction_tensor, axis=-1
        )
        relevant_prediction_tensor = prediction_tensor
        mask_weight_tensor = K.expand_dims(mask_weight_tensor, axis=-1)

        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )
        max_dual_weight_tensor = K.cast(
            K.constant(max_dual_weight_by_channel), dual_weight_tensor.dtype
        )
        mdwt = max_dual_weight_tensor
        for _ in range(3):
            mdwt = K.expand_dims(mdwt, axis=0)
        mdwt = K.expand_dims(mdwt, axis=-1)
        max_dual_weight_tensor = mdwt
        dual_weight_tensor = K.minimum(dual_weight_tensor, mdwt)

        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        for _ in range(3):
            channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

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
            K.abs(relevant_prediction_tensor), mdwt
        )

        def compute_mapd_1row(
                prediction_tensor_1row, censored_prediction_tensor_1row):
            """Computes MAPD for one grid row.

            MAPD = mean absolute pairwise difference

            :param prediction_tensor_1row: E-by-N-by-T-by-S tensor of
                predictions.
            :param censored_prediction_tensor_1row: Same as
                `prediction_tensor_1row`, except that values above the max
                dual weight have been replaced with the max dual weight.
            :return: mapd_tensor_1row: E-by-N-by-T tensor of mean absolute
                pairwise differences.
            """

            pt1row = prediction_tensor_1row
            cpt1row = censored_prediction_tensor_1row

            return K.mean(
                K.maximum(
                    K.abs(K.expand_dims(cpt1row, axis=-1)),
                    K.abs(K.expand_dims(cpt1row, axis=-2))
                ) *
                K.abs(
                    K.expand_dims(pt1row, axis=-1) -
                    K.expand_dims(pt1row, axis=-2)
                ),
                axis=(-2, -1)
            )

        def loop_body(i, mapd_tensor):
            """Body of while-loop for computing MAPD.

            This method is run once for every iteration through the while-loop,
            i.e., once for every grid row.

            :param i: Index of current grid row.
            :param mapd_tensor: M-by-E-by-N-by-T tensor of MAPD values, which
                this method will update.
            :return: i_new: Index of next grid row.
            :return: mapd_tensor: Updated version of input.
            """

            this_mapd_tensor = compute_mapd_1row(
                prediction_tensor_1row=relevant_prediction_tensor[i, ...],
                censored_prediction_tensor_1row=
                censored_relevant_prediction_tensor[i, ...]
            )

            mapd_tensor = mapd_tensor.write(i, this_mapd_tensor)
            return i + 1, mapd_tensor

        mapd_tensor = tensorflow.TensorArray(
            size=relevant_prediction_tensor.shape[0],
            dtype=tensorflow.float32
        )
        i = tensorflow.constant(0)
        condition = lambda i, mapd_tensor: tensorflow.less(
            i, relevant_prediction_tensor.shape[0]
        )
        _, mapd_tensor = tensorflow.while_loop(
            cond=condition,
            body=loop_body,
            loop_vars=[i, mapd_tensor],
            maximum_iterations=relevant_prediction_tensor.shape[0],
            # parallel_iterations=1,
            # swap_memory=True
        )
        mapd_tensor = mapd_tensor.stack()

        mapd_tensor = tensorflow.transpose(
            mapd_tensor, perm=[1, 0, 2, 3]
        )
        error_tensor = channel_weight_tensor * (
            mean_prediction_error_tensor -
            0.5 * mapd_tensor
        )
        actual_dwcrps = (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

        gfs_dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_gfs_prediction_tensor)
        )
        gfs_max_dual_weight_tensor = K.cast(
            K.constant(max_dual_weight_by_channel), gfs_dual_weight_tensor.dtype
        )
        gfs_mdwt = gfs_max_dual_weight_tensor
        for _ in range(3):
            gfs_mdwt = K.expand_dims(gfs_mdwt, axis=0)
        gfs_mdwt = K.expand_dims(gfs_mdwt, axis=-1)
        gfs_dual_weight_tensor = K.minimum(gfs_dual_weight_tensor, gfs_mdwt)

        gfs_abs_error_tensor = K.abs(
            relevant_gfs_prediction_tensor - relevant_target_tensor
        )
        gfs_mean_pred_error_tensor = K.mean(
            gfs_dual_weight_tensor * gfs_abs_error_tensor, axis=-1
        )
        gfs_error_tensor = channel_weight_tensor * gfs_mean_pred_error_tensor
        gfs_dwcrps = (
            K.sum(mask_weight_tensor * gfs_error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(gfs_error_tensor))
        )

        return (actual_dwcrps - gfs_dwcrps) / gfs_dwcrps

    loss.__name__ = function_name
    return loss


def dual_weighted_crpss(
        channel_weights, dmc_index, dc_index, isi_index, function_name,
        max_dual_weight_by_channel=None, test_mode=False):
    """Creates DWCRPSS loss function with no constraints.

    :param channel_weights: See documentation for
        `dual_weighted_mse_all_constraints`.
    :param dmc_index: Same.
    :param dc_index: Same.
    :param isi_index: Same.
    :param function_name: Same.
    :param max_dual_weight_by_channel: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    _check_index_args(
        dmc_index=dmc_index, dc_index=dc_index, isi_index=isi_index
    )

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    if max_dual_weight_by_channel is None:
        max_dual_weight_by_channel = numpy.full(len(channel_weights), 1e12)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWCRPSS).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: DWCRPSS.
        """

        num_target_fields = __get_num_target_fields(
            prediction_tensor=prediction_tensor,
            expect_ensemble=True
        )

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        target_tensor_no_mask = target_tensor[..., :num_target_fields]
        gfs_prediction_tensor = target_tensor[..., num_target_fields:-1]
        mask_weight_tensor = target_tensor[..., -1]

        relevant_target_tensor = K.expand_dims(target_tensor_no_mask, axis=-1)
        relevant_gfs_prediction_tensor = K.expand_dims(
            gfs_prediction_tensor, axis=-1
        )
        relevant_prediction_tensor = prediction_tensor
        mask_weight_tensor = K.expand_dims(mask_weight_tensor, axis=-1)

        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )
        max_dual_weight_tensor = K.cast(
            K.constant(max_dual_weight_by_channel), dual_weight_tensor.dtype
        )
        mdwt = max_dual_weight_tensor
        for _ in range(3):
            mdwt = K.expand_dims(mdwt, axis=0)
        mdwt = K.expand_dims(mdwt, axis=-1)
        max_dual_weight_tensor = mdwt
        dual_weight_tensor = K.minimum(dual_weight_tensor, mdwt)

        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        for _ in range(3):
            channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

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
            K.abs(relevant_prediction_tensor), mdwt
        )

        def compute_mapd_1row(
                prediction_tensor_1row, censored_prediction_tensor_1row):
            """Computes MAPD for one grid row.

            MAPD = mean absolute pairwise difference

            :param prediction_tensor_1row: E-by-N-by-T-by-S tensor of
                predictions.
            :param censored_prediction_tensor_1row: Same as
                `prediction_tensor_1row`, except that values above the max
                dual weight have been replaced with the max dual weight.
            :return: mapd_tensor_1row: E-by-N-by-T tensor of mean absolute
                pairwise differences.
            """

            pt1row = prediction_tensor_1row
            cpt1row = censored_prediction_tensor_1row

            return K.mean(
                K.maximum(
                    K.abs(K.expand_dims(cpt1row, axis=-1)),
                    K.abs(K.expand_dims(cpt1row, axis=-2))
                ) *
                K.abs(
                    K.expand_dims(pt1row, axis=-1) -
                    K.expand_dims(pt1row, axis=-2)
                ),
                axis=(-2, -1)
            )

        def loop_body(i, mapd_tensor):
            """Body of while-loop for computing MAPD.

            This method is run once for every iteration through the while-loop,
            i.e., once for every grid row.

            :param i: Index of current grid row.
            :param mapd_tensor: M-by-E-by-N-by-T tensor of MAPD values, which
                this method will update.
            :return: i_new: Index of next grid row.
            :return: mapd_tensor: Updated version of input.
            """

            this_mapd_tensor = compute_mapd_1row(
                prediction_tensor_1row=relevant_prediction_tensor[i, ...],
                censored_prediction_tensor_1row=
                censored_relevant_prediction_tensor[i, ...]
            )

            mapd_tensor = mapd_tensor.write(i, this_mapd_tensor)
            return i + 1, mapd_tensor

        mapd_tensor = tensorflow.TensorArray(
            size=relevant_prediction_tensor.shape[0],
            dtype=tensorflow.float32
        )
        i = tensorflow.constant(0)
        condition = lambda i, mapd_tensor: tensorflow.less(
            i, relevant_prediction_tensor.shape[0]
        )
        _, mapd_tensor = tensorflow.while_loop(
            cond=condition,
            body=loop_body,
            loop_vars=[i, mapd_tensor],
            maximum_iterations=relevant_prediction_tensor.shape[0],
            # parallel_iterations=1,
            # swap_memory=True
        )
        mapd_tensor = mapd_tensor.stack()

        mapd_tensor = tensorflow.transpose(
            mapd_tensor, perm=[1, 0, 2, 3]
        )
        error_tensor = channel_weight_tensor * (
            mean_prediction_error_tensor -
            0.5 * mapd_tensor
        )
        actual_dwcrps = (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

        gfs_dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_gfs_prediction_tensor)
        )
        gfs_max_dual_weight_tensor = K.cast(
            K.constant(max_dual_weight_by_channel), gfs_dual_weight_tensor.dtype
        )
        gfs_mdwt = gfs_max_dual_weight_tensor
        for _ in range(3):
            gfs_mdwt = K.expand_dims(gfs_mdwt, axis=0)
        gfs_mdwt = K.expand_dims(gfs_mdwt, axis=-1)
        gfs_dual_weight_tensor = K.minimum(gfs_dual_weight_tensor, gfs_mdwt)

        gfs_abs_error_tensor = K.abs(
            relevant_gfs_prediction_tensor - relevant_target_tensor
        )
        gfs_mean_pred_error_tensor = K.mean(
            gfs_dual_weight_tensor * gfs_abs_error_tensor, axis=-1
        )
        gfs_error_tensor = channel_weight_tensor * gfs_mean_pred_error_tensor
        gfs_dwcrps = (
            K.sum(mask_weight_tensor * gfs_error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(gfs_error_tensor))
        )

        return (actual_dwcrps - gfs_dwcrps) / gfs_dwcrps

    loss.__name__ = function_name
    return loss


def dual_weighted_crpss_constrained_bui(
        channel_weights, dmc_index, dc_index, isi_index, function_name,
        max_dual_weight_by_channel=None, test_mode=False):
    """Creates DWCRPSS loss function with constrained BUI only.

    :param channel_weights: See documentation for
        `dual_weighted_mse_all_constraints`.
    :param dmc_index: Same.
    :param dc_index: Same.
    :param isi_index: Same.
    :param function_name: Same.
    :param max_dual_weight_by_channel: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    _check_index_args(
        dmc_index=dmc_index, dc_index=dc_index, isi_index=isi_index
    )

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    if max_dual_weight_by_channel is None:
        max_dual_weight_by_channel = numpy.full(len(channel_weights), 1e12)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWCRPSS).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: DWCRPSS.
        """

        num_target_fields = __get_num_target_fields(
            prediction_tensor=prediction_tensor,
            expect_ensemble=True
        )

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        target_tensor_no_mask = target_tensor[..., :num_target_fields]
        gfs_prediction_tensor = target_tensor[..., num_target_fields:-1]
        mask_weight_tensor = target_tensor[..., -1]

        prediction_tensor, target_tensor_no_mask = _add_bui_to_tensors(
            prediction_tensor=prediction_tensor,
            target_tensor_no_mask=target_tensor_no_mask,
            dmc_index=dmc_index,
            dc_index=dc_index,
            expect_ensemble=True
        )

        _, gfs_prediction_tensor = _add_bui_to_tensors(
            prediction_tensor=None,
            target_tensor_no_mask=gfs_prediction_tensor,
            dmc_index=dmc_index,
            dc_index=dc_index,
            expect_ensemble=True
        )

        relevant_target_tensor = K.expand_dims(target_tensor_no_mask, axis=-1)
        relevant_gfs_prediction_tensor = K.expand_dims(
            gfs_prediction_tensor, axis=-1
        )
        relevant_prediction_tensor = prediction_tensor
        mask_weight_tensor = K.expand_dims(mask_weight_tensor, axis=-1)

        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )
        max_dual_weight_tensor = K.cast(
            K.constant(max_dual_weight_by_channel), dual_weight_tensor.dtype
        )
        mdwt = max_dual_weight_tensor
        for _ in range(3):
            mdwt = K.expand_dims(mdwt, axis=0)
        mdwt = K.expand_dims(mdwt, axis=-1)
        max_dual_weight_tensor = mdwt
        dual_weight_tensor = K.minimum(dual_weight_tensor, mdwt)

        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        for _ in range(3):
            channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

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
            K.abs(relevant_prediction_tensor), mdwt
        )

        def compute_mapd_1row(
                prediction_tensor_1row, censored_prediction_tensor_1row):
            """Computes MAPD for one grid row.

            MAPD = mean absolute pairwise difference

            :param prediction_tensor_1row: E-by-N-by-T-by-S tensor of
                predictions.
            :param censored_prediction_tensor_1row: Same as
                `prediction_tensor_1row`, except that values above the max
                dual weight have been replaced with the max dual weight.
            :return: mapd_tensor_1row: E-by-N-by-T tensor of mean absolute
                pairwise differences.
            """

            pt1row = prediction_tensor_1row
            cpt1row = censored_prediction_tensor_1row

            return K.mean(
                K.maximum(
                    K.abs(K.expand_dims(cpt1row, axis=-1)),
                    K.abs(K.expand_dims(cpt1row, axis=-2))
                ) *
                K.abs(
                    K.expand_dims(pt1row, axis=-1) -
                    K.expand_dims(pt1row, axis=-2)
                ),
                axis=(-2, -1)
            )

        def loop_body(i, mapd_tensor):
            """Body of while-loop for computing MAPD.

            This method is run once for every iteration through the while-loop,
            i.e., once for every grid row.

            :param i: Index of current grid row.
            :param mapd_tensor: M-by-E-by-N-by-T tensor of MAPD values, which
                this method will update.
            :return: i_new: Index of next grid row.
            :return: mapd_tensor: Updated version of input.
            """

            this_mapd_tensor = compute_mapd_1row(
                prediction_tensor_1row=relevant_prediction_tensor[i, ...],
                censored_prediction_tensor_1row=
                censored_relevant_prediction_tensor[i, ...]
            )

            mapd_tensor = mapd_tensor.write(i, this_mapd_tensor)
            return i + 1, mapd_tensor

        mapd_tensor = tensorflow.TensorArray(
            size=relevant_prediction_tensor.shape[0],
            dtype=tensorflow.float32
        )
        i = tensorflow.constant(0)
        condition = lambda i, mapd_tensor: tensorflow.less(
            i, relevant_prediction_tensor.shape[0]
        )
        _, mapd_tensor = tensorflow.while_loop(
            cond=condition,
            body=loop_body,
            loop_vars=[i, mapd_tensor],
            maximum_iterations=relevant_prediction_tensor.shape[0],
            # parallel_iterations=1,
            # swap_memory=True
        )
        mapd_tensor = mapd_tensor.stack()

        mapd_tensor = tensorflow.transpose(
            mapd_tensor, perm=[1, 0, 2, 3]
        )
        error_tensor = channel_weight_tensor * (
            mean_prediction_error_tensor -
            0.5 * mapd_tensor
        )
        actual_dwcrps = (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

        gfs_dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_gfs_prediction_tensor)
        )
        gfs_max_dual_weight_tensor = K.cast(
            K.constant(max_dual_weight_by_channel), gfs_dual_weight_tensor.dtype
        )
        gfs_mdwt = gfs_max_dual_weight_tensor
        for _ in range(3):
            gfs_mdwt = K.expand_dims(gfs_mdwt, axis=0)
        gfs_mdwt = K.expand_dims(gfs_mdwt, axis=-1)
        gfs_dual_weight_tensor = K.minimum(gfs_dual_weight_tensor, gfs_mdwt)

        gfs_abs_error_tensor = K.abs(
            relevant_gfs_prediction_tensor - relevant_target_tensor
        )
        gfs_mean_pred_error_tensor = K.mean(
            gfs_dual_weight_tensor * gfs_abs_error_tensor, axis=-1
        )
        gfs_error_tensor = channel_weight_tensor * gfs_mean_pred_error_tensor
        gfs_dwcrps = (
            K.sum(mask_weight_tensor * gfs_error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(gfs_error_tensor))
        )

        return (actual_dwcrps - gfs_dwcrps) / gfs_dwcrps

    loss.__name__ = function_name
    return loss


def dual_weighted_crpss_constrained_bui_fwi(
        channel_weights, dmc_index, dc_index, isi_index, function_name,
        max_dual_weight_by_channel=None, test_mode=False):
    """Creates DWCRPSS loss function with constrained BUI and FWI only.

    :param channel_weights: See documentation for
        `dual_weighted_mse_all_constraints`.
    :param dmc_index: Same.
    :param dc_index: Same.
    :param isi_index: Same.
    :param function_name: Same.
    :param max_dual_weight_by_channel: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    _check_index_args(
        dmc_index=dmc_index, dc_index=dc_index, isi_index=isi_index
    )

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    if max_dual_weight_by_channel is None:
        max_dual_weight_by_channel = numpy.full(len(channel_weights), 1e12)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWCRPSS).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: DWCRPSS.
        """

        num_target_fields = __get_num_target_fields(
            prediction_tensor=prediction_tensor,
            expect_ensemble=True
        )

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        target_tensor_no_mask = target_tensor[..., :num_target_fields]
        gfs_prediction_tensor = target_tensor[..., num_target_fields:-1]
        mask_weight_tensor = target_tensor[..., -1]

        prediction_tensor, target_tensor_no_mask = _add_bui_to_tensors(
            prediction_tensor=prediction_tensor,
            target_tensor_no_mask=target_tensor_no_mask,
            dmc_index=dmc_index,
            dc_index=dc_index,
            expect_ensemble=True
        )
        prediction_tensor, target_tensor_no_mask = _add_fwi_to_tensors(
            prediction_tensor=prediction_tensor,
            target_tensor_no_mask=target_tensor_no_mask,
            isi_index=isi_index,
            bui_index=-1,
            expect_ensemble=True
        )

        _, gfs_prediction_tensor = _add_bui_to_tensors(
            prediction_tensor=None,
            target_tensor_no_mask=gfs_prediction_tensor,
            dmc_index=dmc_index,
            dc_index=dc_index,
            expect_ensemble=True
        )
        _, gfs_prediction_tensor = _add_fwi_to_tensors(
            prediction_tensor=None,
            target_tensor_no_mask=gfs_prediction_tensor,
            isi_index=isi_index,
            bui_index=-1,
            expect_ensemble=True
        )

        relevant_target_tensor = K.expand_dims(target_tensor_no_mask, axis=-1)
        relevant_gfs_prediction_tensor = K.expand_dims(
            gfs_prediction_tensor, axis=-1
        )
        relevant_prediction_tensor = prediction_tensor
        mask_weight_tensor = K.expand_dims(mask_weight_tensor, axis=-1)

        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )
        max_dual_weight_tensor = K.cast(
            K.constant(max_dual_weight_by_channel), dual_weight_tensor.dtype
        )
        mdwt = max_dual_weight_tensor
        for _ in range(3):
            mdwt = K.expand_dims(mdwt, axis=0)
        mdwt = K.expand_dims(mdwt, axis=-1)
        max_dual_weight_tensor = mdwt
        dual_weight_tensor = K.minimum(dual_weight_tensor, mdwt)

        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        for _ in range(3):
            channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

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
            K.abs(relevant_prediction_tensor), mdwt
        )

        def compute_mapd_1row(
                prediction_tensor_1row, censored_prediction_tensor_1row):
            """Computes MAPD for one grid row.

            MAPD = mean absolute pairwise difference

            :param prediction_tensor_1row: E-by-N-by-T-by-S tensor of
                predictions.
            :param censored_prediction_tensor_1row: Same as
                `prediction_tensor_1row`, except that values above the max
                dual weight have been replaced with the max dual weight.
            :return: mapd_tensor_1row: E-by-N-by-T tensor of mean absolute
                pairwise differences.
            """

            pt1row = prediction_tensor_1row
            cpt1row = censored_prediction_tensor_1row

            return K.mean(
                K.maximum(
                    K.abs(K.expand_dims(cpt1row, axis=-1)),
                    K.abs(K.expand_dims(cpt1row, axis=-2))
                ) *
                K.abs(
                    K.expand_dims(pt1row, axis=-1) -
                    K.expand_dims(pt1row, axis=-2)
                ),
                axis=(-2, -1)
            )

        def loop_body(i, mapd_tensor):
            """Body of while-loop for computing MAPD.

            This method is run once for every iteration through the while-loop,
            i.e., once for every grid row.

            :param i: Index of current grid row.
            :param mapd_tensor: M-by-E-by-N-by-T tensor of MAPD values, which
                this method will update.
            :return: i_new: Index of next grid row.
            :return: mapd_tensor: Updated version of input.
            """

            this_mapd_tensor = compute_mapd_1row(
                prediction_tensor_1row=relevant_prediction_tensor[i, ...],
                censored_prediction_tensor_1row=
                censored_relevant_prediction_tensor[i, ...]
            )

            mapd_tensor = mapd_tensor.write(i, this_mapd_tensor)
            return i + 1, mapd_tensor

        mapd_tensor = tensorflow.TensorArray(
            size=relevant_prediction_tensor.shape[0],
            dtype=tensorflow.float32
        )
        i = tensorflow.constant(0)
        condition = lambda i, mapd_tensor: tensorflow.less(
            i, relevant_prediction_tensor.shape[0]
        )
        _, mapd_tensor = tensorflow.while_loop(
            cond=condition,
            body=loop_body,
            loop_vars=[i, mapd_tensor],
            maximum_iterations=relevant_prediction_tensor.shape[0],
            # parallel_iterations=1,
            # swap_memory=True
        )
        mapd_tensor = mapd_tensor.stack()

        mapd_tensor = tensorflow.transpose(
            mapd_tensor, perm=[1, 0, 2, 3]
        )
        error_tensor = channel_weight_tensor * (
            mean_prediction_error_tensor -
            0.5 * mapd_tensor
        )
        actual_dwcrps = (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

        gfs_dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_gfs_prediction_tensor)
        )
        gfs_max_dual_weight_tensor = K.cast(
            K.constant(max_dual_weight_by_channel), gfs_dual_weight_tensor.dtype
        )
        gfs_mdwt = gfs_max_dual_weight_tensor
        for _ in range(3):
            gfs_mdwt = K.expand_dims(gfs_mdwt, axis=0)
        gfs_mdwt = K.expand_dims(gfs_mdwt, axis=-1)
        gfs_dual_weight_tensor = K.minimum(gfs_dual_weight_tensor, gfs_mdwt)

        gfs_abs_error_tensor = K.abs(
            relevant_gfs_prediction_tensor - relevant_target_tensor
        )
        gfs_mean_pred_error_tensor = K.mean(
            gfs_dual_weight_tensor * gfs_abs_error_tensor, axis=-1
        )
        gfs_error_tensor = channel_weight_tensor * gfs_mean_pred_error_tensor
        gfs_dwcrps = (
            K.sum(mask_weight_tensor * gfs_error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(gfs_error_tensor))
        )

        return (actual_dwcrps - gfs_dwcrps) / gfs_dwcrps

    loss.__name__ = function_name
    return loss
