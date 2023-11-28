"""Custom metrics."""

from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import error_checking


def max_prediction_anywhere(channel_index, function_name, test_mode=False):
    """Creates metric to return max prediction anywhere.

    "Anywhere" = at masked or unmasked grid cell

    :param channel_index: Will compute metric for the [k]th channel, where
        k = `channel_index`.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (max prediction anywhere).

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
        :return: metric: Max prediction.
        """

        return K.max(prediction_tensor[:, :, :, channel_index, ...])

    metric.__name__ = function_name
    return metric


def max_prediction_unmasked(channel_index, function_name, test_mode=False):
    """Creates metric to return max unmasked prediction.

    "Unmasked" = at grid cell with weight >= 0.05

    :param channel_index: See doc for `max_prediction_anywhere`.
    :param function_name: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (max unmasked prediction).

        :param target_tensor: See doc for `max_prediction_anywhere`.
        :param prediction_tensor: Same.
        :return: metric: Max prediction.
        """

        weight_tensor = target_tensor[..., -1]
        mask_tensor = K.cast(weight_tensor >= 0.05, prediction_tensor.dtype)
        return K.max(
            prediction_tensor[:, :, :, channel_index, ...] *
            K.expand_dims(mask_tensor, axis=-1)
        )

    metric.__name__ = function_name
    return metric


def mean_squared_error_anywhere(channel_index, function_name, test_mode=False):
    """Creates function to return mean squared error (MSE) anywhere.

    "Anywhere" = over masked and unmasked grid cells

    :param channel_index: See doc for `max_prediction_anywhere`.
    :param function_name: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (MSE anywhere).

        :param target_tensor: See doc for `max_prediction_anywhere`.
        :param prediction_tensor: Same.
        :return: metric: MSE anywhere.
        """

        squared_error_tensor = (
            target_tensor[..., channel_index] -
            K.mean(prediction_tensor[:, :, :, channel_index, ...], axis=-1)
        ) ** 2

        return K.mean(squared_error_tensor)

    metric.__name__ = function_name
    return metric


def mean_squared_error_unmasked(channel_index, function_name, test_mode=False):
    """Creates function to return MSE at unmasked grid cells.

    :param channel_index: See doc for `max_prediction_anywhere`.
    :param function_name: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (MSE at unmasked grid cells).

        :param target_tensor: See doc for `max_prediction_anywhere`.
        :param prediction_tensor: Same.
        :return: metric: MSE at unmasked grid cells.
        """

        squared_error_tensor = (
            target_tensor[..., channel_index] -
            K.mean(prediction_tensor[:, :, :, channel_index, ...], axis=-1)
        ) ** 2

        weight_tensor = target_tensor[..., -1]
        mask_tensor = K.cast(weight_tensor >= 0.05, prediction_tensor.dtype)
        return K.sum(squared_error_tensor * mask_tensor) / K.sum(mask_tensor)

    metric.__name__ = function_name
    return metric


def dual_weighted_mse_anywhere(channel_index, function_name, test_mode=False):
    """Creates function to return dual-weighted MSE (DWMSE) anywhere.

    "Anywhere" = over masked and unmasked grid cells

    :param channel_index: See doc for `max_prediction_anywhere`.
    :param function_name: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (DWMSE anywhere).

        :param target_tensor: See doc for `max_prediction_anywhere`.
        :param prediction_tensor: Same.
        :return: metric: DWMSE anywhere.
        """

        ensemble_mean_prediction_tensor = K.mean(
            prediction_tensor[:, :, :, channel_index, ...], axis=-1
        )
        dual_weight_tensor = K.maximum(
            K.abs(target_tensor[..., channel_index]),
            K.abs(ensemble_mean_prediction_tensor)
        )
        error_tensor = (
            dual_weight_tensor *
            (target_tensor[..., channel_index] -
             ensemble_mean_prediction_tensor) ** 2
        )

        return K.mean(error_tensor)

    metric.__name__ = function_name
    return metric


def dual_weighted_mse_unmasked(channel_index, function_name, test_mode=False):
    """Creates function to return DWMSE at unmasked grid cells.

    :param channel_index: See doc for `max_prediction_anywhere`.
    :param function_name: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (DWMSE at unmasked grid cells).

        :param target_tensor: See doc for `max_prediction_anywhere`.
        :param prediction_tensor: Same.
        :return: metric: DWMSE at unmasked grid cells.
        """

        ensemble_mean_prediction_tensor = K.mean(
            prediction_tensor[:, :, :, channel_index, ...], axis=-1
        )
        dual_weight_tensor = K.maximum(
            K.abs(target_tensor[..., channel_index]),
            K.abs(ensemble_mean_prediction_tensor)
        )
        error_tensor = (
            dual_weight_tensor *
            (target_tensor[..., channel_index] -
             ensemble_mean_prediction_tensor) ** 2
        )

        weight_tensor = target_tensor[..., -1]
        mask_tensor = K.cast(weight_tensor >= 0.05, prediction_tensor.dtype)
        return K.sum(error_tensor * mask_tensor) / K.sum(mask_tensor)

    metric.__name__ = function_name
    return metric
