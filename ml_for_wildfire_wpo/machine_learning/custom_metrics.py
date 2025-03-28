"""Custom metrics."""

import tensorflow
from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import error_checking

INFINITY = tensorflow.constant(float('inf'))
MASK_PIXEL_IF_WEIGHT_BELOW = 0.05


def max_prediction_anywhere(channel_index, function_name, expect_ensemble=True,
                            is_nn_evidential=False, test_mode=False):
    """Creates metric to return max prediction anywhere.

    "Anywhere" = at masked or unmasked grid cell

    :param channel_index: Will compute metric for the [k]th channel, where
        k = `channel_index`.
    :param function_name: Function name (string).
    :param expect_ensemble: Boolean flag.  If True, will expect
        prediction_tensor to have dimensions E x M x N x T x S.  If False, will
        expect prediction_tensor to have dimensions E x M x N x T.
    :param is_nn_evidential: Boolean flag.  If True, will expect evidential NN,
        where prediction_tensor has dimensions E x M x N x T x 4.
    :param test_mode: Leave this alone.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (max prediction anywhere).

        E = number of examples
        M = number of grid rows
        N = number of grid columns
        T = number of target variables (channels)
        S = ensemble size

        :param target_tensor: This could be an E-by-M-by-N-by-(T + 1) tensor,
            where target_tensor[..., :-1] contains the actual target values and
            target_tensor[..., -1] contains weights.

        This could also be an E-by-M-by-N-by-(2T + 1) tensor, where
            target_tensor[..., :T] contains the actual target values;
            target_tensor[..., T:-1] contains climo errors;
            and target_tensor[..., -1] contains weights.

        :param prediction_tensor: Tensor of predicted values.  If
            expect_ensemble == True, will expect dimensions E x M x N x T x S.
            Otherwise, if is_nn_evidential == True, will expect dimensions
            E x M x N x T x 4.
            Otherwise, will expect E x M x N x T.
        :return: metric: Max prediction.
        """

        if is_nn_evidential:
            return K.max(prediction_tensor[:, :, :, channel_index, 0])

        return K.max(prediction_tensor[:, :, :, channel_index, ...])

    metric.__name__ = function_name
    return metric


def max_target_anywhere(channel_index, function_name, expect_ensemble=True,
                        is_nn_evidential=False, test_mode=False):
    """Creates metric to return max target anywhere.

    "Anywhere" = at masked or unmasked grid cell

    :param channel_index: Will compute metric for the [k]th channel, where
        k = `channel_index`.
    :param function_name: Function name (string).
    :param expect_ensemble: Boolean flag.  If True, will expect
        prediction_tensor to have dimensions E x M x N x T x S.  If False, will
        expect prediction_tensor to have dimensions E x M x N x T.
    :param is_nn_evidential: Boolean flag.  If True, will expect evidential NN,
        where prediction_tensor has dimensions E x M x N x T x 4.
    :param test_mode: Leave this alone.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (max target anywhere).

        E = number of examples
        M = number of grid rows
        N = number of grid columns
        T = number of target variables (channels)
        S = ensemble size

        :param target_tensor: This could be an E-by-M-by-N-by-(T + 1) tensor,
            where target_tensor[..., :-1] contains the actual target values and
            target_tensor[..., -1] contains weights.

        This could also be an E-by-M-by-N-by-(2T + 1) tensor, where
            target_tensor[..., :T] contains the actual target values;
            target_tensor[..., T:-1] contains climo errors;
            and target_tensor[..., -1] contains weights.

        :param prediction_tensor: Tensor of predicted values.  If
            expect_ensemble == True, will expect dimensions E x M x N x T x S.
            Otherwise, if is_nn_evidential == True, will expect dimensions
            E x M x N x T x 4.
            Otherwise, will expect E x M x N x T.
        :return: metric: Max target.
        """

        return K.max(target_tensor[:, :, :, channel_index])

    metric.__name__ = function_name
    return metric


def min_prediction_anywhere(channel_index, function_name, expect_ensemble=True,
                            is_nn_evidential=False, test_mode=False):
    """Creates metric to return min prediction anywhere.

    "Anywhere" = at masked or unmasked grid cell

    :param channel_index: Will compute metric for the [k]th channel, where
        k = `channel_index`.
    :param function_name: Function name (string).
    :param expect_ensemble: Boolean flag.  If True, will expect
        prediction_tensor to have dimensions E x M x N x T x S.  If False, will
        expect prediction_tensor to have dimensions E x M x N x T.
    :param is_nn_evidential: Boolean flag.  If True, will expect evidential NN,
        where prediction_tensor has dimensions E x M x N x T x 4.
    :param test_mode: Leave this alone.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (min prediction anywhere).

        E = number of examples
        M = number of grid rows
        N = number of grid columns
        T = number of target variables (channels)
        S = ensemble size

        :param target_tensor: This could be an E-by-M-by-N-by-(T + 1) tensor,
            where target_tensor[..., :-1] contains the actual target values and
            target_tensor[..., -1] contains weights.

        This could also be an E-by-M-by-N-by-(2T + 1) tensor, where
            target_tensor[..., :T] contains the actual target values;
            target_tensor[..., T:-1] contains climo errors;
            and target_tensor[..., -1] contains weights.

        :param prediction_tensor: Tensor of predicted values.  If
            expect_ensemble == True, will expect dimensions E x M x N x T x S.
            Otherwise, if is_nn_evidential == True, will expect dimensions
            E x M x N x T x 4.
            Otherwise, will expect E x M x N x T.
        :return: metric: min prediction.
        """

        if is_nn_evidential:
            return K.min(prediction_tensor[:, :, :, channel_index, 0])

        return K.min(prediction_tensor[:, :, :, channel_index, ...])

    metric.__name__ = function_name
    return metric


def min_target_anywhere(channel_index, function_name, expect_ensemble=True,
                        is_nn_evidential=False, test_mode=False):
    """Creates metric to return min target anywhere.

    "Anywhere" = at masked or unmasked grid cell

    :param channel_index: Will compute metric for the [k]th channel, where
        k = `channel_index`.
    :param function_name: Function name (string).
    :param expect_ensemble: Boolean flag.  If True, will expect
        prediction_tensor to have dimensions E x M x N x T x S.  If False, will
        expect prediction_tensor to have dimensions E x M x N x T.
    :param is_nn_evidential: Boolean flag.  If True, will expect evidential NN,
        where prediction_tensor has dimensions E x M x N x T x 4.
    :param test_mode: Leave this alone.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (min target anywhere).

        E = number of examples
        M = number of grid rows
        N = number of grid columns
        T = number of target variables (channels)
        S = ensemble size

        :param target_tensor: This could be an E-by-M-by-N-by-(T + 1) tensor,
            where target_tensor[..., :-1] contains the actual target values and
            target_tensor[..., -1] contains weights.

        This could also be an E-by-M-by-N-by-(2T + 1) tensor, where
            target_tensor[..., :T] contains the actual target values;
            target_tensor[..., T:-1] contains climo errors;
            and target_tensor[..., -1] contains weights.

        :param prediction_tensor: Tensor of predicted values.  If
            expect_ensemble == True, will expect dimensions E x M x N x T x S.
            Otherwise, if is_nn_evidential == True, will expect dimensions
            E x M x N x T x 4.
            Otherwise, will expect E x M x N x T.
        :return: metric: Min target.
        """

        return K.min(target_tensor[:, :, :, channel_index])

    metric.__name__ = function_name
    return metric


def max_prediction_unmasked(channel_index, function_name, expect_ensemble=True,
                            is_nn_evidential=False, test_mode=False):
    """Creates metric to return max unmasked prediction.

    "Unmasked" = at grid cell with weight >= MASK_PIXEL_IF_WEIGHT_BELOW

    :param channel_index: See doc for `max_prediction_anywhere`.
    :param function_name: Same.
    :param expect_ensemble: Same.
    :param is_nn_evidential: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (max unmasked prediction).

        :param target_tensor: See doc for `max_prediction_anywhere`.
        :param prediction_tensor: Same.
        :return: metric: Max prediction.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        # Output shape: E x M x N
        weight_tensor = target_tensor[..., -1]
        mask_tensor = K.cast(
            weight_tensor >= MASK_PIXEL_IF_WEIGHT_BELOW,
            prediction_tensor.dtype
        )

        if is_nn_evidential:
            # Input shapes for multiplication: E x M x N and E x M x N
            return K.max(
                prediction_tensor[:, :, :, channel_index, 0] *
                mask_tensor
            )

        if not expect_ensemble:
            # Input shapes for multiplication: E x M x N and E x M x N
            return K.max(
                prediction_tensor[:, :, :, channel_index] *
                mask_tensor
            )

        # Input shapes for multiplication: E x M x N x S and E x M x N x 1
        return K.max(
            prediction_tensor[:, :, :, channel_index, ...] *
            K.expand_dims(mask_tensor, axis=-1)
        )

    metric.__name__ = function_name
    return metric


def max_target_unmasked(channel_index, function_name, expect_ensemble=True,
                        is_nn_evidential=False, test_mode=False):
    """Creates metric to return max unmasked target.

    "Unmasked" = at grid cell with weight >= MASK_PIXEL_IF_WEIGHT_BELOW

    :param channel_index: See doc for `max_prediction_anywhere`.
    :param function_name: Same.
    :param expect_ensemble: Same.
    :param is_nn_evidential: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (max unmasked target).

        :param target_tensor: See doc for `max_prediction_anywhere`.
        :param prediction_tensor: Same.
        :return: metric: Max target.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        # Output shape: E x M x N
        weight_tensor = target_tensor[..., -1]
        mask_tensor = K.cast(
            weight_tensor >= MASK_PIXEL_IF_WEIGHT_BELOW,
            prediction_tensor.dtype
        )

        return K.max(
            target_tensor[:, :, :, channel_index] *
            mask_tensor
        )

    metric.__name__ = function_name
    return metric


def min_prediction_unmasked(channel_index, function_name, expect_ensemble=True,
                            is_nn_evidential=False, test_mode=False):
    """Creates metric to return min unmasked prediction.

    "Unmasked" = at grid cell with weight >= MASK_PIXEL_IF_WEIGHT_BELOW

    :param channel_index: See doc for `min_prediction_anywhere`.
    :param function_name: Same.
    :param expect_ensemble: Same.
    :param is_nn_evidential: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (min unmasked prediction).

        :param target_tensor: See doc for `min_prediction_anywhere`.
        :param prediction_tensor: Same.
        :return: metric: min prediction.
        """

        # Output shape: E x M x N
        weight_tensor = target_tensor[..., -1]
        mask_tensor = tensorflow.where(
            weight_tensor >= MASK_PIXEL_IF_WEIGHT_BELOW,
            weight_tensor,
            INFINITY
        )
        mask_tensor = K.cast(mask_tensor, prediction_tensor.dtype)

        if is_nn_evidential:
            # Input shapes for multiplication: E x M x N and E x M x N
            return K.min(
                prediction_tensor[:, :, :, channel_index, 0] *
                mask_tensor
            )

        if not expect_ensemble:
            # Input shapes for multiplication: E x M x N and E x M x N
            return K.min(
                prediction_tensor[:, :, :, channel_index] *
                mask_tensor
            )

        # Input shapes for multiplication: E x M x N x S and E x M x N x 1
        return K.min(
            prediction_tensor[:, :, :, channel_index, ...] *
            K.expand_dims(mask_tensor, axis=-1)
        )

    metric.__name__ = function_name
    return metric


def min_target_unmasked(channel_index, function_name, expect_ensemble=True,
                        is_nn_evidential=False, test_mode=False):
    """Creates metric to return min unmasked target.

    "Unmasked" = at grid cell with weight >= MASK_PIXEL_IF_WEIGHT_BELOW

    :param channel_index: See doc for `min_prediction_anywhere`.
    :param function_name: Same.
    :param expect_ensemble: Same.
    :param is_nn_evidential: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (min unmasked target).

        :param target_tensor: See doc for `min_prediction_anywhere`.
        :param prediction_tensor: Same.
        :return: metric: Min target.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        # Output shape: E x M x N
        weight_tensor = target_tensor[..., -1]
        mask_tensor = tensorflow.where(
            weight_tensor >= MASK_PIXEL_IF_WEIGHT_BELOW,
            weight_tensor,
            INFINITY
        )
        mask_tensor = K.cast(mask_tensor, target_tensor.dtype)

        return K.min(
            target_tensor[:, :, :, channel_index] *
            mask_tensor
        )

    metric.__name__ = function_name
    return metric


def mean_squared_error_anywhere(
        channel_index, function_name,
        expect_ensemble=True, is_nn_evidential=False, test_mode=False):
    """Creates function to return mean squared error (MSE) anywhere.

    "Anywhere" = over masked and unmasked grid cells

    :param channel_index: See doc for `max_prediction_anywhere`.
    :param function_name: Same.
    :param expect_ensemble: Same.
    :param is_nn_evidential: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (MSE anywhere).

        :param target_tensor: See doc for `max_prediction_anywhere`.
        :param prediction_tensor: Same.
        :return: metric: MSE anywhere.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if is_nn_evidential:
            relevant_target_tensor = target_tensor[..., channel_index]
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index, 0]
            )
        elif expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                target_tensor[..., channel_index], axis=-1
            )
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index, :]
            )
        else:
            relevant_target_tensor = target_tensor[..., channel_index]
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index]
            )

        squared_error_tensor = (
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )
        return K.mean(squared_error_tensor)

    metric.__name__ = function_name
    return metric


def mean_squared_error_unmasked(
        channel_index, function_name,
        expect_ensemble=True, is_nn_evidential=False, test_mode=False):
    """Creates function to return MSE at unmasked grid cells.

    :param channel_index: See doc for `max_prediction_anywhere`.
    :param function_name: Same.
    :param expect_ensemble: Same.
    :param is_nn_evidential: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (MSE at unmasked grid cells).

        :param target_tensor: See doc for `max_prediction_anywhere`.
        :param prediction_tensor: Same.
        :return: metric: MSE at unmasked grid cells.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if is_nn_evidential:
            relevant_target_tensor = target_tensor[..., channel_index]
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index, 0]
            )
            weight_tensor = target_tensor[..., -1]
        elif expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                target_tensor[..., channel_index], axis=-1
            )
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index, :]
            )
            weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)
        else:
            relevant_target_tensor = target_tensor[..., channel_index]
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index]
            )
            weight_tensor = target_tensor[..., -1]

        squared_error_tensor = (
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )

        mask_tensor = K.cast(
            weight_tensor >= MASK_PIXEL_IF_WEIGHT_BELOW,
            prediction_tensor.dtype
        )
        # return K.sum(squared_error_tensor * mask_tensor) / K.sum(mask_tensor)
        return (
            K.sum(mask_tensor * squared_error_tensor) /
            K.sum(mask_tensor * K.ones_like(squared_error_tensor))
        )

    metric.__name__ = function_name
    return metric


def dual_weighted_mse_anywhere(
        channel_index, function_name,
        expect_ensemble=True, is_nn_evidential=False, test_mode=False):
    """Creates function to return dual-weighted MSE (DWMSE) anywhere.

    "Anywhere" = over masked and unmasked grid cells

    :param channel_index: See doc for `max_prediction_anywhere`.
    :param function_name: Same.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (DWMSE anywhere).

        :param target_tensor: See doc for `max_prediction_anywhere`.
        :param prediction_tensor: Same.
        :return: metric: DWMSE anywhere.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if is_nn_evidential:
            relevant_target_tensor = target_tensor[..., channel_index]
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index, 0]
            )
        elif expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                target_tensor[..., channel_index], axis=-1
            )
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index, :]
            )
        else:
            relevant_target_tensor = target_tensor[..., channel_index]
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index]
            )

        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )
        error_tensor = (
            dual_weight_tensor *
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )

        return K.mean(error_tensor)

    metric.__name__ = function_name
    return metric


def dual_weighted_mse_unmasked(
        channel_index, function_name,
        expect_ensemble=True, is_nn_evidential=False, test_mode=False):
    """Creates function to return DWMSE at unmasked grid cells.

    :param channel_index: See doc for `max_prediction_anywhere`.
    :param function_name: Same.
    :param expect_ensemble: Same.
    :param is_nn_evidential: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(is_nn_evidential)
    error_checking.assert_is_boolean(test_mode)
    assert not (expect_ensemble and is_nn_evidential)

    def metric(target_tensor, prediction_tensor):
        """Computes metric (DWMSE at unmasked grid cells).

        :param target_tensor: See doc for `max_prediction_anywhere`.
        :param prediction_tensor: Same.
        :return: metric: DWMSE at unmasked grid cells.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if is_nn_evidential:
            relevant_target_tensor = target_tensor[..., channel_index]
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index, 0]
            )
            weight_tensor = target_tensor[..., -1]
        elif expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                target_tensor[..., channel_index], axis=-1
            )
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index, :]
            )
            weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)
        else:
            relevant_target_tensor = target_tensor[..., channel_index]
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index]
            )
            weight_tensor = target_tensor[..., -1]

        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )
        error_tensor = (
            dual_weight_tensor *
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )

        mask_tensor = K.cast(
            weight_tensor >= MASK_PIXEL_IF_WEIGHT_BELOW,
            prediction_tensor.dtype
        )
        # return K.sum(error_tensor * mask_tensor) / K.sum(mask_tensor)
        return (
            K.sum(mask_tensor * error_tensor) /
            K.sum(mask_tensor * K.ones_like(error_tensor))
        )

    metric.__name__ = function_name
    return metric
