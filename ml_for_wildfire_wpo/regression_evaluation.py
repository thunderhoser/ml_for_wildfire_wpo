"""Methods for evaluating a regression (not classification) model."""

import os
import sys
import copy
import numpy
import xarray
from numba import njit
from scipy.stats import ks_2samp

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import prediction_io
import canadian_fwi_io
import canadian_fwi_utils
import neural_net
import bias_correction

TOLERANCE = 1e-6

RELIABILITY_BIN_DIM = 'reliability_bin'
BOOTSTRAP_REP_DIM = 'bootstrap_replicate'
LATITUDE_DIM = 'grid_row'
LONGITUDE_DIM = 'grid_column'
FIELD_DIM = 'field'

TARGET_STDEV_KEY = 'target_standard_deviation'
PREDICTION_STDEV_KEY = 'prediction_standard_deviation'
TARGET_MEAN_KEY = 'target_mean'
PREDICTION_MEAN_KEY = 'prediction_mean'

MSE_KEY = 'mean_squared_error'
MSE_BIAS_KEY = 'mse_bias'
MSE_VARIANCE_KEY = 'mse_variance'
MSE_SKILL_SCORE_KEY = 'mse_skill_score'
DWMSE_KEY = 'dual_weighted_mean_squared_error'
DWMSE_SKILL_SCORE_KEY = 'dwmse_skill_score'
EVAL_WEIGHT_KEY = 'evaluation_weight'

KS_STATISTIC_KEY = 'kolmogorov_smirnov_statistic'
KS_P_VALUE_KEY = 'kolmogorov_smirnov_p_value'

MAE_KEY = 'mean_absolute_error'
MAE_SKILL_SCORE_KEY = 'mae_skill_score'
BIAS_KEY = 'bias'
SSRAT_KEY = 'spread_skill_ratio'
SSDIFF_KEY = 'spread_skill_difference'
CORRELATION_KEY = 'correlation'
KGE_KEY = 'kling_gupta_efficiency'
RELIABILITY_KEY = 'reliability'
RELIABILITY_X_KEY = 'reliability_x'
RELIABILITY_Y_KEY = 'reliability_y'
RELIABILITY_BIN_CENTER_KEY = 'reliability_bin_center'
RELIABILITY_COUNT_KEY = 'reliability_count'
INV_RELIABILITY_BIN_CENTER_KEY = 'inv_reliability_bin_center'
INV_RELIABILITY_COUNT_KEY = 'inv_reliability_count'

MODEL_FILE_KEY = 'model_file_name'
ISOTONIC_MODEL_FILE_KEY = 'isotonic_model_file_name'
UNCERTAINTY_CALIB_MODEL_FILE_KEY = 'uncertainty_calib_model_file_name'
PREDICTION_FILES_KEY = 'prediction_file_names'


@njit
def __find_examples_in_each_bin(example_to_bin_indices, num_bins):
    """Creates list of data examples in each bin.

    E = number of examples
    B = number of bins

    :param example_to_bin_indices: length-E numpy array indexing examples to
        bins.  example_to_bin_indices[i] = j indicates that the (i)th example is
        a member of the (j)th bin.
    :param num_bins: Total number of bins (B in the above definitions).
    :return: example_indices_by_bin: length-B list, where the (j)th element is a
        1-D numpy array with the indices of examples belonging to the (j)th bin.
    """

    bin_sizes = numpy.zeros(num_bins, dtype=numpy.int64)
    for j in example_to_bin_indices:
        bin_sizes[j] += 1

    example_indices_by_bin = [
        numpy.empty(bs, dtype=numpy.int64) for bs in bin_sizes
    ]

    bin_counters = numpy.zeros(num_bins, dtype=numpy.int64)
    for i, j in enumerate(example_to_bin_indices):
        example_indices_by_bin[j][bin_counters[j]] = i
        bin_counters[j] += 1

    return example_indices_by_bin


def __weighted_stdev(data_values, weights):
    """Computes weighted standard deviation.

    :param data_values: numpy array of data values.
    :param weights: numpy array of weights with the same shape as `data_values`.
    :return: weighted_stdev: Weighted standard deviation (scalar).
    """

    if numpy.sum(weights) < TOLERANCE:
        return numpy.nan

    weighted_mean = numpy.average(data_values, weights=weights)
    weighted_variance = numpy.average(
        (data_values - weighted_mean) ** 2, weights=weights
    )

    # With all weights being 1, this factor would be N / (N - 1).  This helps
    # convert population stdev to sample stdev.
    correction_factor = (
        numpy.sum(weights) /
        (numpy.sum(weights) - numpy.mean(weights))
    )

    return numpy.sqrt(correction_factor * weighted_variance)


def _get_mse_one_scalar(target_values, predicted_values, per_grid_cell,
                        weights=None):
    """Computes mean squared error (MSE) for one scalar target variable.

    :param target_values: numpy array of target (actual) values.
    :param predicted_values: numpy array of predicted values with the same shape
        as `target_values`.
    :param per_grid_cell: Boolean flag.  If True, will compute a separate set of
        scores at each grid cell.  If False, will compute one set of scores for
        the whole domain.
    :param weights: [used only if `per_grid_cell is None`]
        numpy array of evaluation weights with the same shape as
        `target_values`.
    :return: mse_total: Total MSE.
    :return: mse_bias: Bias component.
    :return: mse_variance: Variance component.
    """

    if per_grid_cell:
        mse_total = numpy.mean(
            (target_values - predicted_values) ** 2, axis=0
        )
        mse_bias = numpy.mean(target_values - predicted_values, axis=0) ** 2
    else:
        if numpy.sum(weights) < TOLERANCE:
            return numpy.nan

        mse_total = numpy.average(
            (target_values - predicted_values) ** 2,
            weights=weights
        )
        mse_bias = numpy.average(
            target_values - predicted_values,
            weights=weights
        ) ** 2

    return mse_total, mse_bias, mse_total - mse_bias


def _get_mse_ss_one_scalar(target_values, predicted_values, per_grid_cell,
                           mean_training_target_value, weights=None):
    """Computes MSE skill score for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :param mean_training_target_value: Mean target value over all training
        examples.
    :param weights: See doc for `_get_mse_one_scalar`.
    :return: mse_skill_score: Self-explanatory.
    """

    mse_actual = _get_mse_one_scalar(
        target_values=target_values,
        predicted_values=predicted_values,
        per_grid_cell=per_grid_cell,
        weights=weights
    )[0]
    mse_climo = _get_mse_one_scalar(
        target_values=target_values,
        predicted_values=mean_training_target_value,
        per_grid_cell=per_grid_cell,
        weights=weights
    )[0]

    return (mse_climo - mse_actual) / mse_climo


def _get_dwmse_one_scalar(target_values, predicted_values, per_grid_cell,
                          weights=None):
    """Computes dual-weighted MSE (DWMSE) for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :param weights: Same.
    :return: dwmse: Self-explanatory.
    """

    dual_weights = numpy.maximum(
        numpy.absolute(target_values),
        numpy.absolute(predicted_values)
    )

    if per_grid_cell:
        return numpy.mean(
            dual_weights * (target_values - predicted_values) ** 2,
            axis=0
        )

    if numpy.sum(weights) < TOLERANCE:
        return numpy.nan

    return numpy.average(
        dual_weights * (target_values - predicted_values) ** 2,
        weights=weights
    )


def _get_dwmse_ss_one_scalar(target_values, predicted_values, per_grid_cell,
                             mean_training_target_value, weights=None):
    """Computes DWMSE skill score for one scalar target variable.

    :param target_values: See doc for `_get_dwmse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :param mean_training_target_value: Mean target value over all training
        examples.
    :param weights: See doc for `_get_dwmse_one_scalar`.
    :return: dwmse_skill_score: Self-explanatory.
    """

    dwmse_actual = _get_dwmse_one_scalar(
        target_values=target_values,
        predicted_values=predicted_values,
        per_grid_cell=per_grid_cell,
        weights=weights
    )
    dwmse_climo = _get_dwmse_one_scalar(
        target_values=target_values,
        predicted_values=numpy.array([mean_training_target_value]),
        per_grid_cell=per_grid_cell,
        weights=weights
    )

    return (dwmse_climo - dwmse_actual) / dwmse_climo


def _get_mae_one_scalar(target_values, predicted_values, per_grid_cell,
                        weights=None):
    """Computes mean absolute error (MAE) for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :param weights: Same.
    :return: mean_absolute_error: Self-explanatory.
    """

    if per_grid_cell:
        return numpy.mean(
            numpy.absolute(target_values - predicted_values),
            axis=0
        )

    if numpy.sum(weights) < TOLERANCE:
        return numpy.nan

    return numpy.average(
        numpy.absolute(target_values - predicted_values),
        weights=weights
    )


def _get_mae_ss_one_scalar(target_values, predicted_values, per_grid_cell,
                           mean_training_target_value, weights=None):
    """Computes MAE skill score for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :param mean_training_target_value: See doc for `_get_mse_ss_one_scalar`.
    :param weights: See doc for `_get_mse_one_scalar`.
    :return: mae_skill_score: Self-explanatory.
    """

    mae_actual = _get_mae_one_scalar(
        target_values=target_values,
        predicted_values=predicted_values,
        per_grid_cell=per_grid_cell,
        weights=weights
    )
    mae_climo = _get_mae_one_scalar(
        target_values=target_values,
        predicted_values=mean_training_target_value,
        per_grid_cell=per_grid_cell,
        weights=weights
    )

    return (mae_climo - mae_actual) / mae_climo


def _get_bias_one_scalar(target_values, predicted_values, per_grid_cell,
                         weights=None):
    """Computes bias (mean signed error) for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :param weights: Same.
    :return: bias: Self-explanatory.
    """

    if per_grid_cell:
        return numpy.mean(predicted_values - target_values, axis=0)

    if numpy.sum(weights) < TOLERANCE:
        return numpy.nan

    return numpy.average(
        predicted_values - target_values,
        weights=weights
    )


def _get_correlation_one_scalar(target_values, predicted_values, per_grid_cell,
                                weights=None):
    """Computes Pearson correlation for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :param weights: Same.
    :return: correlation: Self-explanatory.
    """

    if per_grid_cell:
        numerator = numpy.sum(
            (target_values - numpy.mean(target_values)) *
            (predicted_values - numpy.mean(predicted_values)),
            axis=0
        )
        sum_squared_target_diffs = numpy.sum(
            (target_values - numpy.mean(target_values)) ** 2,
            axis=0
        )
        sum_squared_prediction_diffs = numpy.sum(
            (predicted_values - numpy.mean(predicted_values)) ** 2,
            axis=0
        )
    else:
        if numpy.sum(weights) < TOLERANCE:
            return numpy.nan

        mean_target = numpy.average(target_values, weights=weights)
        mean_prediction = numpy.average(predicted_values, weights=weights)

        numerator = numpy.sum(weights) * numpy.average(
            (target_values - mean_target) *
            (predicted_values - mean_prediction),
            weights=weights
        )
        sum_squared_target_diffs = numpy.sum(weights) * numpy.average(
            (target_values - mean_target) ** 2,
            weights=weights
        )
        sum_squared_prediction_diffs = numpy.sum(weights) * numpy.average(
            (predicted_values - mean_prediction) ** 2,
            weights=weights
        )

    correlation = (
        numerator /
        numpy.sqrt(sum_squared_target_diffs * sum_squared_prediction_diffs)
    )

    return correlation


def _get_kge_one_scalar(target_values, predicted_values, per_grid_cell,
                        weights=None):
    """Computes KGE (Kling-Gupta efficiency) for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :param weights: Same.
    :return: kge: Self-explanatory.
    """

    correlation = _get_correlation_one_scalar(
        target_values=target_values,
        predicted_values=predicted_values,
        per_grid_cell=per_grid_cell,
        weights=weights
    )

    if per_grid_cell:
        mean_target_value = numpy.mean(target_values, axis=0)
        mean_predicted_value = numpy.mean(predicted_values, axis=0)
        stdev_target_value = numpy.std(target_values, ddof=1, axis=0)
        stdev_predicted_value = numpy.std(predicted_values, ddof=1, axis=0)
    else:
        if numpy.sum(weights) < TOLERANCE:
            return numpy.nan

        mean_target_value = numpy.average(target_values, weights=weights)
        mean_predicted_value = numpy.average(predicted_values, weights=weights)
        stdev_target_value = __weighted_stdev(
            data_values=target_values, weights=weights
        )
        stdev_predicted_value = __weighted_stdev(
            data_values=predicted_values, weights=weights
        )

    variance_bias = (
        (stdev_predicted_value / mean_predicted_value) *
        (stdev_target_value / mean_target_value) ** -1
    )
    mean_bias = mean_predicted_value / mean_target_value

    kge = 1. - numpy.sqrt(
        (correlation - 1.) ** 2 +
        (variance_bias - 1.) ** 2 +
        (mean_bias - 1.) ** 2
    )

    return kge


def _get_rel_curve_one_scalar(
        target_values, predicted_values, weights,
        num_bins, min_bin_edge, max_bin_edge, invert=False):
    """Computes reliability curve for one scalar target variable.

    B = number of bins

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param weights: Same.
    :param num_bins: Number of bins (points in curve).
    :param min_bin_edge: Value at lower edge of first bin.
    :param max_bin_edge: Value at upper edge of last bin.
    :param invert: Boolean flag.  If True, will return inverted reliability
        curve, which bins by target value and relates target value to
        conditional mean prediction.  If False, will return normal reliability
        curve, which bins by predicted value and relates predicted value to
        conditional mean observation (target).
    :return: mean_predictions: length-B numpy array of x-coordinates.
    :return: mean_observations: length-B numpy array of y-coordinates.
    :return: example_counts: length-B numpy array with num examples in each bin.
    """

    # max_bin_edge = max([max_bin_edge, numpy.finfo(float).eps])
    # min_bin_edge = min([min_bin_edge, 0.])

    # bin_index_by_example = histograms.create_histogram(
    #     input_values=target_values if invert else predicted_values,
    #     num_bins=num_bins, min_value=min_bin_edge, max_value=max_bin_edge
    # )[0]

    bin_cutoffs = numpy.linspace(min_bin_edge, max_bin_edge, num=num_bins + 1)
    example_to_bin_indices = numpy.digitize(
        target_values if invert else predicted_values, bin_cutoffs, right=False
    ) - 1
    example_to_bin_indices = numpy.clip(example_to_bin_indices, 0, num_bins - 1)

    example_indices_by_bin = __find_examples_in_each_bin(
        example_to_bin_indices=example_to_bin_indices,
        num_bins=num_bins
    )

    mean_predictions = numpy.full(num_bins, numpy.nan)
    mean_observations = numpy.full(num_bins, numpy.nan)
    example_counts = numpy.full(num_bins, numpy.nan)

    for j in range(num_bins):
        if numpy.sum(weights[example_indices_by_bin[j]]) < TOLERANCE:
            example_counts[j] = 0.
            continue

        example_counts[j] = numpy.sum(weights[example_indices_by_bin[j]])
        mean_predictions[j] = numpy.average(
            predicted_values[example_indices_by_bin[j]],
            weights=weights[example_indices_by_bin[j]]
        )
        mean_observations[j] = numpy.average(
            target_values[example_indices_by_bin[j]],
            weights=weights[example_indices_by_bin[j]]
        )

    return mean_predictions, mean_observations, example_counts


def _get_ssrat_one_replicate(
        full_squared_error_matrix, full_prediction_variance_matrix,
        full_weight_matrix, example_indices_in_replicate, per_grid_cell):
    """Computes SSRAT for one bootstrap replicate.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields

    :param full_squared_error_matrix: E-by-M-by-N-by-T numpy array of squared
        errors.
    :param full_prediction_variance_matrix: E-by-M-by-N-by-T numpy array of
        ensemble variances.
    :param full_weight_matrix: E-by-M-by-N numpy array of evaluation weights.
    :param example_indices_in_replicate: See documentation for
        `_get_scores_one_replicate`.
    :param per_grid_cell: Same.
    :return: ssrat_matrix: If `per_grid_cell == True`, this is an M-by-N-by-T
        numpy array of spread-skill ratios.  Otherwise, this is a length-T numpy
        array of spread-skill ratios.
    """

    squared_error_matrix = full_squared_error_matrix[
        example_indices_in_replicate, ...
    ]
    prediction_variance_matrix = full_prediction_variance_matrix[
        example_indices_in_replicate, ...
    ]
    weight_matrix = full_weight_matrix[example_indices_in_replicate, ...]

    num_target_fields = squared_error_matrix.shape[3]

    if per_grid_cell:
        num_grid_rows = squared_error_matrix.shape[1]
        num_grid_columns = squared_error_matrix.shape[2]
        these_dim = ((num_grid_rows, num_grid_columns, num_target_fields))
    else:
        these_dim = (num_target_fields,)

    ssrat_matrix = numpy.full(these_dim, numpy.nan)

    for k in range(num_target_fields):
        if per_grid_cell:
            numerator = numpy.sqrt(
                numpy.mean(prediction_variance_matrix[..., k], axis=0)
            )
            denominator = numpy.sqrt(
                numpy.mean(squared_error_matrix[..., k], axis=0)
            )
        else:
            this_flag_matrix = numpy.invert(numpy.logical_or(
                numpy.isnan(prediction_variance_matrix[..., k]),
                numpy.isnan(squared_error_matrix[..., k])
            ))

            these_variances = (
                prediction_variance_matrix[..., k][this_flag_matrix]
            )
            these_squared_errors = (
                squared_error_matrix[..., k][this_flag_matrix]
            )
            these_weights = weight_matrix[this_flag_matrix]

            numerator = numpy.sqrt(numpy.average(
                these_variances, weights=these_weights
            ))
            denominator = numpy.sqrt(numpy.average(
                these_squared_errors, weights=these_weights
            ))

        ssrat_matrix[..., k] = numerator / denominator

    return ssrat_matrix


def _get_ssdiff_one_replicate(
        full_squared_error_matrix, full_prediction_variance_matrix,
        full_weight_matrix, example_indices_in_replicate, per_grid_cell):
    """Computes SSDIFF for one bootstrap replicate.

    SSDIFF = spread-skill difference = spread minus skill

    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields

    :param full_squared_error_matrix: See documentation for
        `_get_ssrat_one_replicate`.
    :param full_prediction_variance_matrix: Same.
    :param full_weight_matrix: Same.
    :param example_indices_in_replicate: Same.
    :param per_grid_cell: Same.
    :return: ssdiff_matrix: If `per_grid_cell == True`, this is an M-by-N-by-T
        numpy array of spread-skill differences.  Otherwise, this is a length-T
        numpy array of spread-skill differences.
    """

    squared_error_matrix = full_squared_error_matrix[
        example_indices_in_replicate, ...
    ]
    prediction_variance_matrix = full_prediction_variance_matrix[
        example_indices_in_replicate, ...
    ]
    weight_matrix = full_weight_matrix[example_indices_in_replicate, ...]

    num_target_fields = squared_error_matrix.shape[3]

    if per_grid_cell:
        num_grid_rows = squared_error_matrix.shape[1]
        num_grid_columns = squared_error_matrix.shape[2]
        these_dim = ((num_grid_rows, num_grid_columns, num_target_fields))
    else:
        these_dim = (num_target_fields,)

    ssdiff_matrix = numpy.full(these_dim, numpy.nan)

    for k in range(num_target_fields):
        if per_grid_cell:
            numerator = numpy.sqrt(
                numpy.mean(prediction_variance_matrix[..., k], axis=0)
            )
            denominator = numpy.sqrt(
                numpy.mean(squared_error_matrix[..., k], axis=0)
            )
        else:
            this_flag_matrix = numpy.invert(numpy.logical_or(
                numpy.isnan(prediction_variance_matrix[..., k]),
                numpy.isnan(squared_error_matrix[..., k])
            ))

            these_variances = (
                prediction_variance_matrix[..., k][this_flag_matrix]
            )
            these_squared_errors = (
                squared_error_matrix[..., k][this_flag_matrix]
            )
            these_weights = weight_matrix[this_flag_matrix]

            numerator = numpy.sqrt(numpy.average(
                these_variances, weights=these_weights
            ))
            denominator = numpy.sqrt(numpy.average(
                these_squared_errors, weights=these_weights
            ))

        ssdiff_matrix[..., k] = numerator - denominator

    return ssdiff_matrix


def _get_scores_one_replicate(
        result_table_xarray,
        full_target_matrix, full_prediction_matrix, full_weight_matrix,
        replicate_index, example_indices_in_replicate,
        mean_training_target_values, num_relia_bins_by_target,
        min_relia_bin_edge_by_target, max_relia_bin_edge_by_target,
        min_relia_bin_edge_prctile_by_target,
        max_relia_bin_edge_prctile_by_target,
        per_grid_cell, keep_it_simple=False):
    """Computes scores for one bootstrap replicate.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields

    :param result_table_xarray: See doc for `get_scores_with_bootstrapping`.
    :param full_target_matrix: E-by-M-by-N-by-T numpy array of correct values.
    :param full_prediction_matrix: E-by-M-by-N-by-T numpy array of predicted
        values.
    :param full_weight_matrix: E-by-M-by-N numpy array of evaluation weights.
    :param replicate_index: Index of current bootstrap replicate.
    :param example_indices_in_replicate: 1-D numpy array with indices of
        examples in this bootstrap replicate.
    :param mean_training_target_values: length-T numpy array with mean target
        values in training data (i.e., "climatology").
    :param num_relia_bins_by_target: See doc for `get_scores_with_bootstrapping`.
    :param min_relia_bin_edge_by_target: Same.
    :param max_relia_bin_edge_by_target: Same.
    :param min_relia_bin_edge_prctile_by_target: Same.
    :param max_relia_bin_edge_prctile_by_target: Same.
    :param per_grid_cell: Same.
    :param keep_it_simple: Same.
    :return: result_table_xarray: Same as input but with values filled for [i]th
        bootstrap replicate, where i = `replicate_index`.
    """

    t = result_table_xarray
    rep_idx = replicate_index + 0
    num_examples = len(example_indices_in_replicate)

    target_matrix = full_target_matrix[example_indices_in_replicate, ...]
    prediction_matrix = full_prediction_matrix[
        example_indices_in_replicate, ...
    ]
    weight_matrix = full_weight_matrix[example_indices_in_replicate, ...]

    num_target_fields = len(mean_training_target_values)

    if per_grid_cell:
        t[TARGET_STDEV_KEY].values[..., rep_idx] = numpy.std(
            target_matrix, ddof=1, axis=0
        )
        t[PREDICTION_STDEV_KEY].values[..., rep_idx] = numpy.std(
            prediction_matrix, ddof=1, axis=0
        )
        t[TARGET_MEAN_KEY].values[..., rep_idx] = numpy.mean(
            target_matrix, axis=0
        )
        t[PREDICTION_MEAN_KEY].values[..., rep_idx] = numpy.mean(
            prediction_matrix, axis=0
        )
    else:
        t[TARGET_STDEV_KEY].values[:, rep_idx] = numpy.array([
            __weighted_stdev(
                data_values=target_matrix[..., k], weights=weight_matrix
            )
            for k in range(num_target_fields)
        ])

        t[PREDICTION_STDEV_KEY].values[:, rep_idx] = numpy.array([
            __weighted_stdev(
                data_values=prediction_matrix[..., k], weights=weight_matrix
            )
            for k in range(num_target_fields)
        ])

        t[TARGET_MEAN_KEY].values[:, rep_idx] = numpy.array([
            numpy.average(
                target_matrix[..., k], weights=weight_matrix
            )
            for k in range(num_target_fields)
        ])

        t[PREDICTION_MEAN_KEY].values[:, rep_idx] = numpy.array([
            numpy.average(
                prediction_matrix[..., k], weights=weight_matrix
            )
            for k in range(num_target_fields)
        ])

    for k in range(num_target_fields):
        t[MAE_KEY].values[..., k, rep_idx] = _get_mae_one_scalar(
            target_values=target_matrix[..., k],
            predicted_values=prediction_matrix[..., k],
            per_grid_cell=per_grid_cell,
            weights=weight_matrix
        )
        t[MAE_SKILL_SCORE_KEY].values[..., k, rep_idx] = _get_mae_ss_one_scalar(
            target_values=target_matrix[..., k],
            predicted_values=prediction_matrix[..., k],
            mean_training_target_value=mean_training_target_values[k],
            per_grid_cell=per_grid_cell,
            weights=weight_matrix
        )

        (
            t[MSE_KEY].values[..., k, rep_idx],
            t[MSE_BIAS_KEY].values[..., k, rep_idx],
            t[MSE_VARIANCE_KEY].values[..., k, rep_idx]
        ) = _get_mse_one_scalar(
            target_values=target_matrix[..., k],
            predicted_values=prediction_matrix[..., k],
            per_grid_cell=per_grid_cell,
            weights=weight_matrix
        )

        t[MSE_SKILL_SCORE_KEY].values[..., k, rep_idx] = _get_mse_ss_one_scalar(
            target_values=target_matrix[..., k],
            predicted_values=prediction_matrix[..., k],
            mean_training_target_value=mean_training_target_values[k],
            per_grid_cell=per_grid_cell,
            weights=weight_matrix
        )
        t[DWMSE_KEY].values[..., k, rep_idx] = _get_dwmse_one_scalar(
            target_values=target_matrix[..., k],
            predicted_values=prediction_matrix[..., k],
            per_grid_cell=per_grid_cell,
            weights=weight_matrix
        )
        t[DWMSE_SKILL_SCORE_KEY].values[..., k, rep_idx] = (
            _get_dwmse_ss_one_scalar(
                target_values=target_matrix[..., k],
                predicted_values=prediction_matrix[..., k],
                mean_training_target_value=mean_training_target_values[k],
                per_grid_cell=per_grid_cell,
                weights=weight_matrix
            )
        )
        t[BIAS_KEY].values[..., k, rep_idx] = _get_bias_one_scalar(
            target_values=target_matrix[..., k],
            predicted_values=prediction_matrix[..., k],
            per_grid_cell=per_grid_cell,
            weights=weight_matrix
        )
        t[CORRELATION_KEY].values[..., k, rep_idx] = (
            _get_correlation_one_scalar(
                target_values=target_matrix[..., k],
                predicted_values=prediction_matrix[..., k],
                per_grid_cell=per_grid_cell,
                weights=weight_matrix
            )
        )
        t[KGE_KEY].values[..., k, rep_idx] = _get_kge_one_scalar(
            target_values=target_matrix[..., k],
            predicted_values=prediction_matrix[..., k],
            per_grid_cell=per_grid_cell,
            weights=weight_matrix
        )

        if keep_it_simple:
            continue

        if num_examples == 0:
            min_bin_edge = 0.
            max_bin_edge = 1.
        elif min_relia_bin_edge_by_target is not None:
            min_bin_edge = min_relia_bin_edge_by_target[k] + 0.
            max_bin_edge = max_relia_bin_edge_by_target[k] + 0.
        else:
            min_bin_edge = numpy.percentile(
                prediction_matrix[..., k],
                min_relia_bin_edge_prctile_by_target[k]
            )
            max_bin_edge = numpy.percentile(
                prediction_matrix[..., k],
                max_relia_bin_edge_prctile_by_target[k]
            )

        num_bins = num_relia_bins_by_target[k]

        if per_grid_cell:
            num_grid_rows = len(t.coords[LATITUDE_DIM].values)
            num_grid_columns = len(t.coords[LONGITUDE_DIM].values)

            for i in range(num_grid_rows):
                print((
                    'Have computed reliability curve for {0:d} of {1:d} grid '
                    'rows...'
                ).format(
                    i, num_grid_rows
                ))

                for j in range(num_grid_columns):
                    (
                        t[RELIABILITY_X_KEY].values[i, j, k, :num_bins, rep_idx],
                        t[RELIABILITY_Y_KEY].values[i, j, k, :num_bins, rep_idx],
                        these_counts
                    ) = _get_rel_curve_one_scalar(
                        target_values=target_matrix[:, i, j, k],
                        predicted_values=prediction_matrix[:, i, j, k],
                        weights=numpy.ones_like(prediction_matrix[:, i, j, k]),
                        num_bins=num_bins,
                        min_bin_edge=min_bin_edge,
                        max_bin_edge=max_bin_edge,
                        invert=False
                    )

                    these_squared_diffs = (
                        t[RELIABILITY_X_KEY].values[i, j, k, :num_bins, rep_idx] -
                        t[RELIABILITY_Y_KEY].values[i, j, k, :num_bins, rep_idx]
                    ) ** 2

                    t[RELIABILITY_KEY].values[i, j, k, rep_idx] = (
                        numpy.nansum(these_counts * these_squared_diffs) /
                        numpy.sum(these_counts)
                    )

                    if rep_idx == 0:
                        (
                            t[RELIABILITY_BIN_CENTER_KEY].values[i, j, k, :num_bins],
                            _,
                            t[RELIABILITY_COUNT_KEY].values[i, j, k, :num_bins]
                        ) = _get_rel_curve_one_scalar(
                            target_values=full_target_matrix[:, i, j, k],
                            predicted_values=full_prediction_matrix[:, i, j, k],
                            weights=numpy.ones_like(full_prediction_matrix[:, i, j, k]),
                            num_bins=num_bins,
                            min_bin_edge=min_bin_edge,
                            max_bin_edge=max_bin_edge,
                            invert=False
                        )

                        (
                            t[INV_RELIABILITY_BIN_CENTER_KEY].values[i, j, k, :num_bins],
                            _,
                            t[INV_RELIABILITY_COUNT_KEY].values[i, j, k, :num_bins]
                        ) = _get_rel_curve_one_scalar(
                            target_values=full_target_matrix[:, i, j, k],
                            predicted_values=full_prediction_matrix[:, i, j, k],
                            weights=numpy.ones_like(full_prediction_matrix[:, i, j, k]),
                            num_bins=num_bins,
                            min_bin_edge=min_bin_edge,
                            max_bin_edge=max_bin_edge,
                            invert=True
                        )

                    if rep_idx == 0 and full_target_matrix.size > 0:
                        (
                            t[KS_STATISTIC_KEY].values[i, j, k],
                            t[KS_P_VALUE_KEY].values[i, j, k]
                        ) = ks_2samp(
                            full_target_matrix[:, i, j, k],
                            full_prediction_matrix[:, i, j, k],
                            alternative='two-sided',
                            mode='auto'
                        )
        else:
            (
                t[RELIABILITY_X_KEY].values[k, :num_bins, rep_idx],
                t[RELIABILITY_Y_KEY].values[k, :num_bins, rep_idx],
                these_counts
            ) = _get_rel_curve_one_scalar(
                target_values=numpy.ravel(target_matrix[..., k]),
                predicted_values=numpy.ravel(prediction_matrix[..., k]),
                weights=numpy.ravel(weight_matrix),
                num_bins=num_bins,
                min_bin_edge=min_bin_edge,
                max_bin_edge=max_bin_edge,
                invert=False
            )

            these_squared_diffs = (
                t[RELIABILITY_X_KEY].values[k, :num_bins, rep_idx] -
                t[RELIABILITY_Y_KEY].values[k, :num_bins, rep_idx]
            ) ** 2

            t[RELIABILITY_KEY].values[k, rep_idx] = (
                numpy.nansum(these_counts * these_squared_diffs) /
                numpy.sum(these_counts)
            )

            if rep_idx == 0:
                (
                    t[RELIABILITY_BIN_CENTER_KEY].values[k, :num_bins],
                    _,
                    t[RELIABILITY_COUNT_KEY].values[k, :num_bins]
                ) = _get_rel_curve_one_scalar(
                    target_values=numpy.ravel(full_target_matrix[..., k]),
                    predicted_values=numpy.ravel(full_prediction_matrix[..., k]),
                    weights=numpy.ravel(full_weight_matrix),
                    num_bins=num_bins,
                    min_bin_edge=min_bin_edge,
                    max_bin_edge=max_bin_edge,
                    invert=False
                )

                (
                    t[INV_RELIABILITY_BIN_CENTER_KEY].values[k, :num_bins],
                    _,
                    t[INV_RELIABILITY_COUNT_KEY].values[k, :num_bins]
                ) = _get_rel_curve_one_scalar(
                    target_values=numpy.ravel(full_target_matrix[..., k]),
                    predicted_values=numpy.ravel(full_prediction_matrix[..., k]),
                    weights=numpy.ravel(full_weight_matrix),
                    num_bins=num_bins,
                    min_bin_edge=min_bin_edge,
                    max_bin_edge=max_bin_edge,
                    invert=True
                )

            if rep_idx == 0 and full_target_matrix.size > 0:
                (
                    t[KS_STATISTIC_KEY].values[k],
                    t[KS_P_VALUE_KEY].values[k]
                ) = ks_2samp(
                    numpy.ravel(full_target_matrix[..., k]),
                    numpy.ravel(full_prediction_matrix[..., k]),
                    alternative='two-sided',
                    mode='auto'
                )

    return t


def confidence_interval_to_polygon(
        x_value_matrix, y_value_matrix, confidence_level, same_order):
    """Turns confidence interval into polygon.

    P = number of points
    B = number of bootstrap replicates
    V = number of vertices in resulting polygon = 2 * P + 1

    :param x_value_matrix: P-by-B numpy array of x-values.
    :param y_value_matrix: P-by-B numpy array of y-values.
    :param confidence_level: Confidence level (in range 0...1).
    :param same_order: Boolean flag.  If True (False), minimum x-values will be
        matched with minimum (maximum) y-values.
    :return: polygon_coord_matrix: V-by-2 numpy array of coordinates
        (x-coordinates in first column, y-coords in second).
    """

    error_checking.assert_is_numpy_array(x_value_matrix, num_dimensions=2)
    error_checking.assert_is_numpy_array(y_value_matrix, num_dimensions=2)

    expected_dim = numpy.array([
        x_value_matrix.shape[0], y_value_matrix.shape[1]
    ], dtype=int)

    error_checking.assert_is_numpy_array(
        y_value_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_leq(confidence_level, 1.)
    error_checking.assert_is_boolean(same_order)

    min_percentile = 50 * (1. - confidence_level)
    max_percentile = 50 * (1. + confidence_level)

    x_values_bottom = numpy.nanpercentile(
        x_value_matrix, min_percentile, axis=1, interpolation='linear'
    )
    x_values_top = numpy.nanpercentile(
        x_value_matrix, max_percentile, axis=1, interpolation='linear'
    )
    y_values_bottom = numpy.nanpercentile(
        y_value_matrix, min_percentile, axis=1, interpolation='linear'
    )
    y_values_top = numpy.nanpercentile(
        y_value_matrix, max_percentile, axis=1, interpolation='linear'
    )

    real_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(x_values_bottom), numpy.isnan(y_values_bottom)
    )))[0]

    if len(real_indices) == 0:
        return None

    x_values_bottom = x_values_bottom[real_indices]
    x_values_top = x_values_top[real_indices]
    y_values_bottom = y_values_bottom[real_indices]
    y_values_top = y_values_top[real_indices]

    x_vertices = numpy.concatenate((
        x_values_top, x_values_bottom[::-1], x_values_top[[0]]
    ))

    if same_order:
        y_vertices = numpy.concatenate((
            y_values_top, y_values_bottom[::-1], y_values_top[[0]]
        ))
    else:
        y_vertices = numpy.concatenate((
            y_values_bottom, y_values_top[::-1], y_values_bottom[[0]]
        ))

    return numpy.transpose(numpy.vstack((
        x_vertices, y_vertices
    )))


def read_inputs(prediction_file_names, isotonic_model_file_name,
                uncertainty_calib_model_file_name,
                target_field_names, mask_pixel_if_weight_below=0.05):
    """Reads inputs (predictions and targets) from many files.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields
    S = number of ensemble members

    :param prediction_file_names: See documentation for
        `get_scores_with_bootstrapping`.
    :param isotonic_model_file_name: Same.
    :param uncertainty_calib_model_file_name: Same.
    :param target_field_names: length-T list of field names desired.
    :param mask_pixel_if_weight_below: Masking threshold.  For any pixel with an
        evaluation weight below this threshold, both the prediction and target
        will be set to NaN, so that the pixel cannot factor into evaluation.
    :return: target_matrix: E-by-M-by-N-by-T numpy array of target values.
    :return: prediction_matrix: E-by-M-by-N-by-T-by-S numpy array of
        predictions.
    :return: weight_matrix: E-by-M-by-N numpy array of evaluation weights.
    :return: model_file_name: Path to model that generated predictions.
    """

    # TODO(thunderhoser): Put this in prediction_io.py and allow it to return
    # grid coords as well (for use in, e.g., plot_predictions.py).

    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_string_list(target_field_names)

    num_times = len(prediction_file_names)
    target_matrix = numpy.array([], dtype=float)
    prediction_matrix = numpy.array([], dtype=float)
    weight_matrix = numpy.array([], dtype=float)
    model_file_name = None

    if isotonic_model_file_name is None:
        isotonic_model_dict = None
    else:
        print('Reading isotonic-regression model from: "{0:s}"...'.format(
            isotonic_model_file_name
        ))
        isotonic_model_dict = bias_correction.read_file(
            isotonic_model_file_name
        )
        assert not isotonic_model_dict[bias_correction.DO_UNCERTAINTY_CALIB_KEY]

    if uncertainty_calib_model_file_name is None:
        uncertainty_calib_model_dict = None
    else:
        print('Reading uncertainty-calibration model from: "{0:s}"...'.format(
            uncertainty_calib_model_file_name
        ))
        uncertainty_calib_model_dict = bias_correction.read_file(
            uncertainty_calib_model_file_name
        )
        ucmd = uncertainty_calib_model_dict

        assert ucmd[bias_correction.DO_UNCERTAINTY_CALIB_KEY]
        assert (
            ucmd[bias_correction.DO_IR_BEFORE_UC_KEY] ==
            (isotonic_model_file_name is not None)
        )

    for i in range(num_times):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        this_prediction_table_xarray = prediction_io.read_file(
            prediction_file_names[i]
        )

        if isotonic_model_dict is not None:
            this_prediction_table_xarray = (
                bias_correction.apply_model_suite(
                    prediction_table_xarray=this_prediction_table_xarray,
                    model_dict=isotonic_model_dict,
                    verbose=False
                )
            )

        if uncertainty_calib_model_dict is not None:
            this_prediction_table_xarray = (
                bias_correction.apply_model_suite(
                    prediction_table_xarray=this_prediction_table_xarray,
                    model_dict=uncertainty_calib_model_dict,
                    verbose=False
                )
            )

        tpt = this_prediction_table_xarray

        if model_file_name is None:
            these_dim = (
                (num_times,) + tpt[prediction_io.TARGET_KEY].values.shape
            )
            target_matrix = numpy.full(these_dim, numpy.nan)

            these_dim = (
                (num_times,) + tpt[prediction_io.PREDICTION_KEY].values.shape
            )
            prediction_matrix = numpy.full(these_dim, numpy.nan)

            these_dim = (
                (num_times,) + tpt[prediction_io.WEIGHT_KEY].values.shape
            )
            weight_matrix = numpy.full(these_dim, numpy.nan)

            model_file_name = copy.deepcopy(
                tpt.attrs[prediction_io.MODEL_FILE_KEY]
            )

        weight_matrix[i, ...] = tpt[prediction_io.WEIGHT_KEY].values
        # assert model_file_name == tpt.attrs[prediction_io.MODEL_FILE_KEY]

        these_indices = numpy.array([
            numpy.where(tpt[prediction_io.FIELD_NAME_KEY].values == f)[0][0]
            for f in target_field_names
        ], dtype=int)

        target_matrix[i, ...] = (
            tpt[prediction_io.TARGET_KEY].values[..., these_indices]
        )
        prediction_matrix[i, ...] = (
            tpt[prediction_io.PREDICTION_KEY].values[..., these_indices, :]
        )

    target_matrix[weight_matrix < mask_pixel_if_weight_below] = numpy.nan
    prediction_matrix[weight_matrix < mask_pixel_if_weight_below] = numpy.nan

    return target_matrix, prediction_matrix, weight_matrix, model_file_name


def _read_inputs_for_ssrat(prediction_file_names, target_field_names):
    """Reads inputs for calculation of spread-skill ratio (SSRAT).

    :param prediction_file_names: See doc for `read_inputs`.
    :param target_field_names: Same.
    :return: prediction_tables_xarray: length-F list of xarray tables in format
        returned by `prediction_io.read_file`.
    """

    # TODO(thunderhoser): This is a HACK.
    # TODO(thunderhoser): This method does *not* do on-the-fly bias correction.

    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_string_list(target_field_names)

    num_times = len(prediction_file_names)
    prediction_tables_xarray = [xarray.Dataset()] * num_times
    model_file_name = None

    for i in range(num_times):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        prediction_tables_xarray[i] = prediction_io.read_file(
            prediction_file_names[i]
        )
        prediction_tables_xarray[i] = (
            prediction_io.prep_for_uncertainty_calib_training(
                prediction_tables_xarray[i]
            )
        )

        pt_i = prediction_tables_xarray[i]
        if model_file_name is None:
            model_file_name = copy.deepcopy(
                pt_i.attrs[prediction_io.MODEL_FILE_KEY]
            )

        assert model_file_name == pt_i.attrs[prediction_io.MODEL_FILE_KEY]

        these_indices = numpy.array([
            numpy.where(pt_i[prediction_io.FIELD_NAME_KEY].values == f)[0][0]
            for f in target_field_names
        ], dtype=int)

        pt_i = pt_i.isel({prediction_io.FIELD_DIM: these_indices})
        prediction_tables_xarray[i] = pt_i

    return prediction_tables_xarray


def get_scores_with_bootstrapping(
        prediction_file_names, num_bootstrap_reps,
        target_field_names, num_relia_bins_by_target,
        min_relia_bin_edge_by_target, max_relia_bin_edge_by_target,
        min_relia_bin_edge_prctile_by_target,
        max_relia_bin_edge_prctile_by_target,
        per_grid_cell, keep_it_simple=False, compute_ssrat=False,
        isotonic_model_file_name=None, uncertainty_calib_model_file_name=None):
    """Computes all scores with bootstrapping.

    T = number of target fields

    :param prediction_file_names: 1-D list of paths to prediction files.  Each
        file will be read by `prediction_io.read_file`.
    :param num_bootstrap_reps: Number of bootstrap replicates.
    :param target_field_names: length-T list of field names.
    :param num_relia_bins_by_target: length-T numpy array with number of bins in
        reliability curve for each target.
    :param min_relia_bin_edge_by_target: length-T numpy array with minimum
        target/predicted value in reliability curve for each target.  If you
        instead want minimum values to be percentiles over the data, make this
        argument None and use `min_relia_bin_edge_prctile_by_target`.
    :param max_relia_bin_edge_by_target: Same as above but for max.
    :param min_relia_bin_edge_prctile_by_target: length-T numpy array with
        percentile level used to determine minimum target/predicted value in
        reliability curve for each target.  If you instead want to specify raw
        values, make this argument None and use `min_relia_bin_edge_by_target`.
    :param max_relia_bin_edge_prctile_by_target: Same as above but for max.
    :param per_grid_cell: Boolean flag.  If True, will compute a separate set of
        scores at each grid cell.  If False, will compute one set of scores for
        the whole domain.
    :param keep_it_simple: Boolean flag.  If True, will avoid Kolmogorov-Smirnov
        test and attributes diagram.
    :param compute_ssrat: Boolean flag.  If True, will compute spread-skill
        ratio (SSRAT) and spread-skill difference (SSDIFF).
    :param isotonic_model_file_name: Path to file with isotonic-regression
        model, which will be used to bias-correct predictions before evaluation.
        Will be read by `bias_correction.read_file`.  If you do not want to
        bias-correct, make this None.
    :param uncertainty_calib_model_file_name: Path to file with uncertainty-
        calibration model, which will be used to bias-correct uncertainties
        before evaluation.  Will be read by `bias_correction.read_file`.  If
        you do not want to bias-correct uncertainties, make this None.
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_integer(num_bootstrap_reps)
    error_checking.assert_is_greater(num_bootstrap_reps, 0)
    error_checking.assert_is_string_list(target_field_names)
    error_checking.assert_is_boolean(per_grid_cell)
    error_checking.assert_is_boolean(keep_it_simple)

    num_target_fields = len(target_field_names)
    expected_dim = numpy.array([num_target_fields], dtype=int)

    error_checking.assert_is_numpy_array(
        num_relia_bins_by_target, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(num_relia_bins_by_target)
    error_checking.assert_is_geq_numpy_array(num_relia_bins_by_target, 10)
    error_checking.assert_is_leq_numpy_array(num_relia_bins_by_target, 1000)

    if (
            min_relia_bin_edge_by_target is None or
            max_relia_bin_edge_by_target is None
    ):
        error_checking.assert_is_numpy_array(
            min_relia_bin_edge_prctile_by_target, exact_dimensions=expected_dim
        )
        error_checking.assert_is_geq_numpy_array(
            min_relia_bin_edge_prctile_by_target, 0.
        )
        error_checking.assert_is_leq_numpy_array(
            min_relia_bin_edge_prctile_by_target, 10.
        )

        error_checking.assert_is_numpy_array(
            max_relia_bin_edge_prctile_by_target, exact_dimensions=expected_dim
        )
        error_checking.assert_is_geq_numpy_array(
            max_relia_bin_edge_prctile_by_target, 90.
        )
        error_checking.assert_is_leq_numpy_array(
            max_relia_bin_edge_prctile_by_target, 100.
        )
    else:
        error_checking.assert_is_numpy_array(
            min_relia_bin_edge_by_target, exact_dimensions=expected_dim
        )
        error_checking.assert_is_numpy_array(
            max_relia_bin_edge_by_target, exact_dimensions=expected_dim
        )

        for j in range(num_target_fields):
            error_checking.assert_is_greater(
                max_relia_bin_edge_by_target[j],
                min_relia_bin_edge_by_target[j]
            )

    if compute_ssrat:
        prediction_tables_xarray = _read_inputs_for_ssrat(
            prediction_file_names=prediction_file_names,
            target_field_names=target_field_names
        )

        prediction_variance_matrix = numpy.stack([
            ptx[prediction_io.PREDICTION_KEY].values[..., 0]
            for ptx in prediction_tables_xarray
        ], axis=0)

        squared_error_matrix = numpy.stack([
            ptx[prediction_io.TARGET_KEY].values
            for ptx in prediction_tables_xarray
        ], axis=0)

        weight_matrix = numpy.stack([
            ptx[prediction_io.WEIGHT_KEY].values
            for ptx in prediction_tables_xarray
        ], axis=0)

        if per_grid_cell:
            num_grid_rows = squared_error_matrix.shape[1]
            num_grid_columns = squared_error_matrix.shape[2]
            these_dim = (
                num_grid_rows, num_grid_columns,
                num_target_fields, num_bootstrap_reps
            )
        else:
            these_dim = (num_target_fields, num_bootstrap_reps)

        ssrat_matrix = numpy.full(these_dim, numpy.nan)
        ssdiff_matrix = numpy.full(these_dim, numpy.nan)
        num_examples = squared_error_matrix.shape[0]
        example_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )

        for i in range(num_bootstrap_reps):
            if num_bootstrap_reps == 1:
                these_indices = example_indices
            else:
                these_indices = numpy.random.choice(
                    example_indices, size=num_examples, replace=True
                )

            print((
                'Computing SSRAT and SSDIFF for {0:d}th of {1:d} bootstrap '
                'replicates...'
            ).format(
                i + 1, num_bootstrap_reps
            ))

            ssrat_matrix[..., i] = _get_ssrat_one_replicate(
                full_squared_error_matrix=squared_error_matrix,
                full_prediction_variance_matrix=prediction_variance_matrix,
                full_weight_matrix=weight_matrix,
                example_indices_in_replicate=these_indices,
                per_grid_cell=per_grid_cell
            )

            ssdiff_matrix[..., i] = _get_ssdiff_one_replicate(
                full_squared_error_matrix=squared_error_matrix,
                full_prediction_variance_matrix=prediction_variance_matrix,
                full_weight_matrix=weight_matrix,
                example_indices_in_replicate=these_indices,
                per_grid_cell=per_grid_cell
            )
    else:
        ssrat_matrix = None
        ssdiff_matrix = None

    (
        target_matrix, prediction_matrix, weight_matrix, model_file_name
    ) = read_inputs(
        prediction_file_names=prediction_file_names,
        isotonic_model_file_name=isotonic_model_file_name,
        uncertainty_calib_model_file_name=uncertainty_calib_model_file_name,
        target_field_names=target_field_names,
        mask_pixel_if_weight_below=-1.
    )
    prediction_matrix = numpy.mean(prediction_matrix, axis=-1)

    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    target_norm_file_name = generator_option_dict[
        neural_net.TARGET_NORM_FILE_KEY
    ]

    # TODO(thunderhoser): This bit will not work if, for some reason, I do not
    # normalized the lagged-target predictors.
    print('Reading climo-mean target values from: "{0:s}"...'.format(
        target_norm_file_name
    ))
    target_norm_param_table_xarray = canadian_fwi_io.read_normalization_file(
        target_norm_file_name
    )
    tnpt = target_norm_param_table_xarray

    these_indices = numpy.array([
        numpy.where(tnpt.coords[canadian_fwi_utils.FIELD_DIM].values == f)[0][0]
        for f in target_field_names
    ], dtype=int)

    mean_training_target_values = (
        tnpt[canadian_fwi_utils.MEAN_VALUE_KEY].values[these_indices]
    )
    for j in range(num_target_fields):
        print('Climo-mean {0:s} = {1:.4f}'.format(
            target_field_names[j], mean_training_target_values[j]
        ))

    num_grid_rows = target_matrix.shape[1]
    num_grid_columns = target_matrix.shape[2]

    if per_grid_cell:
        these_dimensions = (
            num_grid_rows, num_grid_columns, num_target_fields,
            num_bootstrap_reps
        )
        these_dim_keys = (
            LATITUDE_DIM, LONGITUDE_DIM, FIELD_DIM, BOOTSTRAP_REP_DIM
        )
    else:
        these_dimensions = (num_target_fields, num_bootstrap_reps)
        these_dim_keys = (FIELD_DIM, BOOTSTRAP_REP_DIM)

    main_data_dict = {
        TARGET_STDEV_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        PREDICTION_STDEV_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        TARGET_MEAN_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        PREDICTION_MEAN_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MAE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MAE_SKILL_SCORE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MSE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MSE_BIAS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MSE_VARIANCE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MSE_SKILL_SCORE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        DWMSE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        DWMSE_SKILL_SCORE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        BIAS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        CORRELATION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        KGE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        RELIABILITY_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }

    if per_grid_cell:
        these_dim_keys = (LATITUDE_DIM, LONGITUDE_DIM)
        new_dict = {
            EVAL_WEIGHT_KEY: (these_dim_keys, numpy.mean(weight_matrix, axis=0))
        }
        main_data_dict.update(new_dict)

    if compute_ssrat:
        new_dict = {
            SSRAT_KEY: (these_dim_keys, ssrat_matrix),
            SSDIFF_KEY: (these_dim_keys, ssdiff_matrix)
        }
        main_data_dict.update(new_dict)

    if per_grid_cell:
        these_dimensions = (
            num_grid_rows, num_grid_columns, num_target_fields,
            numpy.max(num_relia_bins_by_target), num_bootstrap_reps
        )
        these_dim_keys = (
            LATITUDE_DIM, LONGITUDE_DIM, FIELD_DIM,
            RELIABILITY_BIN_DIM, BOOTSTRAP_REP_DIM
        )
    else:
        these_dimensions = (
            num_target_fields, numpy.max(num_relia_bins_by_target),
            num_bootstrap_reps
        )
        these_dim_keys = (FIELD_DIM, RELIABILITY_BIN_DIM, BOOTSTRAP_REP_DIM)

    new_dict = {
        RELIABILITY_X_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        RELIABILITY_Y_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    if per_grid_cell:
        these_dimensions = (
            num_grid_rows, num_grid_columns, num_target_fields,
            numpy.max(num_relia_bins_by_target)
        )
        these_dim_keys = (
            LATITUDE_DIM, LONGITUDE_DIM, FIELD_DIM, RELIABILITY_BIN_DIM
        )
    else:
        these_dimensions = (
            num_target_fields, numpy.max(num_relia_bins_by_target)
        )
        these_dim_keys = (FIELD_DIM, RELIABILITY_BIN_DIM)

    new_dict = {
        RELIABILITY_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        RELIABILITY_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        INV_RELIABILITY_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        INV_RELIABILITY_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    if per_grid_cell:
        these_dimensions = (num_grid_rows, num_grid_columns, num_target_fields)
        these_dim_keys = (LATITUDE_DIM, LONGITUDE_DIM, FIELD_DIM)
    else:
        these_dimensions = (num_target_fields,)
        these_dim_keys = (FIELD_DIM,)

    new_dict = {
        KS_STATISTIC_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        KS_P_VALUE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    reliability_bin_indices = numpy.linspace(
        0, numpy.max(num_relia_bins_by_target) - 1,
        num=numpy.max(num_relia_bins_by_target), dtype=int
    )
    bootstrap_indices = numpy.linspace(
        0, num_bootstrap_reps - 1, num=num_bootstrap_reps, dtype=int
    )
    metadata_dict = {
        FIELD_DIM: target_field_names,
        RELIABILITY_BIN_DIM: reliability_bin_indices,
        BOOTSTRAP_REP_DIM: bootstrap_indices
    }

    if per_grid_cell:
        this_prediction_table_xarray = prediction_io.read_file(
            prediction_file_names[0]
        )
        tpt = this_prediction_table_xarray

        metadata_dict.update({
            LATITUDE_DIM: tpt[prediction_io.LATITUDE_KEY].values,
            LONGITUDE_DIM: tpt[prediction_io.LONGITUDE_KEY].values
        })

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    result_table_xarray.attrs[ISOTONIC_MODEL_FILE_KEY] = (
        '' if isotonic_model_file_name is None
        else isotonic_model_file_name
    )
    result_table_xarray.attrs[UNCERTAINTY_CALIB_MODEL_FILE_KEY] = (
        '' if uncertainty_calib_model_file_name is None
        else uncertainty_calib_model_file_name
    )
    result_table_xarray.attrs[PREDICTION_FILES_KEY] = ' '.join([
        '{0:s}'.format(f) for f in prediction_file_names
    ])

    num_examples = target_matrix.shape[0]
    example_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int
    )

    for i in range(num_bootstrap_reps):
        if num_bootstrap_reps == 1:
            these_indices = example_indices
        else:
            these_indices = numpy.random.choice(
                example_indices, size=num_examples, replace=True
            )

        print((
            'Computing scores for {0:d}th of {1:d} bootstrap replicates...'
        ).format(
            i + 1, num_bootstrap_reps
        ))

        result_table_xarray = _get_scores_one_replicate(
            result_table_xarray=result_table_xarray,
            full_target_matrix=target_matrix,
            full_prediction_matrix=prediction_matrix,
            full_weight_matrix=weight_matrix,
            replicate_index=i,
            example_indices_in_replicate=these_indices,
            mean_training_target_values=mean_training_target_values,
            num_relia_bins_by_target=num_relia_bins_by_target,
            min_relia_bin_edge_by_target=min_relia_bin_edge_by_target,
            max_relia_bin_edge_by_target=max_relia_bin_edge_by_target,
            min_relia_bin_edge_prctile_by_target=
            min_relia_bin_edge_prctile_by_target,
            max_relia_bin_edge_prctile_by_target=
            max_relia_bin_edge_prctile_by_target,
            per_grid_cell=per_grid_cell,
            keep_it_simple=keep_it_simple
        )

    return result_table_xarray


def write_file(result_table_xarray, netcdf_file_name):
    """Writes evaluation results to NetCDF file.

    :param result_table_xarray: xarray table produced by
        `get_scores_with_bootstrapping`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    # result_table_xarray.to_netcdf(
    #     path=netcdf_file_name, mode='w', format='NETCDF3_64BIT_OFFSET'
    # )

    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_file(netcdf_file_name):
    """Reads evaluation results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table produced by
        `get_scores_with_bootstrapping`.
    """

    result_table_xarray = xarray.open_dataset(netcdf_file_name)
    result_table_xarray.attrs[PREDICTION_FILES_KEY] = (
        result_table_xarray.attrs[PREDICTION_FILES_KEY].split(' ')
    )

    if ISOTONIC_MODEL_FILE_KEY not in result_table_xarray.attrs:
        result_table_xarray.attrs[ISOTONIC_MODEL_FILE_KEY] = ''
    if UNCERTAINTY_CALIB_MODEL_FILE_KEY not in result_table_xarray.attrs:
        result_table_xarray.attrs[UNCERTAINTY_CALIB_MODEL_FILE_KEY] = ''

    if result_table_xarray.attrs[ISOTONIC_MODEL_FILE_KEY] == '':
        result_table_xarray.attrs[ISOTONIC_MODEL_FILE_KEY] = None
    if result_table_xarray.attrs[UNCERTAINTY_CALIB_MODEL_FILE_KEY] == '':
        result_table_xarray.attrs[UNCERTAINTY_CALIB_MODEL_FILE_KEY] = None

    return result_table_xarray
