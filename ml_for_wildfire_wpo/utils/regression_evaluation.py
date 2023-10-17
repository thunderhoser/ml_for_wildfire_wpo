"""Methods for evaluating a regression (not classification) model."""

import copy
import numpy
import xarray
from scipy.stats import ks_2samp
from gewittergefahr.gg_utils import histograms
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.io import prediction_io
from ml_for_wildfire_wpo.io import canadian_fwi_io
from ml_for_wildfire_wpo.utils import canadian_fwi_utils
from ml_for_wildfire_wpo.machine_learning import neural_net

# TODO(thunderhoser): Allow multiple target fields.
# TODO(thunderhoser): Allow multiple lead times.

RELIABILITY_BIN_DIM = 'reliability_bin'
BOOTSTRAP_REP_DIM = 'bootstrap_replicate'
DUMMY_DIM = 'dummy'

TARGET_STDEV_KEY = 'target_standard_deviation'
PREDICTION_STDEV_KEY = 'prediction_standard_deviation'

MSE_KEY = 'mean_squared_error'
MSE_BIAS_KEY = 'mse_bias'
MSE_VARIANCE_KEY = 'mse_variance'
MSE_SKILL_SCORE_KEY = 'mse_skill_score'
DWMSE_KEY = 'dual_weighted_mean_squared_error'
DWMSE_SKILL_SCORE_KEY = 'dwmse_skill_score'

KS_STATISTIC_KEY = 'kolmogorov_smirnov_statistic'
KS_P_VALUE_KEY = 'kolmogorov_smirnov_p_value'

MAE_KEY = 'mean_absolute_error'
MAE_SKILL_SCORE_KEY = 'mae_skill_score'
BIAS_KEY = 'bias'
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
PREDICTION_FILES_KEY = 'prediction_file_names'


def _get_mse_one_scalar(target_values, predicted_values):
    """Computes mean squared error (MSE) for one scalar target variable.

    E = number of examples

    :param target_values: length-E numpy array of target (actual) values.
    :param predicted_values: length-E numpy array of predicted values.
    :return: mse_total: Total MSE.
    :return: mse_bias: Bias component.
    :return: mse_variance: Variance component.
    """

    mse_total = numpy.mean((target_values - predicted_values) ** 2)
    mse_bias = numpy.mean(target_values - predicted_values) ** 2
    mse_variance = mse_total - mse_bias

    return mse_total, mse_bias, mse_variance


def _get_mse_ss_one_scalar(target_values, predicted_values,
                           mean_training_target_value):
    """Computes MSE skill score for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param mean_training_target_value: Mean target value over all training
        examples.
    :return: mse_skill_score: Self-explanatory.
    """

    mse_actual = _get_mse_one_scalar(
        target_values=target_values, predicted_values=predicted_values
    )[0]
    mse_climo = _get_mse_one_scalar(
        target_values=target_values, predicted_values=mean_training_target_value
    )[0]

    return (mse_climo - mse_actual) / mse_climo


def _get_dwmse_one_scalar(target_values, predicted_values):
    """Computes dual-weighted MSE (DWMSE) for one scalar target variable.

    E = number of examples

    :param target_values: length-E numpy array of target (actual) values.
    :param predicted_values: length-E numpy array of predicted values.
    :return: dwmse: Self-explanatory.
    """

    weights = numpy.maximum(
        numpy.absolute(target_values),
        numpy.absolute(predicted_values)
    )
    return numpy.mean(weights * (target_values - predicted_values) ** 2)


def _get_dwmse_ss_one_scalar(target_values, predicted_values,
                             mean_training_target_value):
    """Computes DWMSE skill score for one scalar target variable.

    :param target_values: See doc for `_get_dwmse_one_scalar`.
    :param predicted_values: Same.
    :param mean_training_target_value: Mean target value over all training
        examples.
    :return: dwmse_skill_score: Self-explanatory.
    """

    dwmse_actual = _get_dwmse_one_scalar(
        target_values=target_values, predicted_values=predicted_values
    )
    dwmse_climo = _get_dwmse_one_scalar(
        target_values=target_values,
        predicted_values=numpy.array([mean_training_target_value])
    )

    return (dwmse_climo - dwmse_actual) / dwmse_climo


def _get_mae_one_scalar(target_values, predicted_values):
    """Computes mean absolute error (MAE) for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :return: mean_absolute_error: Self-explanatory.
    """

    return numpy.mean(numpy.abs(target_values - predicted_values))


def _get_mae_ss_one_scalar(target_values, predicted_values,
                           mean_training_target_value):
    """Computes MAE skill score for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param mean_training_target_value: See doc for `_get_mse_ss_one_scalar`.
    :return: mae_skill_score: Self-explanatory.
    """

    mae_actual = _get_mae_one_scalar(
        target_values=target_values, predicted_values=predicted_values
    )
    mae_climo = _get_mae_one_scalar(
        target_values=target_values, predicted_values=mean_training_target_value
    )

    return (mae_climo - mae_actual) / mae_climo


def _get_bias_one_scalar(target_values, predicted_values):
    """Computes bias (mean signed error) for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :return: bias: Self-explanatory.
    """

    return numpy.mean(predicted_values - target_values)


def _get_correlation_one_scalar(target_values, predicted_values):
    """Computes Pearson correlation for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :return: correlation: Self-explanatory.
    """

    numerator = numpy.sum(
        (target_values - numpy.mean(target_values)) *
        (predicted_values - numpy.mean(predicted_values))
    )
    sum_squared_target_diffs = numpy.sum(
        (target_values - numpy.mean(target_values)) ** 2
    )
    sum_squared_prediction_diffs = numpy.sum(
        (predicted_values - numpy.mean(predicted_values)) ** 2
    )

    correlation = (
        numerator /
        numpy.sqrt(sum_squared_target_diffs * sum_squared_prediction_diffs)
    )

    return correlation


def _get_kge_one_scalar(target_values, predicted_values):
    """Computes KGE (Kling-Gupta efficiency) for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :return: kge: Self-explanatory.
    """

    correlation = _get_correlation_one_scalar(
        target_values=target_values, predicted_values=predicted_values
    )

    mean_target_value = numpy.mean(target_values)
    mean_predicted_value = numpy.mean(predicted_values)
    stdev_target_value = numpy.std(target_values, ddof=1)
    stdev_predicted_value = numpy.std(predicted_values, ddof=1)

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
        target_values, predicted_values, num_bins, min_bin_edge, max_bin_edge,
        invert=False):
    """Computes reliability curve for one scalar target variable.

    B = number of bins

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
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

    bin_index_by_example = histograms.create_histogram(
        input_values=target_values if invert else predicted_values,
        num_bins=num_bins, min_value=min_bin_edge, max_value=max_bin_edge
    )[0]

    mean_predictions = numpy.full(num_bins, numpy.nan)
    mean_observations = numpy.full(num_bins, numpy.nan)
    example_counts = numpy.full(num_bins, -1, dtype=int)

    for i in range(num_bins):
        these_example_indices = numpy.where(bin_index_by_example == i)[0]

        example_counts[i] = len(these_example_indices)
        mean_predictions[i] = numpy.mean(
            predicted_values[these_example_indices]
        )
        mean_observations[i] = numpy.mean(target_values[these_example_indices])

    return mean_predictions, mean_observations, example_counts


def _get_scores_one_replicate(
        result_table_xarray, full_target_matrix, full_prediction_matrix,
        replicate_index, example_indices_in_replicate,
        mean_training_target_value,
        min_reliability_bin_edge, max_reliability_bin_edge,
        min_reliability_bin_edge_percentile,
        max_reliability_bin_edge_percentile):
    """Computes scores for one bootstrap replicate.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid

    :param result_table_xarray: See doc for `get_scores_with_bootstrapping`.
    :param full_target_matrix: E-by-M-by-N numpy array of correct values.
    :param full_prediction_matrix: E-by-M-by-N numpy array of predicted values.
    :param replicate_index: Index of current bootstrap replicate.
    :param example_indices_in_replicate: 1-D numpy array with indices of
        examples in this bootstrap replicate.
    :param mean_training_target_value: Mean target value in training data (i.e.,
        "climatology").
    :param min_reliability_bin_edge: See doc for `get_scores_with_bootstrapping`.
    :param max_reliability_bin_edge: Same.
    :param min_reliability_bin_edge_percentile: Same.
    :param max_reliability_bin_edge_percentile: Same.
    :return: result_table_xarray: Same as input but with values filled for [i]th
        bootstrap replicate, where i = `replicate_index`.
    """

    t = result_table_xarray
    i = replicate_index + 0
    num_examples = len(example_indices_in_replicate)

    target_matrix = full_target_matrix[example_indices_in_replicate, ...]
    prediction_matrix = full_prediction_matrix[
        example_indices_in_replicate, ...
    ]

    t[TARGET_STDEV_KEY].values[i] = numpy.std(target_matrix, ddof=1)
    t[PREDICTION_STDEV_KEY].values[i] = numpy.std(prediction_matrix, ddof=1)
    t[MAE_KEY].values[i] = _get_mae_one_scalar(
        target_values=numpy.ravel(target_matrix),
        predicted_values=numpy.ravel(prediction_matrix)
    )
    t[MAE_SKILL_SCORE_KEY].values[i] = _get_mae_ss_one_scalar(
        target_values=numpy.ravel(target_matrix),
        predicted_values=numpy.ravel(prediction_matrix),
        mean_training_target_value=mean_training_target_value
    )
    (
        t[MSE_KEY].values[i],
        t[MSE_BIAS_KEY].values[i],
        t[MSE_VARIANCE_KEY].values[i]
    ) = _get_mse_one_scalar(
        target_values=numpy.ravel(target_matrix),
        predicted_values=numpy.ravel(prediction_matrix)
    )
    t[MSE_SKILL_SCORE_KEY].values[i] = _get_mse_ss_one_scalar(
        target_values=numpy.ravel(target_matrix),
        predicted_values=numpy.ravel(prediction_matrix),
        mean_training_target_value=mean_training_target_value
    )
    t[DWMSE_KEY].values[i] = _get_dwmse_one_scalar(
        target_values=numpy.ravel(target_matrix),
        predicted_values=numpy.ravel(prediction_matrix)
    )
    t[DWMSE_SKILL_SCORE_KEY].values[i] = _get_dwmse_ss_one_scalar(
        target_values=numpy.ravel(target_matrix),
        predicted_values=numpy.ravel(prediction_matrix),
        mean_training_target_value=mean_training_target_value
    )
    t[BIAS_KEY].values[i] = _get_bias_one_scalar(
        target_values=numpy.ravel(target_matrix),
        predicted_values=numpy.ravel(prediction_matrix)
    )
    t[CORRELATION_KEY].values[i] = _get_correlation_one_scalar(
        target_values=numpy.ravel(target_matrix),
        predicted_values=numpy.ravel(prediction_matrix)
    )
    t[KGE_KEY].values[i] = _get_kge_one_scalar(
        target_values=numpy.ravel(target_matrix),
        predicted_values=numpy.ravel(prediction_matrix)
    )

    if num_examples == 0:
        min_bin_edge = 0.
        max_bin_edge = 1.
    elif min_reliability_bin_edge is not None:
        min_bin_edge = min_reliability_bin_edge + 0.
        max_bin_edge = max_reliability_bin_edge + 0.
    else:
        min_bin_edge = numpy.percentile(
            prediction_matrix, min_reliability_bin_edge_percentile
        )
        max_bin_edge = numpy.percentile(
            prediction_matrix, max_reliability_bin_edge_percentile
        )

    (
        t[RELIABILITY_X_KEY].values[:, i],
        t[RELIABILITY_Y_KEY].values[:, i],
        these_counts
    ) = _get_rel_curve_one_scalar(
        target_values=numpy.ravel(target_matrix),
        predicted_values=numpy.ravel(prediction_matrix),
        num_bins=len(t.coords[RELIABILITY_BIN_DIM].values),
        min_bin_edge=min_bin_edge, max_bin_edge=max_bin_edge, invert=False
    )

    these_squared_diffs = (
        t[RELIABILITY_X_KEY].values[:, i] - t[RELIABILITY_Y_KEY].values[:, i]
    ) ** 2

    t[RELIABILITY_KEY].values[i] = (
        numpy.nansum(these_counts * these_squared_diffs) /
        numpy.sum(these_counts)
    )

    if i == 0:
        (
            t[RELIABILITY_BIN_CENTER_KEY].values[:], _,
            t[RELIABILITY_COUNT_KEY].values[:]
        ) = _get_rel_curve_one_scalar(
            target_values=numpy.ravel(full_target_matrix),
            predicted_values=numpy.ravel(full_prediction_matrix),
            num_bins=len(t.coords[RELIABILITY_BIN_DIM].values),
            min_bin_edge=min_bin_edge, max_bin_edge=max_bin_edge,
            invert=False
        )

        if full_target_matrix.size > 0:
            (
                t[KS_STATISTIC_KEY].values[0],
                t[KS_P_VALUE_KEY].values[0]
            ) = ks_2samp(
                numpy.ravel(full_target_matrix),
                numpy.ravel(full_prediction_matrix),
                alternative='two-sided', mode='auto'
            )

        (
            t[INV_RELIABILITY_BIN_CENTER_KEY].values[:], _,
            t[INV_RELIABILITY_COUNT_KEY].values[:]
        ) = _get_rel_curve_one_scalar(
            target_values=numpy.ravel(full_target_matrix),
            predicted_values=numpy.ravel(full_prediction_matrix),
            num_bins=len(t.coords[RELIABILITY_BIN_DIM].values),
            min_bin_edge=min_bin_edge, max_bin_edge=max_bin_edge,
            invert=True
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


def get_scores_with_bootstrapping(
        prediction_file_names, num_bootstrap_reps, num_reliability_bins,
        min_reliability_bin_edge, max_reliability_bin_edge,
        min_reliability_bin_edge_percentile,
        max_reliability_bin_edge_percentile):
    """Computes all scores with bootstrapping.

    :param prediction_file_names: 1-D list of paths to prediction files.  Each
        file will be read by `prediction_io.read_file`.
    :param num_bootstrap_reps: Number of bootstrap replicates.
    :param num_reliability_bins: Number of bins for reliability curve.
    :param min_reliability_bin_edge: Minimum target/predicted value for
        reliability curves.  If you instead want the minimum value to be a
        percentile over the data, make this argument None and use
        `min_reliability_bin_edge_percentile` instead.
    :param max_reliability_bin_edge: Same but for maximum.
    :param min_reliability_bin_edge_percentile: Determines minimum
        target/predicted value for reliability curves.  This percentile must
        range from 0...100.
    :param max_reliability_bin_edge_percentile: Same but for maximum.
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    # TODO(thunderhoser): When I start training ensemble models, I will need to
    # take the mean here.

    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_integer(num_bootstrap_reps)
    error_checking.assert_is_greater(num_bootstrap_reps, 0)
    error_checking.assert_is_integer(num_reliability_bins)
    error_checking.assert_is_geq(num_reliability_bins, 10)
    error_checking.assert_is_leq(num_reliability_bins, 1000)

    if min_reliability_bin_edge is None or max_reliability_bin_edge is None:
        error_checking.assert_is_leq(min_reliability_bin_edge_percentile, 10.)
        error_checking.assert_is_geq(max_reliability_bin_edge_percentile, 90.)
    else:
        error_checking.assert_is_greater(
            max_reliability_bin_edge, min_reliability_bin_edge
        )

    num_times = len(prediction_file_names)
    target_matrix = numpy.array([], dtype=float)
    prediction_matrix = numpy.array([], dtype=float)
    weight_matrix = numpy.array([], dtype=float)
    model_file_name = None

    for i in range(num_times):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        this_prediction_table_xarray = prediction_io.read_file(
            prediction_file_names[i]
        )
        tpt = this_prediction_table_xarray

        if model_file_name is None:
            these_dim = (
                (num_times,) + tpt[prediction_io.TARGET_KEY].values.shape
            )
            target_matrix = numpy.full(these_dim, numpy.nan)
            prediction_matrix = numpy.full(these_dim, numpy.nan)
            weight_matrix = numpy.full(these_dim, numpy.nan)

            model_file_name = copy.deepcopy(
                tpt.attrs[prediction_io.MODEL_FILE_KEY]
            )

        target_matrix[i, ...] = tpt[prediction_io.TARGET_KEY].values
        prediction_matrix[i, ...] = tpt[prediction_io.PREDICTION_KEY].values
        weight_matrix[i, ...] = tpt[prediction_io.WEIGHT_KEY].values
        assert model_file_name == tpt.attrs[prediction_io.MODEL_FILE_KEY]

    # TODO(thunderhoser): This is a HACK.  I should use the weight matrix to
    # actually weight the various scores.
    target_matrix[weight_matrix < 0.05] = 0.
    prediction_matrix[weight_matrix < 0.05] = 0.

    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    target_field_name = generator_option_dict[neural_net.TARGET_FIELD_KEY]
    target_norm_file_name = generator_option_dict[
        neural_net.TARGET_NORM_FILE_KEY
    ]

    # TODO(thunderhoser): This bit will not work if, for some reason, I do not
    # normalized the lagged-target predictors.
    print('Reading climo-mean {0:s} value from: "{1:s}"...'.format(
        target_field_name, target_norm_file_name
    ))
    target_norm_param_table_xarray = canadian_fwi_io.read_normalization_file(
        target_norm_file_name
    )
    tnpt = target_norm_param_table_xarray

    field_index = numpy.where(
        tnpt.coords[canadian_fwi_utils.FIELD_DIM].values == target_field_name
    )[0][0]
    mean_training_target_value = (
        tnpt[canadian_fwi_utils.MEAN_VALUE_KEY].values[field_index]
    )
    print('Climo-mean {0:s} = {1:.4f}'.format(
        target_field_name, mean_training_target_value
    ))

    these_dimensions = (num_bootstrap_reps,)
    these_dim_keys = (BOOTSTRAP_REP_DIM,)
    main_data_dict = {
        TARGET_STDEV_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        PREDICTION_STDEV_KEY: (
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

    these_dimensions = (num_reliability_bins, num_bootstrap_reps)
    these_dim_keys = (RELIABILITY_BIN_DIM, BOOTSTRAP_REP_DIM)
    new_dict = {
        RELIABILITY_X_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        RELIABILITY_Y_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    these_dimensions = (num_reliability_bins,)
    these_dim_keys = (RELIABILITY_BIN_DIM,)
    new_dict = {
        RELIABILITY_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        RELIABILITY_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
        ),
        INV_RELIABILITY_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        INV_RELIABILITY_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
        )
    }
    main_data_dict.update(new_dict)

    these_dimensions = (1,)
    these_dim_keys = (DUMMY_DIM,)
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
        0, num_reliability_bins - 1, num=num_reliability_bins, dtype=int
    )
    bootstrap_indices = numpy.linspace(
        0, num_bootstrap_reps - 1, num=num_bootstrap_reps, dtype=int
    )
    metadata_dict = {
        RELIABILITY_BIN_DIM: reliability_bin_indices,
        BOOTSTRAP_REP_DIM: bootstrap_indices
    }

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
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
            replicate_index=i,
            example_indices_in_replicate=these_indices,
            mean_training_target_value=mean_training_target_value,
            min_reliability_bin_edge=min_reliability_bin_edge,
            max_reliability_bin_edge=max_reliability_bin_edge,
            min_reliability_bin_edge_percentile=
            min_reliability_bin_edge_percentile,
            max_reliability_bin_edge_percentile=
            max_reliability_bin_edge_percentile
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

    return result_table_xarray
