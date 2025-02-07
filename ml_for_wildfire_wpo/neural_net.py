"""Helper methods for training a neural network to predict fire weather."""

import os
import sys
import time
import random
import pickle
import warnings
import numpy
import pandas
import keras
from scipy.interpolate import interp1d

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import number_rounding
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import gfs_io
import gfs_daily_io
import era5_constant_io
import canadian_fwi_io
import misc_utils
import gfs_utils
import gfs_daily_utils
import era5_constant_utils
import canadian_fwi_utils
import normalization
import custom_losses
import custom_metrics

try:
    from tensorflow.keras.saving import load_model
except:
    pass

TOLERANCE = 1e-6
DATE_FORMAT = '%Y%m%d'

GRID_SPACING_DEG = 0.25
MASK_PIXEL_IF_WEIGHT_BELOW = 0.05

DAYS_TO_HOURS = 24
DAYS_TO_SECONDS = 86400
DEGREES_TO_RADIANS = numpy.pi / 180.

INNER_LATITUDE_LIMITS_KEY = 'inner_latitude_limits_deg_n'
INNER_LONGITUDE_LIMITS_KEY = 'inner_longitude_limits_deg_e'
OUTER_LATITUDE_BUFFER_KEY = 'outer_latitude_buffer_deg'
OUTER_LONGITUDE_BUFFER_KEY = 'outer_longitude_buffer_deg'
INIT_DATE_LIMITS_KEY = 'init_date_limit_strings'
GFS_PREDICTOR_FIELDS_KEY = 'gfs_predictor_field_names'
GFS_PRESSURE_LEVELS_KEY = 'gfs_pressure_levels_mb'
MODEL_LEAD_TO_GFS_PRED_LEADS_KEY = 'model_lead_days_to_gfs_pred_leads_hours'
GFS_DIRECTORY_KEY = 'gfs_directory_name'
GFS_NORM_FILE_KEY = 'gfs_normalization_file_name'
GFS_USE_QUANTILE_NORM_KEY = 'gfs_use_quantile_norm'
ERA5_CONSTANT_PREDICTOR_FIELDS_KEY = 'era5_constant_predictor_field_names'
ERA5_CONSTANT_FILE_KEY = 'era5_constant_file_name'
ERA5_NORM_FILE_KEY = 'era5_normalization_file_name'
ERA5_USE_QUANTILE_NORM_KEY = 'era5_use_quantile_norm'
TARGET_FIELDS_KEY = 'target_field_names'
MODEL_LEAD_TO_TARGET_LAGS_KEY = 'model_lead_days_to_target_lags_days'
MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY = 'model_lead_days_to_gfs_target_leads_days'
MODEL_LEAD_TO_FREQ_KEY = 'model_lead_days_to_freq'
COMPARE_TO_GFS_IN_LOSS_KEY = 'compare_to_gfs_in_loss'
TARGET_DIRECTORY_KEY = 'target_dir_name'
GFS_FORECAST_TARGET_DIR_KEY = 'gfs_forecast_target_dir_name'
TARGET_NORM_FILE_KEY = 'target_normalization_file_name'
TARGETS_USE_QUANTILE_NORM_KEY = 'targets_use_quantile_norm'
BATCH_SIZE_KEY = 'num_examples_per_batch'
SENTINEL_VALUE_KEY = 'sentinel_value'
DO_RESIDUAL_PREDICTION_KEY = 'do_residual_prediction'
USE_LEAD_TIME_AS_PRED_KEY = 'use_lead_time_as_predictor'
CHANGE_LEAD_EVERY_N_BATCHES_KEY = 'change_model_lead_every_n_batches'
OUTER_PATCH_SIZE_DEG_KEY = 'outer_patch_size_deg'
OUTER_PATCH_OVERLAP_DEG_KEY = 'outer_patch_overlap_deg'

DEFAULT_GENERATOR_OPTION_DICT = {
    INNER_LATITUDE_LIMITS_KEY: numpy.array([17, 73], dtype=float),
    INNER_LONGITUDE_LIMITS_KEY: numpy.array([171, -65], dtype=float),
    OUTER_LATITUDE_BUFFER_KEY: 5.,
    OUTER_LONGITUDE_BUFFER_KEY: 5.,
    # SENTINEL_VALUE_KEY: -10.
}

NUM_EPOCHS_KEY = 'num_epochs'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
TRAINING_OPTIONS_KEY = 'training_option_dict'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
VALIDATION_OPTIONS_KEY = 'validation_option_dict'
LOSS_FUNCTION_KEY = 'loss_function_string'
METRIC_FUNCTIONS_KEY = 'metric_function_strings'
OPTIMIZER_FUNCTION_KEY = 'optimizer_function_string'
CHIU_NET_ARCHITECTURE_KEY = 'chiu_net_architecture_dict'
CHIU_NET_PP_ARCHITECTURE_KEY = 'chiu_net_pp_architecture_dict'
PLATEAU_PATIENCE_KEY = 'plateau_patience_epochs'
PLATEAU_LR_MUTIPLIER_KEY = 'plateau_learning_rate_multiplier'
EARLY_STOPPING_PATIENCE_KEY = 'early_stopping_patience_epochs'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY,
    LOSS_FUNCTION_KEY, METRIC_FUNCTIONS_KEY, OPTIMIZER_FUNCTION_KEY,
    CHIU_NET_ARCHITECTURE_KEY, CHIU_NET_PP_ARCHITECTURE_KEY,
    PLATEAU_PATIENCE_KEY, PLATEAU_LR_MUTIPLIER_KEY, EARLY_STOPPING_PATIENCE_KEY
]

NUM_FULL_ROWS_KEY = 'num_rows_in_full_grid'
NUM_FULL_COLUMNS_KEY = 'num_columns_in_full_grid'
NUM_PATCH_ROWS_KEY = 'num_rows_in_patch'
NUM_PATCH_COLUMNS_KEY = 'num_columns_in_patch'
PATCH_OVERLAP_SIZE_PX_KEY = 'patch_overlap_size_pixels'
PATCH_START_ROW_KEY = 'patch_start_row'
PATCH_START_COLUMN_KEY = 'patch_start_column'

PREDICTOR_MATRICES_KEY = 'predictor_matrices'
TARGETS_AND_WEIGHTS_KEY = 'target_matrix_with_weights'
GRID_LATITUDE_MATRIX_KEY = 'grid_latitudes_deg_n'
GRID_LONGITUDE_MATRIX_KEY = 'grid_longitudes_deg_e'
INPUT_LAYER_NAMES_KEY = 'input_layer_names'

GFS_3D_LAYER_NAME = 'gfs_3d_inputs'
GFS_2D_LAYER_NAME = 'gfs_2d_inputs'
LEAD_TIME_LAYER_NAME = 'lead_time'
LAGLEAD_TARGET_LAYER_NAME = 'lagged_target_inputs'
ERA5_LAYER_NAME = 'era5_inputs'
PREDN_BASELINE_LAYER_NAME = 'predn_baseline_inputs'

VALID_INPUT_LAYER_NAMES = [
    GFS_3D_LAYER_NAME, GFS_2D_LAYER_NAME, LEAD_TIME_LAYER_NAME,
    LAGLEAD_TARGET_LAYER_NAME, ERA5_LAYER_NAME, PREDN_BASELINE_LAYER_NAME
]

PREDICTOR_MATRIX_3D_GFS_KEY = 'gfs_predictor_matrix_3d'
PREDICTOR_MATRIX_2D_GFS_KEY = 'gfs_predictor_matrix_2d'
PREDICTOR_MATRIX_BASELINE_KEY = 'baseline_prediction_matrix'
PREDICTOR_MATRIX_LAGLEAD_KEY = 'laglead_target_predictor_matrix'
PREDICTOR_MATRIX_ERA5_KEY = 'era5_constant_matrix'
TARGET_MATRIX_WITH_WEIGHTS_KEY = 'target_matrix_with_weights'


def __report_data_properties(
        gfs_predictor_matrix_3d, gfs_predictor_matrix_2d,
        lead_time_predictors_days, era5_constant_matrix,
        laglead_target_predictor_matrix, baseline_prediction_matrix,
        target_matrix_with_weights, sentinel_value):
    """Reports data properties at end of data-generator or data-creator.

    :param gfs_predictor_matrix_3d: See output doc for `data_generator`.
    :param gfs_predictor_matrix_2d: Same.
    :param lead_time_predictors_days: Same.
    :param era5_constant_matrix: Same.
    :param laglead_target_predictor_matrix: Same.
    :param baseline_prediction_matrix: Same.
    :param target_matrix_with_weights: Same.
    :param sentinel_value: Same.
    :return: predictor_matrices: Tuple with 32-bit predictor matrices.
    :return: input_layer_names: 1-D list with same length as
        "predictor_matrices", indicating the input layer for every predictor
        matrix.
    """

    error_checking.assert_is_numpy_array_without_nan(target_matrix_with_weights)

    print((
        'Shape of target matrix with weights in last channel = {0:s} ... '
        'NaN fraction = {1:.4f} ... min/max = {2:.4f}/{3:.4f}'
    ).format(
        str(target_matrix_with_weights.shape),
        numpy.mean(numpy.isnan(target_matrix_with_weights)),
        numpy.min(target_matrix_with_weights),
        numpy.max(target_matrix_with_weights)
    ))

    predictor_matrices = (
        gfs_predictor_matrix_3d, gfs_predictor_matrix_2d,
        lead_time_predictors_days, era5_constant_matrix,
        laglead_target_predictor_matrix, baseline_prediction_matrix
    )
    pred_matrix_descriptions = [
        '3-D GFS predictor matrix', '2-D GFS predictor matrix',
        'lead-time predictor matrix', 'ERA5-constant predictor matrix',
        'lag/lead-target predictor matrix', 'residual baseline matrix'
    ]
    input_layer_names = [
        GFS_3D_LAYER_NAME, GFS_2D_LAYER_NAME,
        LEAD_TIME_LAYER_NAME, ERA5_LAYER_NAME,
        LAGLEAD_TARGET_LAYER_NAME, PREDN_BASELINE_LAYER_NAME
    ]
    allow_nan_flags = [True, True, False, False, True, True]

    for k in range(len(predictor_matrices)):
        if predictor_matrices[k] is None:
            continue

        print((
            'Shape of {0:s}: {1:s} ... NaN fraction = {2:.4f} ... '
            'min/max = {3:.4f}/{4:.4f}'
        ).format(
            pred_matrix_descriptions[k],
            str(predictor_matrices[k].shape),
            numpy.mean(numpy.isnan(predictor_matrices[k])),
            numpy.nanmin(predictor_matrices[k]),
            numpy.nanmax(predictor_matrices[k])
        ))

        if allow_nan_flags[k]:
            predictor_matrices[k][numpy.isnan(predictor_matrices[k])] = (
                sentinel_value
            )
        else:
            error_checking.assert_is_numpy_array_without_nan(
                predictor_matrices[k]
            )

    if baseline_prediction_matrix is not None:
        these_min = numpy.nanmin(
            baseline_prediction_matrix, axis=(0, 1, 2)
        )
        these_max = numpy.nanmax(
            baseline_prediction_matrix, axis=(0, 1, 2)
        )

        print('Min values in residual baseline matrix: {0:s}'.format(
            str(these_min)
        ))
        print('Max values in residual baseline matrix: {0:s}'.format(
            str(these_max)
        ))

    good_flags = numpy.array(
        [pm is not None for pm in predictor_matrices], dtype=bool
    )
    good_indices = numpy.where(good_flags)[0]

    predictor_matrices = tuple([predictor_matrices[k] for k in good_indices])
    input_layer_names = tuple([input_layer_names[k] for k in good_indices])
    return predictor_matrices, input_layer_names


def __determine_num_times_for_interp(generator_option_dict):
    """For both GFS Wx variables and lag/lead targets, computes # interp times.

    :param generator_option_dict: See documentation for `data_generator`.
    :return: num_gfs_hours_for_interp: Number of time steps to which GFS Wx
        variables will be interpolated.  If no interpolation is needed (i.e.,
        all model lead times require the same number of GFS Wx lead times), this
        will be None.
    :return: num_target_times_for_interp: Number of time steps to which lag/lead
        targets will be interpolated.  If no interpolation is needed (i.e.,
        all model lead times require the same number of lag/lead target times),
        this will be None.
    """

    model_lead_days_to_gfs_pred_leads_hours = generator_option_dict[
        MODEL_LEAD_TO_GFS_PRED_LEADS_KEY
    ]
    model_lead_days_to_target_lags_days = generator_option_dict[
        MODEL_LEAD_TO_TARGET_LAGS_KEY
    ]
    model_lead_days_to_gfs_target_leads_days = generator_option_dict[
        MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY
    ]
    model_lead_times_days = numpy.array(
        list(model_lead_days_to_gfs_pred_leads_hours.keys()),
        dtype=int
    )

    num_gfs_hours_by_model_lead = numpy.array([
        len(model_lead_days_to_gfs_pred_leads_hours[l])
        for l in model_lead_times_days
    ], dtype=int)

    if len(numpy.unique(num_gfs_hours_by_model_lead)) == 1:
        num_gfs_hours_for_interp = None
    else:
        num_gfs_hours_for_interp = numpy.max(num_gfs_hours_by_model_lead)

    if model_lead_days_to_target_lags_days is None:
        num_target_lags_by_model_lead = numpy.full(
            len(model_lead_times_days), 0, dtype=int
        )
    else:
        num_target_lags_by_model_lead = numpy.array([
            len(model_lead_days_to_target_lags_days[l])
            for l in model_lead_times_days
        ], dtype=int)

    if model_lead_days_to_gfs_target_leads_days is None:
        num_gfs_target_leads_by_model_lead = numpy.full(
            len(model_lead_times_days), 0, dtype=int
        )
    else:
        num_gfs_target_leads_by_model_lead = numpy.array([
            len(model_lead_days_to_gfs_target_leads_days[l])
            for l in model_lead_times_days
        ], dtype=int)

    num_target_time_steps_by_model_lead = (
        num_target_lags_by_model_lead + num_gfs_target_leads_by_model_lead
    )

    if len(numpy.unique(num_target_time_steps_by_model_lead)) == 1:
        num_target_times_for_interp = None
    else:
        num_target_times_for_interp = numpy.max(
            num_target_time_steps_by_model_lead
        )

    return num_gfs_hours_for_interp, num_target_times_for_interp


def __make_trapezoidal_weight_matrix(patch_size_pixels,
                                     patch_overlap_size_pixels):
    """Creates trapezoidal weight matrix for applying patchwise NN to full grid.

    :param patch_size_pixels: See doc for `__update_patch_metalocation_dict`.
    :param patch_overlap_size_pixels: Same.
    :return: trapezoidal_weight_matrix: M-by-M numpy array of weights, where
        M = patch size.
    """

    middle_length = patch_size_pixels - patch_overlap_size_pixels
    middle_start_index = (patch_size_pixels - middle_length) // 2

    weights_before_plateau = numpy.linspace(
        0, 1,
        num=middle_start_index, endpoint=False, dtype=float
    )
    weights_before_plateau = numpy.linspace(
        weights_before_plateau[1], 1,
        num=middle_start_index, endpoint=False, dtype=float
    )
    trapezoidal_weights = numpy.concatenate([
        weights_before_plateau,
        numpy.full(middle_length, 1.),
        weights_before_plateau[::-1]
    ])

    first_weight_matrix, second_weight_matrix = numpy.meshgrid(
        trapezoidal_weights, trapezoidal_weights
    )
    return first_weight_matrix * second_weight_matrix


def __init_patch_metalocation_dict(
        num_rows_in_full_grid, num_columns_in_full_grid,
        patch_size_pixels, patch_overlap_size_pixels):
    """Initializes patch-metalocation dictionary.

    To understand what the "patch-metalocation dictionary" is, see documentation
    for `__update_patch_metalocation_dict`.

    :param num_rows_in_full_grid: Number of rows in full grid.
    :param num_columns_in_full_grid: Number of columns in full grid.
    :param patch_size_pixels: See doc for `__update_patch_metalocation_dict`.
    :param patch_overlap_size_pixels: Same.
    :return: patch_metalocation_dict: Same.
    """

    return {
        NUM_FULL_ROWS_KEY: num_rows_in_full_grid,
        NUM_FULL_COLUMNS_KEY: num_columns_in_full_grid,
        NUM_PATCH_ROWS_KEY: patch_size_pixels,
        NUM_PATCH_COLUMNS_KEY: patch_size_pixels,
        PATCH_OVERLAP_SIZE_PX_KEY: patch_overlap_size_pixels,
        PATCH_START_ROW_KEY: -1,
        PATCH_START_COLUMN_KEY: -1
    }


def __update_patch_metalocation_dict(patch_metalocation_dict):
    """Updates patch-metalocation dictionary.

    This is fancy talk for "determines where the next patch will be, when
    applying a patchwise-trained neural net over the full grid".

    :param patch_metalocation_dict: Dictionary with the following keys.
    patch_metalocation_dict["num_rows_in_full_grid"]: Number of rows in full
        grid.
    patch_metalocation_dict["num_columns_in_full_grid"]: Number of columns in
        full grid.
    patch_metalocation_dict["num_rows_in_patch"]: Number of rows in each patch.
    patch_metalocation_dict["num_columns_in_patch"]: Number of columns in each
        patch.
    patch_metalocation_dict["patch_overlap_size_pixels"]: Overlap between
        adjacent patches.
    patch_metalocation_dict["patch_start_row"]: First row covered by the
        current patch location.
    patch_metalocation_dict["patch_start_column"]: First column covered
        by the current patch location.

    :return: patch_metalocation_dict: Same as input, except that keys
        "patch_start_row" and "patch_start_column" have been updated.
    """

    pmld = patch_metalocation_dict
    num_rows_in_full_grid = pmld[NUM_FULL_ROWS_KEY]
    num_columns_in_full_grid = pmld[NUM_FULL_COLUMNS_KEY]
    num_rows_in_patch = pmld[NUM_PATCH_ROWS_KEY]
    num_columns_in_patch = pmld[NUM_PATCH_COLUMNS_KEY]
    patch_overlap_size_pixels = pmld[PATCH_OVERLAP_SIZE_PX_KEY]
    patch_end_row = pmld[PATCH_START_ROW_KEY] + num_rows_in_patch - 1
    patch_end_column = pmld[PATCH_START_COLUMN_KEY] + num_columns_in_patch - 1

    if pmld[PATCH_START_ROW_KEY] < 0:
        patch_end_row = num_rows_in_patch - 1
        patch_end_column = num_columns_in_patch - 1
    elif patch_end_column >= num_columns_in_full_grid - 1:
        if patch_end_row >= num_rows_in_full_grid - 1:
            patch_end_row = -1
            patch_end_column = -1
        else:
            patch_end_row += num_rows_in_patch - 2 * patch_overlap_size_pixels
            patch_end_column = num_columns_in_patch - 1
    else:
        patch_end_column += num_columns_in_patch - 2 * patch_overlap_size_pixels

    patch_end_row = min([
        patch_end_row, num_rows_in_full_grid - 1
    ])
    patch_end_column = min([
        patch_end_column, num_columns_in_full_grid - 1
    ])

    patch_start_row = patch_end_row - num_rows_in_patch + 1
    patch_start_column = patch_end_column - num_columns_in_patch + 1

    pmld[PATCH_START_ROW_KEY] = patch_start_row
    pmld[PATCH_START_COLUMN_KEY] = patch_start_column
    patch_metalocation_dict = pmld

    return patch_metalocation_dict


def __init_matrices_1batch_patchwise(generator_option_dict, gfs_file_names):
    """Inits predictor and target matrices for one batch of patchwise training.

    :param generator_option_dict: See documentation for `option_dict` in
        `data_generator_fast_patches`.
    :param gfs_file_names: 1-D list of paths to files with GFS forecasts.
    :return: matrix_dict: Dictionary with the following keys.
    matrix_dict["gfs_predictor_matrix_3d"]: numpy array for GFS data with three
        spatial dimensions.  If not using 3-D GFS data, this is None instead of
        an array.
    matrix_dict["gfs_predictor_matrix_2d"]: numpy array for GFS data with two
        spatial dimensions.  If not using 2-D GFS data, this is None instead of
        an array.
    matrix_dict["baseline_prediction_matrix"]: numpy array for residual baseline
        predictions, i.e., FWI predictions from GFS model.  If not using
        residual baseline, this is None instead of an array.
    matrix_dict["laglead_target_predictor_matrix"]: numpy array for lag/lead
        target fields used as predictors.
    matrix_dict["era5_constant_matrix"]: numpy array for ERA5 time-invariant
        fields used as predictors.  If not using such fields, this is None
        instead of an array.
    matrix_dict["target_matrix_with_weights"]: numpy array for target fields,
        with evaluation weights as the last channel.
    """

    option_dict = generator_option_dict

    inner_latitude_limits_deg_n = option_dict[INNER_LATITUDE_LIMITS_KEY]
    inner_longitude_limits_deg_e = option_dict[INNER_LONGITUDE_LIMITS_KEY]
    outer_latitude_buffer_deg = option_dict[OUTER_LATITUDE_BUFFER_KEY]
    outer_longitude_buffer_deg = option_dict[OUTER_LONGITUDE_BUFFER_KEY]
    gfs_predictor_field_names = option_dict[GFS_PREDICTOR_FIELDS_KEY]
    gfs_pressure_levels_mb = option_dict[GFS_PRESSURE_LEVELS_KEY]
    model_lead_days_to_gfs_pred_leads_hours = option_dict[
        MODEL_LEAD_TO_GFS_PRED_LEADS_KEY
    ]
    era5_constant_predictor_field_names = option_dict[
        ERA5_CONSTANT_PREDICTOR_FIELDS_KEY
    ]
    era5_constant_file_name = option_dict[ERA5_CONSTANT_FILE_KEY]
    target_field_names = option_dict[TARGET_FIELDS_KEY]
    model_lead_days_to_target_lags_days = option_dict[
        MODEL_LEAD_TO_TARGET_LAGS_KEY
    ]
    model_lead_days_to_gfs_target_leads_days = option_dict[
        MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY
    ]
    model_lead_days_to_freq = option_dict[MODEL_LEAD_TO_FREQ_KEY]
    compare_to_gfs_in_loss = option_dict[COMPARE_TO_GFS_IN_LOSS_KEY]
    target_dir_name = option_dict[TARGET_DIRECTORY_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    do_residual_prediction = option_dict[DO_RESIDUAL_PREDICTION_KEY]
    patch_size_deg = option_dict[OUTER_PATCH_SIZE_DEG_KEY]

    assert patch_size_deg is not None

    patch_size_pixels = int(numpy.round(
        float(patch_size_deg) / GRID_SPACING_DEG
    ))
    model_lead_times_days = numpy.array(
        list(model_lead_days_to_freq.keys()),
        dtype=int
    )
    num_gfs_hours_for_interp, num_target_times_for_interp = (
        __determine_num_times_for_interp(option_dict)
    )

    # TODO(thunderhoser): The longitude command below might fail.
    outer_latitude_limits_deg_n = inner_latitude_limits_deg_n + numpy.array([
        -1 * outer_latitude_buffer_deg, outer_latitude_buffer_deg
    ])
    outer_longitude_limits_deg_e = inner_longitude_limits_deg_e + numpy.array([
        -1 * outer_longitude_buffer_deg, outer_longitude_buffer_deg
    ])

    this_3d_matrix, this_2d_matrix = _read_gfs_data_1example(
        gfs_file_name=gfs_file_names[0],
        desired_row_indices=None,
        desired_column_indices=None,
        latitude_limits_deg_n=outer_latitude_limits_deg_n,
        longitude_limits_deg_e=outer_longitude_limits_deg_e,
        lead_times_hours=
        model_lead_days_to_gfs_pred_leads_hours[model_lead_times_days[0]],
        field_names=gfs_predictor_field_names,
        pressure_levels_mb=gfs_pressure_levels_mb,
        norm_param_table_xarray=None,
        use_quantile_norm=False,
        num_lead_times_for_interp=num_gfs_hours_for_interp
    )[:2]

    bs = num_examples_per_batch
    psp = patch_size_pixels

    if this_3d_matrix is None:
        gfs_predictor_matrix_3d = None
    else:
        these_dim = (bs, psp, psp) + this_3d_matrix.shape[2:]
        gfs_predictor_matrix_3d = numpy.full(these_dim, numpy.nan)

    if this_2d_matrix is None:
        gfs_predictor_matrix_2d = None
    else:
        these_dim = (bs, psp, psp) + this_2d_matrix.shape[2:]
        gfs_predictor_matrix_2d = numpy.full(these_dim, numpy.nan)

    if model_lead_days_to_target_lags_days is None:
        target_lag_times_days = numpy.array([], dtype=int)
    else:
        target_lag_times_days = model_lead_days_to_target_lags_days[
            model_lead_times_days[0]
        ]

    if model_lead_days_to_gfs_target_leads_days is None:
        gfs_target_lead_times_days = numpy.array([], dtype=int)
    else:
        gfs_target_lead_times_days = model_lead_days_to_gfs_target_leads_days[
            model_lead_times_days[0]
        ]

    if do_residual_prediction:
        this_matrix = _read_lagged_targets_1example(
            gfs_init_date_string=gfs_io.file_name_to_date(gfs_file_names[0]),
            target_dir_name=target_dir_name,
            target_lag_times_days=numpy.array(
                [numpy.min(target_lag_times_days)]
            ),
            desired_row_indices=None,
            desired_column_indices=None,
            latitude_limits_deg_n=inner_latitude_limits_deg_n,
            longitude_limits_deg_e=inner_longitude_limits_deg_e,
            target_field_names=target_field_names,
            norm_param_table_xarray=None,
            use_quantile_norm=False
        )[0]

        these_dim = (bs, psp, psp) + this_matrix.shape[3:]
        baseline_prediction_matrix = numpy.full(these_dim, numpy.nan)
    else:
        baseline_prediction_matrix = None

    this_matrix = _read_lagged_targets_1example(
        gfs_init_date_string=gfs_io.file_name_to_date(gfs_file_names[0]),
        target_dir_name=target_dir_name,
        target_lag_times_days=target_lag_times_days,
        desired_row_indices=None,
        desired_column_indices=None,
        latitude_limits_deg_n=inner_latitude_limits_deg_n,
        longitude_limits_deg_e=inner_longitude_limits_deg_e,
        target_field_names=target_field_names,
        norm_param_table_xarray=None,
        use_quantile_norm=False
    )[0]

    if num_target_times_for_interp is None:
        this_num_times = (
            len(target_lag_times_days) + len(gfs_target_lead_times_days)
        )
    else:
        this_num_times = num_target_times_for_interp + 0

    these_dim = (bs, psp, psp, this_num_times) + this_matrix.shape[3:]
    laglead_target_predictor_matrix = numpy.full(these_dim, numpy.nan)

    if era5_constant_predictor_field_names is None:
        era5_constant_matrix = None
    else:
        this_matrix = _get_era5_constants(
            era5_constant_file_name=era5_constant_file_name,
            latitude_limits_deg_n=outer_latitude_limits_deg_n,
            longitude_limits_deg_e=outer_longitude_limits_deg_e,
            field_names=era5_constant_predictor_field_names,
            norm_param_table_xarray=None,
            use_quantile_norm=False
        )

        these_dim = (bs, psp, psp) + this_matrix.shape[2:]
        era5_constant_matrix = numpy.full(these_dim, numpy.nan)

    num_target_channels = 1 + (
        (int(compare_to_gfs_in_loss) + 1) * len(target_field_names)
    )
    these_dim = (bs, psp, psp, num_target_channels)
    target_matrix_with_weights = numpy.full(these_dim, numpy.nan)

    return {
        PREDICTOR_MATRIX_3D_GFS_KEY: gfs_predictor_matrix_3d,
        PREDICTOR_MATRIX_2D_GFS_KEY: gfs_predictor_matrix_2d,
        PREDICTOR_MATRIX_BASELINE_KEY: baseline_prediction_matrix,
        PREDICTOR_MATRIX_LAGLEAD_KEY: laglead_target_predictor_matrix,
        PREDICTOR_MATRIX_ERA5_KEY: era5_constant_matrix,
        TARGET_MATRIX_WITH_WEIGHTS_KEY: target_matrix_with_weights
    }


def __increment_init_time(current_index, gfs_file_names):
    """Increments initialization time for generator.

    This allows the generator to read the next init time.

    :param current_index: Current index.  If current_index == k, this means the
        last file read is gfs_file_names[k].
    :param gfs_file_names: 1-D list of paths to GFS files, one per init time.
    :return: current_index: Updated version of input.
    :return: gfs_file_names: Possibly shuffled version of input.
    """

    if current_index == len(gfs_file_names) - 1:
        random.shuffle(gfs_file_names)
        current_index = 0
    else:
        current_index += 1

    return current_index, gfs_file_names


def _check_generator_args(option_dict):
    """Checks input arguments for generator.

    :param option_dict: See doc for `data_generator`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_numpy_array(
        option_dict[INNER_LATITUDE_LIMITS_KEY],
        exact_dimensions=numpy.array([2], dtype=int)
    )
    error_checking.assert_is_valid_lat_numpy_array(
        option_dict[INNER_LATITUDE_LIMITS_KEY], allow_nan=False
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(option_dict[INNER_LATITUDE_LIMITS_KEY]),
        0
    )

    error_checking.assert_is_numpy_array(
        option_dict[INNER_LONGITUDE_LIMITS_KEY],
        exact_dimensions=numpy.array([2], dtype=int)
    )
    error_checking.assert_is_valid_lng_numpy_array(
        option_dict[INNER_LONGITUDE_LIMITS_KEY],
        positive_in_west_flag=False, negative_in_west_flag=False,
        allow_nan=False
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.absolute(numpy.diff(option_dict[INNER_LONGITUDE_LIMITS_KEY])),
        0
    )

    error_checking.assert_is_greater(option_dict[OUTER_LATITUDE_BUFFER_KEY], 0)
    option_dict[OUTER_LATITUDE_BUFFER_KEY] = number_rounding.round_to_nearest(
        option_dict[OUTER_LATITUDE_BUFFER_KEY], GRID_SPACING_DEG
    )

    outer_latitude_limits_deg_n = option_dict[INNER_LATITUDE_LIMITS_KEY] + (
        numpy.array([
            -1 * option_dict[OUTER_LATITUDE_BUFFER_KEY],
            option_dict[OUTER_LATITUDE_BUFFER_KEY]
        ])
    )
    error_checking.assert_is_valid_lat_numpy_array(outer_latitude_limits_deg_n)

    error_checking.assert_is_greater(option_dict[OUTER_LONGITUDE_BUFFER_KEY], 0)
    option_dict[OUTER_LONGITUDE_BUFFER_KEY] = number_rounding.round_to_nearest(
        option_dict[OUTER_LONGITUDE_BUFFER_KEY], GRID_SPACING_DEG
    )

    error_checking.assert_is_string_list(option_dict[INIT_DATE_LIMITS_KEY])
    init_dates_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, DATE_FORMAT)
        for t in option_dict[INIT_DATE_LIMITS_KEY]
    ], dtype=int)
    error_checking.assert_is_greater_numpy_array(init_dates_unix_sec, 0)

    error_checking.assert_is_string_list(option_dict[GFS_PREDICTOR_FIELDS_KEY])
    for this_field_name in option_dict[GFS_PREDICTOR_FIELDS_KEY]:
        gfs_utils.check_field_name(this_field_name)

    if any([
            f in gfs_utils.ALL_3D_FIELD_NAMES
            for f in option_dict[GFS_PREDICTOR_FIELDS_KEY]
    ]):
        assert option_dict[GFS_PRESSURE_LEVELS_KEY] is not None
    else:
        option_dict[GFS_PRESSURE_LEVELS_KEY] = None

    if option_dict[GFS_PRESSURE_LEVELS_KEY] is not None:
        error_checking.assert_is_numpy_array(
            option_dict[GFS_PRESSURE_LEVELS_KEY], num_dimensions=1
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[GFS_PRESSURE_LEVELS_KEY]
        )
        error_checking.assert_is_greater_numpy_array(
            option_dict[GFS_PRESSURE_LEVELS_KEY], 0
        )

    model_lead_days_to_gfs_pred_leads_hours = option_dict[
        MODEL_LEAD_TO_GFS_PRED_LEADS_KEY
    ]
    model_lead_times_days = numpy.array(
        list(model_lead_days_to_gfs_pred_leads_hours.keys()),
        dtype=int
    )

    error_checking.assert_is_integer_numpy_array(model_lead_times_days)
    error_checking.assert_is_greater_numpy_array(model_lead_times_days, 0)
    error_checking.assert_equals(
        len(model_lead_times_days),
        len(numpy.unique(model_lead_times_days))
    )

    for d in model_lead_times_days:
        these_pred_lead_times_hours = model_lead_days_to_gfs_pred_leads_hours[d]

        error_checking.assert_is_numpy_array(
            these_pred_lead_times_hours, num_dimensions=1
        )
        error_checking.assert_is_integer_numpy_array(
            these_pred_lead_times_hours
        )
        error_checking.assert_is_geq_numpy_array(these_pred_lead_times_hours, 0)

        these_pred_lead_times_hours = numpy.sort(these_pred_lead_times_hours)
        model_lead_days_to_gfs_pred_leads_hours[d] = these_pred_lead_times_hours

    model_lead_days_to_target_lags_days = option_dict[
        MODEL_LEAD_TO_TARGET_LAGS_KEY
    ]
    new_lead_times_days = numpy.array(
        list(model_lead_days_to_target_lags_days.keys()),
        dtype=int
    )
    assert numpy.array_equal(
        numpy.sort(model_lead_times_days),
        numpy.sort(new_lead_times_days)
    )

    for d in model_lead_times_days:
        these_lag_times_days = model_lead_days_to_target_lags_days[d]

        error_checking.assert_is_numpy_array(
            these_lag_times_days, num_dimensions=1
        )
        error_checking.assert_is_integer_numpy_array(these_lag_times_days)
        error_checking.assert_is_greater_numpy_array(these_lag_times_days, 0)

        these_lag_times_days = numpy.sort(these_lag_times_days)[::-1]
        model_lead_days_to_target_lags_days[d] = these_lag_times_days

    model_lead_days_to_freq = option_dict[MODEL_LEAD_TO_FREQ_KEY]
    new_lead_times_days = numpy.array(
        list(model_lead_days_to_freq.keys()),
        dtype=int
    )
    assert numpy.array_equal(
        numpy.sort(model_lead_times_days),
        numpy.sort(new_lead_times_days)
    )

    model_lead_time_freqs = numpy.array(
        [model_lead_days_to_freq[d] for d in model_lead_times_days],
        dtype=float
    )
    error_checking.assert_is_geq_numpy_array(model_lead_time_freqs, 0.)
    error_checking.assert_is_leq_numpy_array(model_lead_time_freqs, 1.)
    model_lead_time_freqs = (
        model_lead_time_freqs / numpy.sum(model_lead_time_freqs)
    )

    option_dict[MODEL_LEAD_TO_FREQ_KEY] = dict(zip(
        model_lead_times_days, model_lead_time_freqs
    ))

    error_checking.assert_is_boolean(option_dict[COMPARE_TO_GFS_IN_LOSS_KEY])

    error_checking.assert_directory_exists(option_dict[GFS_DIRECTORY_KEY])
    if option_dict[GFS_NORM_FILE_KEY] is None:
        option_dict[GFS_USE_QUANTILE_NORM_KEY] = False
    else:
        error_checking.assert_file_exists(option_dict[GFS_NORM_FILE_KEY])
        error_checking.assert_is_boolean(option_dict[GFS_USE_QUANTILE_NORM_KEY])

    if option_dict[ERA5_CONSTANT_PREDICTOR_FIELDS_KEY] is not None:
        error_checking.assert_is_string_list(
            option_dict[ERA5_CONSTANT_PREDICTOR_FIELDS_KEY]
        )
        for this_field_name in option_dict[ERA5_CONSTANT_PREDICTOR_FIELDS_KEY]:
            era5_constant_utils.check_field_name(this_field_name)

    if option_dict[ERA5_CONSTANT_FILE_KEY] is not None:
        error_checking.assert_file_exists(option_dict[ERA5_CONSTANT_FILE_KEY])
    if option_dict[ERA5_NORM_FILE_KEY] is None:
        option_dict[ERA5_USE_QUANTILE_NORM_KEY] = False
    else:
        error_checking.assert_file_exists(option_dict[ERA5_NORM_FILE_KEY])
        error_checking.assert_is_boolean(
            option_dict[ERA5_USE_QUANTILE_NORM_KEY]
        )

    error_checking.assert_is_string_list(option_dict[TARGET_FIELDS_KEY])
    for this_field_name in option_dict[TARGET_FIELDS_KEY]:
        canadian_fwi_utils.check_field_name(this_field_name)

    if option_dict[MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY] is not None:
        assert option_dict[GFS_FORECAST_TARGET_DIR_KEY] is not None

        model_lead_days_to_gfs_target_leads_days = option_dict[
            MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY
        ]

        for d in model_lead_times_days:
            these_target_lead_times_days = (
                model_lead_days_to_gfs_target_leads_days[d]
            )

            error_checking.assert_is_numpy_array(
                these_target_lead_times_days, num_dimensions=1
            )
            error_checking.assert_is_integer_numpy_array(
                these_target_lead_times_days
            )
            error_checking.assert_is_greater_numpy_array(
                these_target_lead_times_days, 0
            )

            these_target_lead_times_days = numpy.sort(
                these_target_lead_times_days
            )
            model_lead_days_to_gfs_target_leads_days[d] = (
                these_target_lead_times_days
            )

    if option_dict[GFS_FORECAST_TARGET_DIR_KEY] is not None:
        error_checking.assert_directory_exists(
            option_dict[GFS_FORECAST_TARGET_DIR_KEY]
        )

    error_checking.assert_directory_exists(option_dict[TARGET_DIRECTORY_KEY])
    if option_dict[TARGET_NORM_FILE_KEY] is None:
        option_dict[TARGETS_USE_QUANTILE_NORM_KEY] = False
    else:
        error_checking.assert_file_exists(
            option_dict[TARGET_NORM_FILE_KEY]
        )
        error_checking.assert_is_boolean(
            option_dict[TARGETS_USE_QUANTILE_NORM_KEY]
        )

    error_checking.assert_is_integer(option_dict[BATCH_SIZE_KEY])
    error_checking.assert_is_geq(option_dict[BATCH_SIZE_KEY], 1)
    # error_checking.assert_is_not_nan(option_dict[SENTINEL_VALUE_KEY])
    error_checking.assert_is_boolean(option_dict[DO_RESIDUAL_PREDICTION_KEY])
    error_checking.assert_is_boolean(option_dict[USE_LEAD_TIME_AS_PRED_KEY])

    if option_dict[CHANGE_LEAD_EVERY_N_BATCHES_KEY] is not None:
        error_checking.assert_is_integer(
            option_dict[CHANGE_LEAD_EVERY_N_BATCHES_KEY]
        )
        error_checking.assert_is_greater(
            option_dict[CHANGE_LEAD_EVERY_N_BATCHES_KEY], 0
        )

    if option_dict[OUTER_PATCH_SIZE_DEG_KEY] is None:
        option_dict[OUTER_PATCH_OVERLAP_DEG_KEY] = None
        return option_dict

    option_dict[OUTER_PATCH_SIZE_DEG_KEY] = number_rounding.round_to_nearest(
        option_dict[OUTER_PATCH_SIZE_DEG_KEY], GRID_SPACING_DEG
    )
    error_checking.assert_is_greater(
        option_dict[OUTER_PATCH_SIZE_DEG_KEY],
        option_dict[OUTER_LATITUDE_BUFFER_KEY]
    )
    error_checking.assert_is_greater(
        option_dict[OUTER_PATCH_SIZE_DEG_KEY],
        option_dict[OUTER_LONGITUDE_BUFFER_KEY]
    )

    option_dict[OUTER_PATCH_OVERLAP_DEG_KEY] = number_rounding.round_to_nearest(
        option_dict[OUTER_PATCH_OVERLAP_DEG_KEY], GRID_SPACING_DEG
    )
    error_checking.assert_is_greater(
        option_dict[OUTER_PATCH_OVERLAP_DEG_KEY], 0.
    )
    error_checking.assert_is_less_than(
        option_dict[OUTER_PATCH_OVERLAP_DEG_KEY],
        option_dict[OUTER_PATCH_SIZE_DEG_KEY]
    )

    return option_dict


def _find_gfs_forecast_target_file_1example(daily_gfs_dir_name,
                                            init_date_string):
    """Finds files with raw-GFS-forecast target fields for one example.

    :param daily_gfs_dir_name: Name of directory with daily GFS data.
    :param init_date_string: Initialization date (format "yyyymmdd") for GFS
        model run.
    :return: daily_gfs_file_name: File path.
    """

    try:
        return gfs_daily_io.find_file(
            directory_name=daily_gfs_dir_name,
            init_date_string=init_date_string,
            raise_error_if_missing=True
        )
    except:
        return None


def _find_target_files_needed_1example(
        gfs_init_date_string, target_dir_name, target_lead_times_days):
    """Finds target files needed for one data example.

    L = number of lead times

    :param gfs_init_date_string: Initialization date (format "yyyymmdd") for GFS
        model run.
    :param target_dir_name: Name of directory with target fields.
    :param target_lead_times_days: length-L numpy array of lead times.
    :return: target_file_names: length-L list of paths to target files.
    """

    gfs_init_date_unix_sec = time_conversion.string_to_unix_sec(
        gfs_init_date_string, DATE_FORMAT
    )
    target_dates_unix_sec = numpy.sort(
        gfs_init_date_unix_sec +
        target_lead_times_days * DAYS_TO_SECONDS
    )
    target_date_strings = [
        time_conversion.unix_sec_to_string(d, DATE_FORMAT)
        for d in target_dates_unix_sec
    ]

    return [
        canadian_fwi_io.find_file(
            directory_name=target_dir_name,
            valid_date_string=d,
            raise_error_if_missing=True
        ) for d in target_date_strings
    ]


def _get_2d_gfs_fields(gfs_table_xarray, field_names):
    """Returns 2-D fields from GFS table.

    M = number of rows in grid
    N = number of columns in grid
    L = number of lead times
    F = number of fields

    :param gfs_table_xarray: xarray table with GFS data.
    :param field_names: length-F list of field names.
    :return: predictor_matrix: None or an M-by-N-by-L-by-F numpy array.
    """

    if len(field_names) == 0:
        return None

    predictor_matrix = numpy.stack([
        gfs_utils.get_field(
            gfs_table_xarray=gfs_table_xarray, field_name=f
        )
        for f in field_names
    ], axis=-1)

    predictor_matrix = numpy.swapaxes(predictor_matrix, 0, 1)
    predictor_matrix = numpy.swapaxes(predictor_matrix, 1, 2)
    return predictor_matrix


def _get_3d_gfs_fields(gfs_table_xarray, field_names, pressure_levels_mb):
    """Returns 3-D fields from GFS table.

    M = number of rows in grid
    N = number of columns in grid
    P = number of pressure levels in grid
    L = number of lead times
    F = number of fields

    :param gfs_table_xarray: xarray table with GFS data.
    :param field_names: length-F list of field names.
    :param pressure_levels_mb: length-P numpy array of pressure levels.
    :return: predictor_matrix: None or an M-by-N-by-P-by-L-by-F numpy array.
    """

    if len(field_names) == 0:
        return None

    predictor_matrix = numpy.stack([
        gfs_utils.get_field(
            gfs_table_xarray=gfs_table_xarray,
            field_name=f, pressure_levels_mb=pressure_levels_mb
        )
        for f in field_names
    ], axis=-1)

    predictor_matrix = numpy.swapaxes(predictor_matrix, 0, 1)
    predictor_matrix = numpy.swapaxes(predictor_matrix, 1, 2)
    predictor_matrix = numpy.swapaxes(predictor_matrix, 2, 3)
    return predictor_matrix


def _pad_inner_to_outer_domain(
        data_matrix, outer_latitude_buffer_deg, outer_longitude_buffer_deg,
        is_example_axis_present, fill_value):
    """Pads inner domain to outer domain.

    :param data_matrix: numpy array of data values.
    :param outer_latitude_buffer_deg: Meridional buffer between inner
        and outer domains.  For example, if this value is 5, then the outer
        domain will extend 5 deg further south, and 5 deg further north, than
        the inner domain.
    :param outer_longitude_buffer_deg: Same but for longitude.
    :param is_example_axis_present: Boolean flag.  If True, will assume that the
        second and third axes of `data_matrix` are the row and column axes,
        respectively.  If False, this will be the first and second axes.
    :param fill_value: This value will be used to fill around the edges.
    :return: data_matrix: Same as input but with larger domain.
    """

    num_row_padding_pixels = int(numpy.round(
        outer_latitude_buffer_deg / GRID_SPACING_DEG
    ))
    num_column_padding_pixels = int(numpy.round(
        outer_longitude_buffer_deg / GRID_SPACING_DEG
    ))

    if is_example_axis_present:
        padding_arg = (
            (0, 0),
            (num_row_padding_pixels, num_row_padding_pixels),
            (num_column_padding_pixels, num_column_padding_pixels)
        )
    else:
        padding_arg = (
            (num_row_padding_pixels, num_row_padding_pixels),
            (num_column_padding_pixels, num_column_padding_pixels)
        )

    for i in range(len(data_matrix.shape)):
        if i < 2 + int(is_example_axis_present):
            continue

        padding_arg += ((0, 0),)

    return numpy.pad(
        data_matrix, pad_width=padding_arg, mode='constant',
        constant_values=fill_value
    )


def _get_gfs_forecast_target_fields(
        daily_gfs_file_name, lead_times_days,
        desired_row_indices, desired_column_indices,
        field_names, norm_param_table_xarray, use_quantile_norm):
    """Reads GFS-forecast target fields from one file.

    M = number of rows in grid
    N = number of columns in grid
    L = number of lead times
    T = number of target fields

    :param daily_gfs_file_name: Path to input file.
    :param lead_times_days: length-L numpy array of lead times.
    :param desired_row_indices: length-M numpy array of indices.
    :param desired_column_indices: length-N numpy array of indices.
    :param field_names: length-T list of field names.
    :param norm_param_table_xarray: xarray table with normalization parameters.
        If you do not want normalization, make this None.
    :param use_quantile_norm: Boolean flag.  If True, will use quantile
        normalization before converting to z-scores.
    :return: data_matrix: M-by-N-by-L-by-T numpy array of data values.
    """

    print('Reading data from: "{0:s}"...'.format(daily_gfs_file_name))
    daily_gfs_table_xarray = gfs_daily_io.read_file(daily_gfs_file_name)

    daily_gfs_table_xarray = gfs_daily_utils.subset_by_row(
        daily_gfs_table_xarray=daily_gfs_table_xarray,
        desired_row_indices=desired_row_indices
    )
    daily_gfs_table_xarray = gfs_daily_utils.subset_by_column(
        daily_gfs_table_xarray=daily_gfs_table_xarray,
        desired_column_indices=desired_column_indices
    )
    daily_gfs_table_xarray = gfs_daily_utils.subset_by_lead_time(
        daily_gfs_table_xarray=daily_gfs_table_xarray,
        lead_times_days=lead_times_days
    )
    if norm_param_table_xarray is not None:
        daily_gfs_table_xarray = normalization.normalize_gfs_fwi_forecasts(
            daily_gfs_table_xarray=daily_gfs_table_xarray,
            norm_param_table_xarray=norm_param_table_xarray,
            use_quantile_norm=use_quantile_norm
        )

    data_matrix = numpy.stack([
        gfs_daily_utils.get_field(
            daily_gfs_table_xarray=daily_gfs_table_xarray, field_name=f
        )
        for f in field_names
    ], axis=-1)

    assert not numpy.any(numpy.isnan(data_matrix))
    return data_matrix


def _get_target_fields(
        target_file_name, desired_row_indices, desired_column_indices,
        field_names, norm_param_table_xarray, use_quantile_norm):
    """Reads target fields from one file.

    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields

    :param target_file_name: Path to input file.
    :param desired_row_indices: length-M numpy array of indices.
    :param desired_column_indices: length-N numpy array of indices.
    :param field_names: length-T list of field names.
    :param norm_param_table_xarray: xarray table with normalization parameters.
        If you do not want normalization, make this None.
    :param use_quantile_norm: Boolean flag.  If True, will use quantile
        normalization before converting to z-scores.
    :return: data_matrix: M-by-N-by-T numpy array of data values.
    """

    print('Reading data from: "{0:s}"...'.format(target_file_name))
    fwi_table_xarray = canadian_fwi_io.read_file(target_file_name)

    fwi_table_xarray = canadian_fwi_utils.subset_by_row(
        fwi_table_xarray=fwi_table_xarray,
        desired_row_indices=desired_row_indices
    )
    fwi_table_xarray = canadian_fwi_utils.subset_by_column(
        fwi_table_xarray=fwi_table_xarray,
        desired_column_indices=desired_column_indices
    )
    if norm_param_table_xarray is not None:
        fwi_table_xarray = normalization.normalize_targets(
            fwi_table_xarray=fwi_table_xarray,
            norm_param_table_xarray=norm_param_table_xarray,
            use_quantile_norm=use_quantile_norm
        )

    data_matrix = numpy.stack([
        canadian_fwi_utils.get_field(
            fwi_table_xarray=fwi_table_xarray, field_name=f
        )
        for f in field_names
    ], axis=-1)

    assert not numpy.any(numpy.isnan(data_matrix))
    return data_matrix


def _create_weight_matrix(
        era5_constant_file_name, inner_latitude_limits_deg_n,
        inner_longitude_limits_deg_e, outer_latitude_buffer_deg,
        outer_longitude_buffer_deg):
    """Creates weight matrix for loss function.

    M = number of grid rows in outer domain
    N = number of grid columns in outer domain

    At each pixel, this weight is the product of the land mask (1 for land, 0
    for sea) and cos(latitude).

    :param era5_constant_file_name: See documentation for `data_generator`.
    :param inner_latitude_limits_deg_n: Same.
    :param inner_longitude_limits_deg_e: Same.
    :param outer_latitude_buffer_deg: Same.
    :param outer_longitude_buffer_deg: Same.
    :return: weight_matrix: M-by-N numpy array of weights in range 0...1.
    """

    era5_constant_table_xarray = era5_constant_io.read_file(
        era5_constant_file_name
    )
    ect = era5_constant_table_xarray

    desired_row_indices = misc_utils.desired_latitudes_to_rows(
        grid_latitudes_deg_n=
        ect.coords[era5_constant_utils.LATITUDE_DIM].values,
        start_latitude_deg_n=inner_latitude_limits_deg_n[0],
        end_latitude_deg_n=inner_latitude_limits_deg_n[1]
    )
    desired_column_indices = misc_utils.desired_longitudes_to_columns(
        grid_longitudes_deg_e=
        ect.coords[era5_constant_utils.LONGITUDE_DIM].values,
        start_longitude_deg_e=inner_longitude_limits_deg_e[0],
        end_longitude_deg_e=inner_longitude_limits_deg_e[1]
    )

    grid_latitudes_deg_n = (
        ect.coords[era5_constant_utils.LATITUDE_DIM].values[desired_row_indices]
    )
    latitude_cosines = numpy.cos(DEGREES_TO_RADIANS * grid_latitudes_deg_n)
    latitude_cosine_matrix = numpy.repeat(
        numpy.expand_dims(latitude_cosines, axis=1),
        axis=1, repeats=len(desired_column_indices)
    )

    ect = era5_constant_utils.subset_by_row(
        era5_constant_table_xarray=ect,
        desired_row_indices=desired_row_indices
    )
    ect = era5_constant_utils.subset_by_column(
        era5_constant_table_xarray=ect,
        desired_column_indices=desired_column_indices
    )

    land_mask_matrix = era5_constant_utils.get_field(
        era5_constant_table_xarray=ect,
        field_name=era5_constant_utils.LAND_SEA_MASK_NAME
    )
    weight_matrix = (
        latitude_cosine_matrix *
        (land_mask_matrix >= MASK_PIXEL_IF_WEIGHT_BELOW)
    )

    return _pad_inner_to_outer_domain(
        data_matrix=weight_matrix,
        outer_latitude_buffer_deg=outer_latitude_buffer_deg,
        outer_longitude_buffer_deg=outer_longitude_buffer_deg,
        is_example_axis_present=False, fill_value=0.
    )


def _get_era5_constants(
        era5_constant_file_name, latitude_limits_deg_n, longitude_limits_deg_e,
        field_names, norm_param_table_xarray, use_quantile_norm):
    """Reads ERA5 constants.

    M = number of rows in grid
    N = number of columns in grid
    F = number of fields

    :param era5_constant_file_name: See documentation for `data_generator`.
    :param latitude_limits_deg_n: Same.
    :param longitude_limits_deg_e: Same.
    :param field_names: Same.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, in format returned by
        `era5_constant_io.read_normalization_file`.  If you do not want
        normalization, make this None.
    :param use_quantile_norm: Boolean flag.  If True, will use quantile
        normalization before converting to z-scores.
    :return: predictor_matrix: M-by-N-by-F numpy array with ERA5 constants to
        use as predictors.
    """

    print('Reading data from: "{0:s}"...'.format(era5_constant_file_name))
    era5_constant_table_xarray = era5_constant_io.read_file(
        era5_constant_file_name
    )
    ect = era5_constant_table_xarray

    desired_row_indices = misc_utils.desired_latitudes_to_rows(
        grid_latitudes_deg_n=
        ect.coords[era5_constant_utils.LATITUDE_DIM].values,
        start_latitude_deg_n=latitude_limits_deg_n[0],
        end_latitude_deg_n=latitude_limits_deg_n[1]
    )
    desired_column_indices = misc_utils.desired_longitudes_to_columns(
        grid_longitudes_deg_e=
        ect.coords[era5_constant_utils.LONGITUDE_DIM].values,
        start_longitude_deg_e=longitude_limits_deg_e[0],
        end_longitude_deg_e=longitude_limits_deg_e[1]
    )
    ect = era5_constant_utils.subset_by_row(
        era5_constant_table_xarray=ect,
        desired_row_indices=desired_row_indices
    )
    ect = era5_constant_utils.subset_by_column(
        era5_constant_table_xarray=ect,
        desired_column_indices=desired_column_indices
    )

    if norm_param_table_xarray is not None:
        exec_start_time_unix_sec = time.time()
        ect = normalization.normalize_era5_constants(
            era5_constant_table_xarray=ect,
            norm_param_table_xarray=norm_param_table_xarray,
            use_quantile_norm=use_quantile_norm
        )

        print('Normalizing ERA5 data took {0:.4f} seconds.'.format(
            time.time() - exec_start_time_unix_sec
        ))

    return numpy.stack([
        era5_constant_utils.get_field(
            era5_constant_table_xarray=ect, field_name=f
        )
        for f in field_names
    ], axis=-1)


def _read_gfs_data_1example(
        gfs_file_name, desired_row_indices, desired_column_indices,
        latitude_limits_deg_n, longitude_limits_deg_e, lead_times_hours,
        field_names, pressure_levels_mb, norm_param_table_xarray,
        use_quantile_norm, num_lead_times_for_interp):
    """Reads GFS data for one example.

    M = number of rows in grid
    N = number of columns in grid
    P = number of pressure levels in grid
    L = number of lead times
    FFF = number of 3-D fields
    FF = number of 2-D fields

    :param gfs_file_name: Path to input file (will be read by
        `gfs_io.read_file`).
    :param desired_row_indices: length-M numpy array with indices of desired
        rows.  If this is None, it will be computed on the fly from
        `latitude_limits_deg_n`.
    :param desired_column_indices: length-N numpy array with indices of desired
        columns.  If this is None, it will be computed on the fly from
        `longitude_limits_deg_e`.
    :param latitude_limits_deg_n: See documentation for `data_generator`.  If
        `desired_row_indices is not None`, this argument is not used.
    :param longitude_limits_deg_e: See documentation for `data_generator`.  If
        `desired_column_indices is not None`, this argument is not used.
    :param lead_times_hours: See documentation for `data_generator`.
    :param field_names: See documentation for `data_generator`.
    :param pressure_levels_mb: See documentation for `data_generator`.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, in format returned by `gfs_io.read_normalization_file`.  If
        you do not want normalization, make this None.
    :param use_quantile_norm: Boolean flag.  If True, will use quantile
        normalization before converting to z-scores.
    :param num_lead_times_for_interp: Will interpolate to this number of evenly
        spaced lead times.  If you do not want to interpolate, make this None.
    :return: predictor_matrix_3d: None or an M-by-N-by-P-by-L-by-FFF numpy
        array.
    :return: predictor_matrix_2d: None or an M-by-N-by-L-by-FF numpy array.
    :return: desired_row_indices: See input documentation.
    :return: desired_column_indices: See input documentation.
    """

    field_names_2d = [
        f for f in field_names
        if f in gfs_utils.ALL_2D_FIELD_NAMES
    ]
    field_names_3d = [
        f for f in field_names
        if f in gfs_utils.ALL_3D_FIELD_NAMES
    ]

    print('Reading data from: "{0:s}"...'.format(gfs_file_name))
    gfs_table_xarray = gfs_io.read_file(gfs_file_name)

    if desired_row_indices is None or len(desired_row_indices) == 0:
        desired_row_indices = misc_utils.desired_latitudes_to_rows(
            grid_latitudes_deg_n=
            gfs_table_xarray.coords[gfs_utils.LATITUDE_DIM].values,
            start_latitude_deg_n=latitude_limits_deg_n[0],
            end_latitude_deg_n=latitude_limits_deg_n[1]
        )
        desired_column_indices = (
            misc_utils.desired_longitudes_to_columns(
                grid_longitudes_deg_e=
                gfs_table_xarray.coords[gfs_utils.LONGITUDE_DIM].values,
                start_longitude_deg_e=longitude_limits_deg_e[0],
                end_longitude_deg_e=longitude_limits_deg_e[1]
            )
        )

    # TODO(thunderhoser): I don't know if subsetting the whole table
    # first -- before extracting desired fields -- is more efficient.
    gfs_table_xarray = gfs_utils.subset_by_forecast_hour(
        gfs_table_xarray=gfs_table_xarray,
        desired_forecast_hours=lead_times_hours
    )
    gfs_table_xarray = gfs_utils.subset_by_row(
        gfs_table_xarray=gfs_table_xarray,
        desired_row_indices=desired_row_indices
    )
    gfs_table_xarray = gfs_utils.subset_by_column(
        gfs_table_xarray=gfs_table_xarray,
        desired_column_indices=desired_column_indices
    )

    if norm_param_table_xarray is not None:
        exec_start_time_unix_sec = time.time()
        gfs_table_xarray = normalization.normalize_gfs_data(
            gfs_table_xarray=gfs_table_xarray,
            norm_param_table_xarray=norm_param_table_xarray,
            use_quantile_norm=use_quantile_norm
        )

        print('Normalizing GFS data took {0:.4f} seconds.'.format(
            time.time() - exec_start_time_unix_sec
        ))

    predictor_matrix_3d = _get_3d_gfs_fields(
        gfs_table_xarray=gfs_table_xarray,
        field_names=field_names_3d,
        pressure_levels_mb=pressure_levels_mb
    )
    if predictor_matrix_3d is not None:
        predictor_matrix_3d = predictor_matrix_3d.astype('float32')

    predictor_matrix_2d = _get_2d_gfs_fields(
        gfs_table_xarray=gfs_table_xarray,
        field_names=field_names_2d
    )
    if predictor_matrix_2d is not None:
        predictor_matrix_2d = predictor_matrix_2d.astype('float32')

    if num_lead_times_for_interp is not None:
        if predictor_matrix_3d is not None:
            predictor_matrix_3d = _interp_predictors_by_lead_time(
                predictor_matrix=predictor_matrix_3d,
                source_lead_times_hours=lead_times_hours,
                num_target_lead_times=num_lead_times_for_interp
            )
            predictor_matrix_3d = predictor_matrix_3d.astype('float32')

        if predictor_matrix_2d is not None:
            predictor_matrix_2d = _interp_predictors_by_lead_time(
                predictor_matrix=predictor_matrix_2d,
                source_lead_times_hours=lead_times_hours,
                num_target_lead_times=num_lead_times_for_interp
            )
            predictor_matrix_2d = predictor_matrix_2d.astype('float32')

    return (
        predictor_matrix_3d, predictor_matrix_2d,
        desired_row_indices, desired_column_indices
    )


def _read_gfs_forecast_targets_1example(
        daily_gfs_dir_name, init_date_string, target_lead_times_days,
        desired_row_indices, desired_column_indices,
        latitude_limits_deg_n, longitude_limits_deg_e,
        target_field_names, norm_param_table_xarray, use_quantile_norm):
    """Reads raw-GFS-forecast target fields for one example.

    m = number of rows in grid
    n = number of columns in grid
    l = number of lead times
    T = number of target fields

    :param daily_gfs_dir_name: Name of directory with daily GFS data.
    :param init_date_string: GFS initialization date (format "yyyymmdd") for
        the given data example.
    :param target_lead_times_days: See documentation for `data_generator`.
    :param desired_row_indices: See documentation for
        `_read_lagged_targets_1example`.
    :param desired_column_indices: Same.
    :param latitude_limits_deg_n: Same.
    :param longitude_limits_deg_e: Same.
    :param target_field_names: Same.
    :param norm_param_table_xarray: Same.
    :param use_quantile_norm: Same.
    :return: target_field_matrix: m-by-n-by-l-by-T numpy array.
    :return: desired_row_indices: See input documentation.
    :return: desired_column_indices: See input documentation.
    """

    daily_gfs_file_name = _find_gfs_forecast_target_file_1example(
        daily_gfs_dir_name=daily_gfs_dir_name,
        init_date_string=init_date_string
    )

    if daily_gfs_file_name is None:
        return None, None, None

    if desired_row_indices is None or len(desired_row_indices) == 0:
        dgfst = gfs_daily_io.read_file(daily_gfs_file_name)

        desired_row_indices = misc_utils.desired_latitudes_to_rows(
            grid_latitudes_deg_n=
            dgfst.coords[gfs_daily_utils.LATITUDE_DIM].values,
            start_latitude_deg_n=latitude_limits_deg_n[0],
            end_latitude_deg_n=latitude_limits_deg_n[1]
        )
        desired_column_indices = misc_utils.desired_longitudes_to_columns(
            grid_longitudes_deg_e=
            dgfst.coords[gfs_daily_utils.LONGITUDE_DIM].values,
            start_longitude_deg_e=longitude_limits_deg_e[0],
            end_longitude_deg_e=longitude_limits_deg_e[1]
        )

    target_field_matrix = _get_gfs_forecast_target_fields(
        daily_gfs_file_name=daily_gfs_file_name,
        lead_times_days=target_lead_times_days,
        desired_row_indices=desired_row_indices,
        desired_column_indices=desired_column_indices,
        field_names=target_field_names,
        norm_param_table_xarray=norm_param_table_xarray,
        use_quantile_norm=use_quantile_norm
    )
    target_field_matrix = target_field_matrix.astype('float32')

    return target_field_matrix, desired_row_indices, desired_column_indices


def _read_lagged_targets_1example(
        gfs_init_date_string, target_dir_name, target_lag_times_days,
        desired_row_indices, desired_column_indices,
        latitude_limits_deg_n, longitude_limits_deg_e,
        target_field_names, norm_param_table_xarray, use_quantile_norm):
    """Reads lagged-target fields for one example.

    m = number of rows in grid
    n = number of columns in grid
    l = number of lag times
    T = number of target fields

    :param gfs_init_date_string: GFS initialization date (format "yyyymmdd") for
        the given data example.
    :param target_dir_name: See documentation for `data_generator`.
    :param target_lag_times_days: Same.
    :param desired_row_indices: length-m numpy array with indices of desired
        rows.  If this is None, it will be computed on the fly from
        `latitude_limits_deg_n`.
    :param desired_column_indices: length-n numpy array with indices of desired
        columns.  If this is None, it will be computed on the fly from
        `longitude_limits_deg_e`.
    :param latitude_limits_deg_n: See documentation for `data_generator`.  If
        `desired_row_indices is not None`, this argument is not used.
    :param longitude_limits_deg_e: See documentation for `data_generator`.  If
        `desired_column_indices is not None`, this argument is not used.
    :param target_field_names: See documentation for `data_generator`.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, in format returned by
        `canadian_fwi_io.read_normalization_file`.  If you do not want
        normalization, make this None.
    :param use_quantile_norm: Boolean flag.  If True, will use quantile
        normalization before converting to z-scores.
    :return: target_field_matrix: m-by-n-by-l-by-T numpy array.
    :return: desired_row_indices: See input documentation.
    :return: desired_column_indices: See input documentation.
    """

    target_file_names = _find_target_files_needed_1example(
        gfs_init_date_string=gfs_init_date_string,
        target_dir_name=target_dir_name,
        target_lead_times_days=-1 * target_lag_times_days
    )

    if desired_row_indices is None or len(desired_row_indices) == 0:
        fwit = canadian_fwi_io.read_file(target_file_names[0])

        desired_row_indices = misc_utils.desired_latitudes_to_rows(
            grid_latitudes_deg_n=
            fwit.coords[canadian_fwi_utils.LATITUDE_DIM].values,
            start_latitude_deg_n=latitude_limits_deg_n[0],
            end_latitude_deg_n=latitude_limits_deg_n[1]
        )
        desired_column_indices = misc_utils.desired_longitudes_to_columns(
            grid_longitudes_deg_e=
            fwit.coords[canadian_fwi_utils.LONGITUDE_DIM].values,
            start_longitude_deg_e=longitude_limits_deg_e[0],
            end_longitude_deg_e=longitude_limits_deg_e[1]
        )

    target_field_matrix = numpy.stack([
        _get_target_fields(
            target_file_name=f,
            desired_row_indices=desired_row_indices,
            desired_column_indices=desired_column_indices,
            field_names=target_field_names,
            norm_param_table_xarray=norm_param_table_xarray,
            use_quantile_norm=use_quantile_norm
        )
        for f in target_file_names
    ], axis=-2)

    target_field_matrix = target_field_matrix.astype('float32')

    return target_field_matrix, desired_row_indices, desired_column_indices


def _interp_predictors_by_lead_time(predictor_matrix, source_lead_times_hours,
                                    num_target_lead_times):
    """Interpolates predictors to fill missing lead times.

    M = number of rows in grid
    N = number of columns in grid
    S = number of source lead times
    T = number of target lead times
    F = number of fields
    P = number of pressure levels

    :param predictor_matrix: M-by-N-by-S-by-F or M-by-N-by-P-by-S-by-F numpy
        array of predictor values.
    :param source_lead_times_hours: length-S numpy array of source lead times.
        This method assumes that `source_lead_times_hours` is sorted.
    :param num_target_lead_times: T in the above definitions.
    :return: predictor_matrix: M-by-N-by-T-by-F or M-by-N-by-P-by-T-by-F numpy
        array of interpolated predictor values.
    """

    target_lead_times_hours = numpy.linspace(
        source_lead_times_hours[0], source_lead_times_hours[-1],
        num=num_target_lead_times, dtype=float
    )
    target_to_source_index = numpy.full(num_target_lead_times, -1, dtype=int)

    for t in range(num_target_lead_times):
        good_indices = numpy.where(
            numpy.absolute(source_lead_times_hours - target_lead_times_hours[t])
            <= TOLERANCE
        )[0]

        if len(good_indices) == 0:
            continue

        target_to_source_index[t] = good_indices[0]

    num_source_lead_times = predictor_matrix.shape[-2]
    num_fields = predictor_matrix.shape[-1]
    has_pressure_levels = len(predictor_matrix.shape) == 5

    if has_pressure_levels:
        num_pressure_levels = predictor_matrix.shape[2]
        these_dims = (
            predictor_matrix.shape[:2] +
            (num_source_lead_times, num_pressure_levels * num_fields)
        )
        predictor_matrix = numpy.reshape(predictor_matrix, these_dims)
    else:
        num_pressure_levels = 0

    num_pressure_field_combos = predictor_matrix.shape[-1]
    new_predictor_matrix = numpy.full(
        predictor_matrix.shape[:2] +
        (num_target_lead_times, num_pressure_field_combos),
        numpy.nan
    )

    for p in range(num_pressure_field_combos):
        missing_target_time_flags = numpy.full(
            num_target_lead_times, True, dtype=bool
        )

        for t in range(num_target_lead_times):
            if target_to_source_index[t] == -1:
                continue

            this_predictor_matrix = predictor_matrix[
                ..., target_to_source_index[t], p
            ]
            missing_target_time_flags[t] = numpy.all(numpy.isnan(
                this_predictor_matrix
            ))

            if missing_target_time_flags[t]:
                continue

            new_predictor_matrix[..., t, p] = this_predictor_matrix

        missing_target_time_indices = numpy.where(missing_target_time_flags)[0]
        if len(missing_target_time_indices) == 0:
            continue

        missing_source_time_flags = numpy.all(
            numpy.isnan(predictor_matrix[..., p]),
            axis=(0, 1)
        )
        filled_source_time_indices = numpy.where(
            numpy.invert(missing_source_time_flags)
        )[0]

        if len(filled_source_time_indices) == 0:
            continue

        missing_target_time_indices = [
            t for t in missing_target_time_indices
            if target_lead_times_hours[t] >= numpy.min(
                source_lead_times_hours[filled_source_time_indices]
            )
        ]
        missing_target_time_indices = [
            t for t in missing_target_time_indices
            if target_lead_times_hours[t] <= numpy.max(
                source_lead_times_hours[filled_source_time_indices]
            )
        ]
        missing_target_time_indices = numpy.array(
            missing_target_time_indices, dtype=int
        )
        if len(missing_target_time_indices) == 0:
            continue

        interp_object = interp1d(
            x=source_lead_times_hours[filled_source_time_indices],
            y=predictor_matrix[..., filled_source_time_indices, p],
            axis=2,
            kind='linear',
            bounds_error=True,
            assume_sorted=True
        )

        print((
            'Interpolating data from lead times of {0:s} hours to lead '
            'times of {1:s} hours...'
        ).format(
            str(source_lead_times_hours[filled_source_time_indices]),
            str(target_lead_times_hours[missing_target_time_indices])
        ))

        new_predictor_matrix[
            ..., missing_target_time_indices, p
        ] = interp_object(target_lead_times_hours[missing_target_time_indices])

    if has_pressure_levels:
        these_dims = (
            predictor_matrix.shape[:2] +
            (num_pressure_levels, num_target_lead_times, num_fields)
        )
        new_predictor_matrix = numpy.reshape(new_predictor_matrix, these_dims)

    return new_predictor_matrix


def create_learning_curriculum(lead_times_days, start_epoch_by_lead_time,
                               num_rampup_epochs):
    """Creates learning curriculum -- used for models with multi target times.

    L = number of target lead times

    :param lead_times_days: length-L numpy array of lead times.
    :param start_epoch_by_lead_time: length-L numpy array of start epochs.
        start_epoch_by_lead_time[k] is the first epoch at which the model will
        be trained to predict lead time lead_times_days[k].
    :param num_rampup_epochs: Number of ramp-up epochs.  This is the number of
        epochs it takes for the frequency of the shortest lead time to go from
        its max to min -- or for the frequency of any other lead time to go from
        its min to max.
    :return: epoch_and_lead_time_to_freq: Double indexed dictionary, where each
        key is (epoch_num, lead_time_days).  The corresponding value is the
        frequency with which the given lead time, at the given epoch, will be
        used for training.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(lead_times_days, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(lead_times_days)
    error_checking.assert_is_greater_numpy_array(lead_times_days, 0)

    num_lead_times = len(lead_times_days)
    error_checking.assert_is_numpy_array(
        start_epoch_by_lead_time,
        exact_dimensions=numpy.array([num_lead_times], dtype=int)
    )

    sort_indices = numpy.argsort(lead_times_days)
    lead_times_days = lead_times_days[sort_indices]
    start_epoch_by_lead_time = start_epoch_by_lead_time[sort_indices]
    start_epoch_by_lead_time[0] = 0

    error_checking.assert_is_integer_numpy_array(start_epoch_by_lead_time)
    error_checking.assert_is_geq_numpy_array(start_epoch_by_lead_time, 0)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(start_epoch_by_lead_time), 0
    )

    error_checking.assert_is_integer(num_rampup_epochs)
    error_checking.assert_is_greater(num_rampup_epochs, 0)

    # Do actual stuff.
    num_epochs = 10 * numpy.max(start_epoch_by_lead_time + num_rampup_epochs)
    epoch_nums = numpy.linspace(0, num_epochs - 1, num=num_epochs, dtype=float)

    multiplier = 10. / num_rampup_epochs
    num_lead_times_reciprocal = 1. / num_lead_times

    this_offset = 0.5 * num_rampup_epochs + start_epoch_by_lead_time[1]
    these_exp_values = numpy.exp(-multiplier * (epoch_nums - this_offset))
    first_lead_time_freqs = 1. - 1. / (1. + these_exp_values)
    first_lead_time_freqs = (
        num_lead_times_reciprocal +
        (1. - num_lead_times_reciprocal) * first_lead_time_freqs
    )

    epoch_and_lead_time_to_freq = dict()
    for i in range(num_epochs):
        epoch_and_lead_time_to_freq[i + 1, lead_times_days[0]] = (
            first_lead_time_freqs[i]
        )

    for j in range(1, num_lead_times):
        this_offset = 0.5 * num_rampup_epochs + start_epoch_by_lead_time[j]
        these_exp_values = numpy.exp(-multiplier * (epoch_nums - this_offset))
        these_lead_time_freqs = (
            num_lead_times_reciprocal / (1. + these_exp_values)
        )

        for i in range(num_epochs):
            epoch_and_lead_time_to_freq[i + 1, lead_times_days[j]] = (
                these_lead_time_freqs[i]
            )

    return epoch_and_lead_time_to_freq


def data_generator(option_dict):
    """Generates training or validation data for NN with flexible laad time.

    E = number of examples per batch = "batch size"
    M = number of grid rows in outer domain
    N = number of grid columns in outer domain
    L = number of GFS forecast hours (lead times)
    P = number of GFS pressure levels
    FFF = number of 3-D GFS predictor fields
    FF = number of 2-D GFS predictor fields
    F = number of ERA5-constant predictor fields
    l = number of time steps for lag/lead-target predictor fields
    T = number of target fields

    :param option_dict: Dictionary with the following keys.
    option_dict["inner_latitude_limits_deg_n"]: length-2 numpy array with
        meridional limits (deg north) of bounding box for inner (target) domain.
    option_dict["inner_longitude_limits_deg_e"]: Same but for longitude (deg
        east).
    option_dict["outer_latitude_buffer_deg"]: Meridional buffer between inner
        and outer domains.  For example, if this value is 5, then the outer
        domain will extend 5 deg further south, and 5 deg further north, than
        the inner domain.
    option_dict["outer_longitude_buffer_deg"]: Same but for longitude (deg
        east).
    option_dict["init_date_limit_strings"]: length-2 list with first and last
        GFS init dates to be used (format "yyyymmdd").  Will always use the 00Z
        model run.
    option_dict["gfs_predictor_field_names"]: 1-D list with names of GFS fields
        to be used as predictors.
    option_dict["gfs_pressure_levels_mb"]: 1-D numpy array with pressure levels
        to be used for GFS predictors (only the 3-D fields in the list
        "gfs_predictor_field_names").  If there are no 3-D fields, make this
        None.
    option_dict["model_lead_days_to_gfs_pred_leads_hours"]: Dictionary, where
        each key is a model lead time (days) and the corresponding value is a
        1-D numpy array of lead times (hours) for GFS-based predictors.
    option_dict["gfs_directory_name"]: Name of directory with GFS data.  Files
        therein will be found by `gfs_io.find_file` and read by
        `gfs_io.read_file`.
    option_dict["gfs_normalization_file_name"]: Path to file with normalization
        params for GFS data.  Will be read by `gfs_io.read_normalization_file`.
    option_dict["gfs_use_quantile_norm"]: Boolean flag.  If True, will do
        quantile normalization, then convert quantiles to standard normal
        distribution.  If False, will convert directly to z-scores.
    option_dict["era5_constant_predictor_field_names"]: 1-D list with names of
        ERA5 constant fields to be used as predictors.  If you do not want such
        predictors, make this None.
    option_dict["era5_constant_file_name"]: Path to file with ERA5 constants.
        Will be read by `era5_constant_io.read_file`.
    option_dict["era5_normalization_file_name"]: Path to file with normalization
        params for ERA5 time-constant data.  Will be read by
        `era5_constant_io.read_normalization_file`.
    option_dict["era5_use_quantile_norm"]: Same as "gfs_use_quantile_norm" but
        for ERA5 data.
    option_dict["target_field_names"]: length-T list with names of target fields
        (fire-weather indices).
    option_dict["model_lead_days_to_target_lags_days"]: Dictionary, where
        each key is a model lead time (days) and the corresponding value is a
        1-D numpy array of lag times (days) for reanalyzed target fields.
    option_dict["model_lead_days_to_gfs_target_leads_days"]: Dictionary, where
        each key is a model lead time (days) and the corresponding value is a
        1-D numpy array of lead times times (days) for GFS-based forecasts of
        target fields.
    option_dict["model_lead_days_to_freq"]: Dictionary, where each key is a
        model lead time (days) and the corresponding value is the frequency with
        which the model should be trained on this lead time (from 0...1).
    option_dict["compare_to_gfs_in_loss"]: Boolean flag.  If True, the loss
        function involves comparing to the GFS forecast for the same target
        variables at the same lead time.  In other words, the loss function
        involves a skill score, except with GFS instead of climo.
    option_dict["target_dir_name"]: Name of directory with target fields.
        Files therein will be found by `canadian_fwo_io.find_file` and read by
        `canadian_fwo_io.read_file`.
    option_dict["gfs_forecast_target_dir_name"]: Name of directory with raw-GFS
        forecasts of target fields.  Files therein will be found by
        `gfs_daily_io.find_file` and read by `gfs_daily_io.read_file`.
    option_dict["target_normalization_file_name"]: Path to file with
        normalization params for target fields.  Will be read by
        `canadian_fwi_io.read_normalization_file`.
    option_dict["targets_use_quantile_norm"]: Same as "gfs_use_quantile_norm"
        but for target fields.
    option_dict["num_examples_per_batch"]: Number of data examples per batch,
        usually just called "batch size".
    option_dict["sentinel_value"]: All NaN will be replaced with this value.
    option_dict["do_residual_prediction"]: Boolean flag.  If True, the neural
        net does residual prediction -- i.e., it predicts the FWI fields'
        difference between the most recent lag time and the target time.
        If False, the neural net does basic prediction, without using the
        residual.
    option_dict["use_lead_time_as_predictor"]: Boolean flag.  If True, will use
        the model lead time as a scalar predictor.
    option_dict["change_model_lead_every_n_batches"]: Will change model lead
        time only once every N batches, where N is this variable.  If you want
        to allow different model lead times in the same batch, make this None.
    option_dict["outer_patch_size_deg"]: Size of outer domain (in degrees) for
        each patch.  Recall that the outer domain is the predictor domain, while
        the inner domain is the target domain.  If you want to do full-domain
        training instead of patchwise training, make this None.
    option_dict["outer_patch_overlap_size_deg"]: Amount of overlap (in degrees)
        between adjacent outer patches.  If you want to do full-domain training
        instead of patchwise training, make this None.

    :return: predictor_matrices: List with the following items.  Some items may
        be missing.

    predictor_matrices[0]: E-by-M-by-N-by-P-by-L-by-FFF numpy array of 3-D GFS
        predictors.
    predictor_matrices[1]: E-by-M-by-N-by-L-by-FF numpy array of 2-D GFS
        predictors.
    predictor_matrices[2]: length-E numpy array of model lead times (days).
    predictor_matrices[3]: E-by-M-by-N-by-F numpy array of ERA5-constant
        predictors.
    predictor_matrices[4]: E-by-M-by-N-by-l-by-T numpy array of lag/lead-target
        predictors.
    predictor_matrices[5]: E-by-M-by-N-by-T numpy array of baseline values for
        residual prediction.

    :return: target_matrix: If `compare_to_gfs_in_loss == False`, this is an
        E-by-M-by-N-by-(T + 1) numpy array, where target_matrix[..., 0]
        through target_matrix[..., -2] contain values of the target fields and
        target_matrix[..., -1] contains a binary mask.  Where the binary mask is
        0, predictions should not be evaluated.

    If `compare_to_gfs_in_loss == True`, this is an
        E-by-M-by-N-by-(2T + 1) numpy array, where
        target_matrix[:T] contains actual values of the target fields;
        target_matrix[T:-1] contains GFS-forecast values of the target fields;
        and target_matrix[..., -1] contains the binary mask.
    """

    option_dict = _check_generator_args(option_dict)

    inner_latitude_limits_deg_n = option_dict[INNER_LATITUDE_LIMITS_KEY]
    inner_longitude_limits_deg_e = option_dict[INNER_LONGITUDE_LIMITS_KEY]
    outer_latitude_buffer_deg = option_dict[OUTER_LATITUDE_BUFFER_KEY]
    outer_longitude_buffer_deg = option_dict[OUTER_LONGITUDE_BUFFER_KEY]
    init_date_limit_strings = option_dict[INIT_DATE_LIMITS_KEY]
    gfs_predictor_field_names = option_dict[GFS_PREDICTOR_FIELDS_KEY]
    gfs_pressure_levels_mb = option_dict[GFS_PRESSURE_LEVELS_KEY]
    model_lead_days_to_gfs_pred_leads_hours = option_dict[
        MODEL_LEAD_TO_GFS_PRED_LEADS_KEY
    ]
    gfs_directory_name = option_dict[GFS_DIRECTORY_KEY]
    gfs_normalization_file_name = option_dict[GFS_NORM_FILE_KEY]
    gfs_use_quantile_norm = option_dict[GFS_USE_QUANTILE_NORM_KEY]
    era5_constant_predictor_field_names = option_dict[
        ERA5_CONSTANT_PREDICTOR_FIELDS_KEY
    ]
    era5_constant_file_name = option_dict[ERA5_CONSTANT_FILE_KEY]
    era5_normalization_file_name = option_dict[ERA5_NORM_FILE_KEY]
    era5_use_quantile_norm = option_dict[ERA5_USE_QUANTILE_NORM_KEY]
    target_field_names = option_dict[TARGET_FIELDS_KEY]
    model_lead_days_to_target_lags_days = option_dict[
        MODEL_LEAD_TO_TARGET_LAGS_KEY
    ]
    model_lead_days_to_gfs_target_leads_days = option_dict[
        MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY
    ]
    model_lead_days_to_freq = option_dict[MODEL_LEAD_TO_FREQ_KEY]
    compare_to_gfs_in_loss = option_dict[COMPARE_TO_GFS_IN_LOSS_KEY]
    target_dir_name = option_dict[TARGET_DIRECTORY_KEY]
    gfs_forecast_target_dir_name = option_dict[GFS_FORECAST_TARGET_DIR_KEY]
    target_normalization_file_name = option_dict[TARGET_NORM_FILE_KEY]
    targets_use_quantile_norm = option_dict[TARGETS_USE_QUANTILE_NORM_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    sentinel_value = option_dict[SENTINEL_VALUE_KEY]
    do_residual_prediction = option_dict[DO_RESIDUAL_PREDICTION_KEY]
    use_lead_time_as_predictor = option_dict[USE_LEAD_TIME_AS_PRED_KEY]
    change_model_lead_every_n_batches = option_dict[
        CHANGE_LEAD_EVERY_N_BATCHES_KEY
    ]

    model_lead_times_days = numpy.array(
        list(model_lead_days_to_freq.keys()),
        dtype=int
    )
    model_lead_time_freqs = numpy.array(
        [model_lead_days_to_freq[d] for d in model_lead_times_days],
        dtype=float
    )

    if gfs_normalization_file_name is None:
        gfs_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            gfs_normalization_file_name
        ))
        gfs_norm_param_table_xarray = gfs_io.read_normalization_file(
            gfs_normalization_file_name
        )

    if target_normalization_file_name is None:
        target_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            target_normalization_file_name
        ))
        target_norm_param_table_xarray = (
            canadian_fwi_io.read_normalization_file(
                target_normalization_file_name
            )
        )

    if era5_normalization_file_name is None:
        era5_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            era5_normalization_file_name
        ))
        era5_norm_param_table_xarray = era5_constant_io.read_normalization_file(
            era5_normalization_file_name
        )

    # TODO(thunderhoser): The longitude command below might fail.
    outer_latitude_limits_deg_n = inner_latitude_limits_deg_n + numpy.array([
        -1 * outer_latitude_buffer_deg, outer_latitude_buffer_deg
    ])
    outer_longitude_limits_deg_e = inner_longitude_limits_deg_e + numpy.array([
        -1 * outer_longitude_buffer_deg, outer_longitude_buffer_deg
    ])

    gfs_file_names = gfs_io.find_files_for_period(
        directory_name=gfs_directory_name,
        first_init_date_string=init_date_limit_strings[0],
        last_init_date_string=init_date_limit_strings[1],
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )
    random.shuffle(gfs_file_names)

    if era5_constant_predictor_field_names is None:
        era5_constant_matrix = None
    else:
        era5_constant_matrix = _get_era5_constants(
            era5_constant_file_name=era5_constant_file_name,
            latitude_limits_deg_n=outer_latitude_limits_deg_n,
            longitude_limits_deg_e=outer_longitude_limits_deg_e,
            field_names=era5_constant_predictor_field_names,
            norm_param_table_xarray=era5_norm_param_table_xarray,
            use_quantile_norm=era5_use_quantile_norm
        )
        era5_constant_matrix = numpy.repeat(
            numpy.expand_dims(era5_constant_matrix, axis=0),
            axis=0, repeats=num_examples_per_batch
        )
        era5_constant_matrix = era5_constant_matrix.astype('float32')

    weight_matrix = _create_weight_matrix(
        era5_constant_file_name=era5_constant_file_name,
        inner_latitude_limits_deg_n=inner_latitude_limits_deg_n,
        inner_longitude_limits_deg_e=inner_longitude_limits_deg_e,
        outer_latitude_buffer_deg=outer_latitude_buffer_deg,
        outer_longitude_buffer_deg=outer_longitude_buffer_deg
    )
    weight_matrix = numpy.repeat(
        numpy.expand_dims(weight_matrix, axis=0),
        axis=0, repeats=num_examples_per_batch
    )
    weight_matrix = numpy.expand_dims(weight_matrix, axis=-1)

    gfs_file_index = len(gfs_file_names)
    desired_gfs_row_indices = numpy.array([], dtype=int)
    desired_gfs_column_indices = numpy.array([], dtype=int)
    desired_target_row_indices = numpy.array([], dtype=int)
    desired_target_column_indices = numpy.array([], dtype=int)
    desired_gfs_fcst_target_row_indices = numpy.array([], dtype=int)
    desired_gfs_fcst_target_column_indices = numpy.array([], dtype=int)

    num_gfs_hours_for_interp, num_target_times_for_interp = (
        __determine_num_times_for_interp(option_dict)
    )
    num_batches_provided = 0

    while True:
        gfs_predictor_matrix_3d = None
        gfs_predictor_matrix_2d = None
        laglead_target_predictor_matrix = None
        baseline_prediction_matrix = None
        target_matrix = None
        num_examples_in_memory = 0

        bs = num_examples_per_batch

        change_model_lead_now = (
            change_model_lead_every_n_batches is not None and
            numpy.mod(num_batches_provided, change_model_lead_every_n_batches)
            == 0
        )

        if change_model_lead_now:
            model_lead_time_days = random.choices(
                model_lead_times_days, weights=model_lead_time_freqs, k=1
            )[0]

        while num_examples_in_memory < num_examples_per_batch:
            if gfs_file_index == len(gfs_file_names):
                random.shuffle(gfs_file_names)
                gfs_file_index = 0

            if change_model_lead_every_n_batches is None:
                model_lead_time_days = random.choices(
                    model_lead_times_days, weights=model_lead_time_freqs, k=1
                )[0]

            gfs_pred_lead_times_hours = model_lead_days_to_gfs_pred_leads_hours[
                model_lead_time_days
            ]

            if model_lead_days_to_target_lags_days is None:
                target_lag_times_days = numpy.array([], dtype=int)
            else:
                target_lag_times_days = model_lead_days_to_target_lags_days[
                    model_lead_time_days
                ]

            if model_lead_days_to_gfs_target_leads_days is None:
                gfs_target_lead_times_days = numpy.array([], dtype=int)
            else:
                gfs_target_lead_times_days = model_lead_days_to_gfs_target_leads_days[
                    model_lead_time_days
                ]

            if use_lead_time_as_predictor:
                lead_time_predictors_days = numpy.full(
                    num_examples_per_batch, model_lead_time_days, dtype=float
                )
            else:
                lead_time_predictors_days = None

            (
                this_gfs_predictor_matrix_3d, this_gfs_predictor_matrix_2d,
                desired_gfs_row_indices, desired_gfs_column_indices
            ) = _read_gfs_data_1example(
                gfs_file_name=gfs_file_names[gfs_file_index],
                desired_row_indices=desired_gfs_row_indices,
                desired_column_indices=desired_gfs_column_indices,
                latitude_limits_deg_n=outer_latitude_limits_deg_n,
                longitude_limits_deg_e=outer_longitude_limits_deg_e,
                lead_times_hours=gfs_pred_lead_times_hours,
                field_names=gfs_predictor_field_names,
                pressure_levels_mb=gfs_pressure_levels_mb,
                norm_param_table_xarray=gfs_norm_param_table_xarray,
                use_quantile_norm=gfs_use_quantile_norm,
                num_lead_times_for_interp=num_gfs_hours_for_interp
            )

            if this_gfs_predictor_matrix_3d is not None:
                if gfs_predictor_matrix_3d is None:
                    these_dim = (bs,) + this_gfs_predictor_matrix_3d.shape
                    gfs_predictor_matrix_3d = numpy.full(these_dim, numpy.nan)

                gfs_predictor_matrix_3d[num_examples_in_memory, ...] = (
                    this_gfs_predictor_matrix_3d
                )
                del this_gfs_predictor_matrix_3d

            if this_gfs_predictor_matrix_2d is not None:
                if gfs_predictor_matrix_2d is None:
                    these_dim = (bs,) + this_gfs_predictor_matrix_2d.shape
                    gfs_predictor_matrix_2d = numpy.full(these_dim, numpy.nan)

                gfs_predictor_matrix_2d[num_examples_in_memory, ...] = (
                    this_gfs_predictor_matrix_2d
                )
                del this_gfs_predictor_matrix_2d

            if do_residual_prediction:
                (
                    this_baseline_prediction_matrix, _, _
                ) = _read_lagged_targets_1example(
                    gfs_init_date_string=gfs_io.file_name_to_date(
                        gfs_file_names[gfs_file_index]
                    ),
                    target_dir_name=target_dir_name,
                    target_lag_times_days=numpy.array(
                        [numpy.min(target_lag_times_days)]
                    ),
                    desired_row_indices=desired_target_row_indices,
                    desired_column_indices=desired_target_column_indices,
                    latitude_limits_deg_n=inner_latitude_limits_deg_n,
                    longitude_limits_deg_e=inner_longitude_limits_deg_e,
                    target_field_names=target_field_names,
                    norm_param_table_xarray=None,
                    use_quantile_norm=False
                )
            else:
                this_baseline_prediction_matrix = None

            if this_baseline_prediction_matrix is not None:
                if baseline_prediction_matrix is None:
                    these_dim = (bs,) + this_baseline_prediction_matrix.shape
                    baseline_prediction_matrix = numpy.full(
                        these_dim, numpy.nan
                    )

                baseline_prediction_matrix[num_examples_in_memory, ...] = (
                    this_baseline_prediction_matrix
                )
                del this_baseline_prediction_matrix

            (
                this_laglead_target_predictor_matrix,
                desired_target_row_indices,
                desired_target_column_indices
            ) = _read_lagged_targets_1example(
                gfs_init_date_string=gfs_io.file_name_to_date(
                    gfs_file_names[gfs_file_index]
                ),
                target_dir_name=target_dir_name,
                target_lag_times_days=target_lag_times_days,
                desired_row_indices=desired_target_row_indices,
                desired_column_indices=desired_target_column_indices,
                latitude_limits_deg_n=inner_latitude_limits_deg_n,
                longitude_limits_deg_e=inner_longitude_limits_deg_e,
                target_field_names=target_field_names,
                norm_param_table_xarray=target_norm_param_table_xarray,
                use_quantile_norm=targets_use_quantile_norm
            )

            if len(gfs_target_lead_times_days) > 0:
                (
                    new_matrix,
                    desired_gfs_fcst_target_row_indices,
                    desired_gfs_fcst_target_column_indices
                ) = _read_gfs_forecast_targets_1example(
                    daily_gfs_dir_name=gfs_forecast_target_dir_name,
                    init_date_string=gfs_io.file_name_to_date(
                        gfs_file_names[gfs_file_index]
                    ),
                    target_lead_times_days=gfs_target_lead_times_days,
                    desired_row_indices=desired_gfs_fcst_target_row_indices,
                    desired_column_indices=
                    desired_gfs_fcst_target_column_indices,
                    latitude_limits_deg_n=inner_latitude_limits_deg_n,
                    longitude_limits_deg_e=inner_longitude_limits_deg_e,
                    target_field_names=target_field_names,
                    norm_param_table_xarray=target_norm_param_table_xarray,
                    use_quantile_norm=targets_use_quantile_norm
                )

                if new_matrix is None:
                    gfs_file_index += 1
                    continue

                this_laglead_target_predictor_matrix = numpy.concatenate(
                    [this_laglead_target_predictor_matrix, new_matrix],
                    axis=-2
                )
                del new_matrix

            if num_target_times_for_interp is not None:
                source_lead_times_days = numpy.concatenate([
                    numpy.sort(-1 * target_lag_times_days),
                    gfs_target_lead_times_days
                ])

                this_laglead_target_predictor_matrix = (
                    _interp_predictors_by_lead_time(
                        predictor_matrix=this_laglead_target_predictor_matrix,
                        source_lead_times_hours=
                        DAYS_TO_HOURS * source_lead_times_days,
                        num_target_lead_times=num_target_times_for_interp
                    )
                )

                this_laglead_target_predictor_matrix = (
                    this_laglead_target_predictor_matrix.astype('float32')
                )

            if laglead_target_predictor_matrix is None:
                these_dim = (bs,) + this_laglead_target_predictor_matrix.shape
                laglead_target_predictor_matrix = numpy.full(
                    these_dim, numpy.nan
                )

            laglead_target_predictor_matrix[num_examples_in_memory, ...] = (
                this_laglead_target_predictor_matrix
            )

            target_file_name = _find_target_files_needed_1example(
                gfs_init_date_string=
                gfs_io.file_name_to_date(gfs_file_names[gfs_file_index]),
                target_dir_name=target_dir_name,
                target_lead_times_days=
                numpy.array([model_lead_time_days], dtype=int)
            )[0]

            this_target_matrix = _get_target_fields(
                target_file_name=target_file_name,
                desired_row_indices=desired_target_row_indices,
                desired_column_indices=desired_target_column_indices,
                field_names=target_field_names,
                norm_param_table_xarray=None,
                use_quantile_norm=False
            )

            if compare_to_gfs_in_loss:
                (
                    new_target_matrix,
                    desired_gfs_fcst_target_row_indices,
                    desired_gfs_fcst_target_column_indices
                ) = _read_gfs_forecast_targets_1example(
                    daily_gfs_dir_name=gfs_forecast_target_dir_name,
                    init_date_string=gfs_io.file_name_to_date(
                        gfs_file_names[gfs_file_index]
                    ),
                    target_lead_times_days=numpy.array(
                        [model_lead_time_days], dtype=int
                    ),
                    desired_row_indices=desired_gfs_fcst_target_row_indices,
                    desired_column_indices=
                    desired_gfs_fcst_target_column_indices,
                    latitude_limits_deg_n=inner_latitude_limits_deg_n,
                    longitude_limits_deg_e=inner_longitude_limits_deg_e,
                    target_field_names=target_field_names,
                    norm_param_table_xarray=None,
                    use_quantile_norm=False
                )

                if new_target_matrix is None:
                    gfs_file_index += 1
                    continue

                new_target_matrix = new_target_matrix[..., 0, :]
                this_target_matrix = numpy.concatenate(
                    [this_target_matrix, new_target_matrix], axis=-1
                )
                del new_target_matrix

            if target_matrix is None:
                these_dim = (bs,) + this_target_matrix.shape
                target_matrix = numpy.full(these_dim, numpy.nan)

            target_matrix[num_examples_in_memory, ...] = this_target_matrix
            del this_target_matrix

            num_examples_in_memory += 1
            gfs_file_index += 1

        if do_residual_prediction:
            baseline_prediction_matrix = _pad_inner_to_outer_domain(
                data_matrix=baseline_prediction_matrix,
                outer_latitude_buffer_deg=outer_latitude_buffer_deg,
                outer_longitude_buffer_deg=outer_longitude_buffer_deg,
                is_example_axis_present=True, fill_value=sentinel_value
            )
            baseline_prediction_matrix = baseline_prediction_matrix[..., 0, :]

        laglead_target_predictor_matrix = _pad_inner_to_outer_domain(
            data_matrix=laglead_target_predictor_matrix,
            outer_latitude_buffer_deg=outer_latitude_buffer_deg,
            outer_longitude_buffer_deg=outer_longitude_buffer_deg,
            is_example_axis_present=True, fill_value=sentinel_value
        )

        target_matrix = _pad_inner_to_outer_domain(
            data_matrix=target_matrix,
            outer_latitude_buffer_deg=outer_latitude_buffer_deg,
            outer_longitude_buffer_deg=outer_longitude_buffer_deg,
            is_example_axis_present=True, fill_value=0.
        )
        target_matrix_with_weights = numpy.concatenate(
            [target_matrix, weight_matrix], axis=-1
        )

        predictor_matrices = __report_data_properties(
            gfs_predictor_matrix_3d=gfs_predictor_matrix_3d,
            gfs_predictor_matrix_2d=gfs_predictor_matrix_2d,
            lead_time_predictors_days=lead_time_predictors_days,
            era5_constant_matrix=era5_constant_matrix,
            laglead_target_predictor_matrix=laglead_target_predictor_matrix,
            baseline_prediction_matrix=baseline_prediction_matrix,
            target_matrix_with_weights=target_matrix_with_weights,
            sentinel_value=sentinel_value
        )[0]

        print('MODEL LEAD TIME: {0:d} days'.format(model_lead_time_days))
        num_batches_provided += 1
        yield predictor_matrices, target_matrix_with_weights


def data_generator_fast_patches(option_dict):
    """Same as `data_generator` but for patchwise training.

    :param option_dict: See documentation for `data_generator`.
    :return: predictor_matrices: Same.
    :return: target_matrix: Same.
    """

    option_dict = _check_generator_args(option_dict)

    inner_latitude_limits_deg_n = option_dict[INNER_LATITUDE_LIMITS_KEY]
    inner_longitude_limits_deg_e = option_dict[INNER_LONGITUDE_LIMITS_KEY]
    outer_latitude_buffer_deg = option_dict[OUTER_LATITUDE_BUFFER_KEY]
    outer_longitude_buffer_deg = option_dict[OUTER_LONGITUDE_BUFFER_KEY]
    init_date_limit_strings = option_dict[INIT_DATE_LIMITS_KEY]
    gfs_predictor_field_names = option_dict[GFS_PREDICTOR_FIELDS_KEY]
    gfs_pressure_levels_mb = option_dict[GFS_PRESSURE_LEVELS_KEY]
    model_lead_days_to_gfs_pred_leads_hours = option_dict[
        MODEL_LEAD_TO_GFS_PRED_LEADS_KEY
    ]
    gfs_directory_name = option_dict[GFS_DIRECTORY_KEY]
    gfs_normalization_file_name = option_dict[GFS_NORM_FILE_KEY]
    gfs_use_quantile_norm = option_dict[GFS_USE_QUANTILE_NORM_KEY]
    era5_constant_predictor_field_names = option_dict[
        ERA5_CONSTANT_PREDICTOR_FIELDS_KEY
    ]
    era5_constant_file_name = option_dict[ERA5_CONSTANT_FILE_KEY]
    era5_normalization_file_name = option_dict[ERA5_NORM_FILE_KEY]
    era5_use_quantile_norm = option_dict[ERA5_USE_QUANTILE_NORM_KEY]
    target_field_names = option_dict[TARGET_FIELDS_KEY]
    model_lead_days_to_target_lags_days = option_dict[
        MODEL_LEAD_TO_TARGET_LAGS_KEY
    ]
    model_lead_days_to_gfs_target_leads_days = option_dict[
        MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY
    ]
    model_lead_days_to_freq = option_dict[MODEL_LEAD_TO_FREQ_KEY]
    compare_to_gfs_in_loss = option_dict[COMPARE_TO_GFS_IN_LOSS_KEY]
    target_dir_name = option_dict[TARGET_DIRECTORY_KEY]
    gfs_forecast_target_dir_name = option_dict[GFS_FORECAST_TARGET_DIR_KEY]
    target_normalization_file_name = option_dict[TARGET_NORM_FILE_KEY]
    targets_use_quantile_norm = option_dict[TARGETS_USE_QUANTILE_NORM_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    sentinel_value = option_dict[SENTINEL_VALUE_KEY]
    do_residual_prediction = option_dict[DO_RESIDUAL_PREDICTION_KEY]
    use_lead_time_as_predictor = option_dict[USE_LEAD_TIME_AS_PRED_KEY]
    # change_model_lead_every_n_batches = option_dict[
    #     CHANGE_LEAD_EVERY_N_BATCHES_KEY
    # ]
    patch_size_deg = option_dict[OUTER_PATCH_SIZE_DEG_KEY]
    patch_overlap_size_deg = option_dict[OUTER_PATCH_OVERLAP_DEG_KEY]

    assert patch_size_deg is not None

    patch_size_pixels = int(numpy.round(
        float(patch_size_deg) / GRID_SPACING_DEG
    ))
    patch_overlap_size_pixels = int(numpy.round(
        float(patch_overlap_size_deg) / GRID_SPACING_DEG
    ))

    model_lead_times_days = numpy.array(
        list(model_lead_days_to_freq.keys()),
        dtype=int
    )
    model_lead_time_freqs = numpy.array(
        [model_lead_days_to_freq[d] for d in model_lead_times_days],
        dtype=float
    )

    if gfs_normalization_file_name is None:
        gfs_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            gfs_normalization_file_name
        ))
        gfs_norm_param_table_xarray = gfs_io.read_normalization_file(
            gfs_normalization_file_name
        )

    if target_normalization_file_name is None:
        target_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            target_normalization_file_name
        ))
        target_norm_param_table_xarray = (
            canadian_fwi_io.read_normalization_file(
                target_normalization_file_name
            )
        )

    if era5_normalization_file_name is None:
        era5_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            era5_normalization_file_name
        ))
        era5_norm_param_table_xarray = era5_constant_io.read_normalization_file(
            era5_normalization_file_name
        )

    # TODO(thunderhoser): The longitude command below might fail.
    outer_latitude_limits_deg_n = inner_latitude_limits_deg_n + numpy.array([
        -1 * outer_latitude_buffer_deg, outer_latitude_buffer_deg
    ])
    outer_longitude_limits_deg_e = inner_longitude_limits_deg_e + numpy.array([
        -1 * outer_longitude_buffer_deg, outer_longitude_buffer_deg
    ])

    gfs_file_names = gfs_io.find_files_for_period(
        directory_name=gfs_directory_name,
        first_init_date_string=init_date_limit_strings[0],
        last_init_date_string=init_date_limit_strings[1],
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )
    random.shuffle(gfs_file_names)

    if era5_constant_predictor_field_names is None:
        full_era5_constant_matrix = None
    else:
        full_era5_constant_matrix = _get_era5_constants(
            era5_constant_file_name=era5_constant_file_name,
            latitude_limits_deg_n=outer_latitude_limits_deg_n,
            longitude_limits_deg_e=outer_longitude_limits_deg_e,
            field_names=era5_constant_predictor_field_names,
            norm_param_table_xarray=era5_norm_param_table_xarray,
            use_quantile_norm=era5_use_quantile_norm
        )
        full_era5_constant_matrix = full_era5_constant_matrix.astype('float32')

    full_weight_matrix = _create_weight_matrix(
        era5_constant_file_name=era5_constant_file_name,
        inner_latitude_limits_deg_n=inner_latitude_limits_deg_n,
        inner_longitude_limits_deg_e=inner_longitude_limits_deg_e,
        outer_latitude_buffer_deg=outer_latitude_buffer_deg,
        outer_longitude_buffer_deg=outer_longitude_buffer_deg
    )
    full_weight_matrix = numpy.expand_dims(full_weight_matrix, axis=-1)

    gfs_file_index = 0
    desired_gfs_row_indices = numpy.array([], dtype=int)
    desired_gfs_column_indices = numpy.array([], dtype=int)
    desired_target_row_indices = numpy.array([], dtype=int)
    desired_target_column_indices = numpy.array([], dtype=int)
    desired_gfs_fcst_target_row_indices = numpy.array([], dtype=int)
    desired_gfs_fcst_target_column_indices = numpy.array([], dtype=int)

    num_gfs_hours_for_interp, num_target_times_for_interp = (
        __determine_num_times_for_interp(option_dict)
    )

    patch_metalocation_dict = __init_patch_metalocation_dict(
        num_rows_in_full_grid=full_weight_matrix.shape[0],
        num_columns_in_full_grid=full_weight_matrix.shape[1],
        patch_size_pixels=patch_size_pixels,
        patch_overlap_size_pixels=patch_overlap_size_pixels
    )

    full_gfs_predictor_matrix_3d = None
    full_gfs_predictor_matrix_2d = None
    full_laglead_target_predictor_matrix = None
    full_baseline_prediction_matrix = None
    full_target_matrix = None
    full_target_matrix_with_weights = None

    model_lead_time_days = random.choices(
        model_lead_times_days, weights=model_lead_time_freqs, k=1
    )[0]

    empty_matrix_dict = __init_matrices_1batch_patchwise(
        generator_option_dict=option_dict,
        gfs_file_names=gfs_file_names
    )

    while True:
        emd = empty_matrix_dict

        gfs_predictor_matrix_3d = emd[PREDICTOR_MATRIX_3D_GFS_KEY] + 0.
        gfs_predictor_matrix_2d = emd[PREDICTOR_MATRIX_2D_GFS_KEY] + 0.
        laglead_target_predictor_matrix = emd[PREDICTOR_MATRIX_LAGLEAD_KEY] + 0.
        era5_constant_matrix = emd[PREDICTOR_MATRIX_ERA5_KEY] + 0.
        baseline_prediction_matrix = emd[PREDICTOR_MATRIX_BASELINE_KEY] + 0.
        target_matrix_with_weights = emd[TARGET_MATRIX_WITH_WEIGHTS_KEY] + 0.

        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            patch_metalocation_dict = __update_patch_metalocation_dict(
                patch_metalocation_dict
            )

            if patch_metalocation_dict[PATCH_START_ROW_KEY] < 0:
                full_gfs_predictor_matrix_3d = None
                full_gfs_predictor_matrix_2d = None
                full_laglead_target_predictor_matrix = None
                full_baseline_prediction_matrix = None
                full_target_matrix = None
                full_target_matrix_with_weights = None

                model_lead_time_days = random.choices(
                    model_lead_times_days, weights=model_lead_time_freqs, k=1
                )[0]

                gfs_file_index, gfs_file_names = __increment_init_time(
                    current_index=gfs_file_index,
                    gfs_file_names=gfs_file_names
                )
                continue

            gfs_pred_lead_times_hours = model_lead_days_to_gfs_pred_leads_hours[
                model_lead_time_days
            ]

            if model_lead_days_to_target_lags_days is None:
                target_lag_times_days = numpy.array([], dtype=int)
            else:
                target_lag_times_days = model_lead_days_to_target_lags_days[
                    model_lead_time_days
                ]

            if model_lead_days_to_gfs_target_leads_days is None:
                gfs_target_lead_times_days = numpy.array([], dtype=int)
            else:
                gfs_target_lead_times_days = model_lead_days_to_gfs_target_leads_days[
                    model_lead_time_days
                ]

            if use_lead_time_as_predictor:
                lead_time_predictors_days = numpy.full(
                    num_examples_per_batch, model_lead_time_days, dtype=float
                )
            else:
                lead_time_predictors_days = None

            if (
                    full_gfs_predictor_matrix_3d is None
                    and full_gfs_predictor_matrix_2d is None
            ):
                (
                    full_gfs_predictor_matrix_3d, full_gfs_predictor_matrix_2d,
                    desired_gfs_row_indices, desired_gfs_column_indices
                ) = _read_gfs_data_1example(
                    gfs_file_name=gfs_file_names[gfs_file_index],
                    desired_row_indices=desired_gfs_row_indices,
                    desired_column_indices=desired_gfs_column_indices,
                    latitude_limits_deg_n=outer_latitude_limits_deg_n,
                    longitude_limits_deg_e=outer_longitude_limits_deg_e,
                    lead_times_hours=gfs_pred_lead_times_hours,
                    field_names=gfs_predictor_field_names,
                    pressure_levels_mb=gfs_pressure_levels_mb,
                    norm_param_table_xarray=gfs_norm_param_table_xarray,
                    use_quantile_norm=gfs_use_quantile_norm,
                    num_lead_times_for_interp=num_gfs_hours_for_interp
                )

            if (
                    do_residual_prediction and
                    full_baseline_prediction_matrix is None
            ):
                (
                    full_baseline_prediction_matrix, _, _
                ) = _read_lagged_targets_1example(
                    gfs_init_date_string=gfs_io.file_name_to_date(
                        gfs_file_names[gfs_file_index]
                    ),
                    target_dir_name=target_dir_name,
                    target_lag_times_days=numpy.array(
                        [numpy.min(target_lag_times_days)]
                    ),
                    desired_row_indices=desired_target_row_indices,
                    desired_column_indices=desired_target_column_indices,
                    latitude_limits_deg_n=inner_latitude_limits_deg_n,
                    longitude_limits_deg_e=inner_longitude_limits_deg_e,
                    target_field_names=target_field_names,
                    norm_param_table_xarray=None,
                    use_quantile_norm=False
                )

                full_baseline_prediction_matrix = _pad_inner_to_outer_domain(
                    data_matrix=full_baseline_prediction_matrix,
                    outer_latitude_buffer_deg=outer_latitude_buffer_deg,
                    outer_longitude_buffer_deg=outer_longitude_buffer_deg,
                    is_example_axis_present=False, fill_value=sentinel_value
                )
                full_baseline_prediction_matrix = (
                    full_baseline_prediction_matrix[..., 0, :]
                )

            if full_laglead_target_predictor_matrix is None:
                (
                    full_laglead_target_predictor_matrix,
                    desired_target_row_indices,
                    desired_target_column_indices
                ) = _read_lagged_targets_1example(
                    gfs_init_date_string=gfs_io.file_name_to_date(
                        gfs_file_names[gfs_file_index]
                    ),
                    target_dir_name=target_dir_name,
                    target_lag_times_days=target_lag_times_days,
                    desired_row_indices=desired_target_row_indices,
                    desired_column_indices=desired_target_column_indices,
                    latitude_limits_deg_n=inner_latitude_limits_deg_n,
                    longitude_limits_deg_e=inner_longitude_limits_deg_e,
                    target_field_names=target_field_names,
                    norm_param_table_xarray=target_norm_param_table_xarray,
                    use_quantile_norm=targets_use_quantile_norm
                )

                if len(gfs_target_lead_times_days) > 0:
                    (
                        new_matrix,
                        desired_gfs_fcst_target_row_indices,
                        desired_gfs_fcst_target_column_indices
                    ) = _read_gfs_forecast_targets_1example(
                        daily_gfs_dir_name=gfs_forecast_target_dir_name,
                        init_date_string=gfs_io.file_name_to_date(
                            gfs_file_names[gfs_file_index]
                        ),
                        target_lead_times_days=gfs_target_lead_times_days,
                        desired_row_indices=desired_gfs_fcst_target_row_indices,
                        desired_column_indices=
                        desired_gfs_fcst_target_column_indices,
                        latitude_limits_deg_n=inner_latitude_limits_deg_n,
                        longitude_limits_deg_e=inner_longitude_limits_deg_e,
                        target_field_names=target_field_names,
                        norm_param_table_xarray=target_norm_param_table_xarray,
                        use_quantile_norm=targets_use_quantile_norm
                    )

                    if new_matrix is None:
                        full_gfs_predictor_matrix_3d = None
                        full_gfs_predictor_matrix_2d = None
                        full_laglead_target_predictor_matrix = None
                        full_baseline_prediction_matrix = None
                        full_target_matrix = None
                        full_target_matrix_with_weights = None

                        gfs_file_index, gfs_file_names = __increment_init_time(
                            current_index=gfs_file_index,
                            gfs_file_names=gfs_file_names
                        )
                        continue

                    full_laglead_target_predictor_matrix = numpy.concatenate(
                        [full_laglead_target_predictor_matrix, new_matrix],
                        axis=-2
                    )
                    del new_matrix

                if num_target_times_for_interp is not None:
                    source_lead_times_days = numpy.concatenate([
                        numpy.sort(-1 * target_lag_times_days),
                        gfs_target_lead_times_days
                    ])

                    full_laglead_target_predictor_matrix = (
                        _interp_predictors_by_lead_time(
                            predictor_matrix=
                            full_laglead_target_predictor_matrix,
                            source_lead_times_hours=
                            DAYS_TO_HOURS * source_lead_times_days,
                            num_target_lead_times=num_target_times_for_interp
                        )
                    )

                    full_laglead_target_predictor_matrix = (
                        full_laglead_target_predictor_matrix.astype('float32')
                    )

                full_laglead_target_predictor_matrix = (
                    _pad_inner_to_outer_domain(
                        data_matrix=full_laglead_target_predictor_matrix,
                        outer_latitude_buffer_deg=outer_latitude_buffer_deg,
                        outer_longitude_buffer_deg=outer_longitude_buffer_deg,
                        is_example_axis_present=False, fill_value=sentinel_value
                    )
                )

            if full_target_matrix is None:
                target_file_name = _find_target_files_needed_1example(
                    gfs_init_date_string=
                    gfs_io.file_name_to_date(gfs_file_names[gfs_file_index]),
                    target_dir_name=target_dir_name,
                    target_lead_times_days=
                    numpy.array([model_lead_time_days], dtype=int)
                )[0]

                full_target_matrix = _get_target_fields(
                    target_file_name=target_file_name,
                    desired_row_indices=desired_target_row_indices,
                    desired_column_indices=desired_target_column_indices,
                    field_names=target_field_names,
                    norm_param_table_xarray=None,
                    use_quantile_norm=False
                )

                if compare_to_gfs_in_loss:
                    (
                        new_target_matrix,
                        desired_gfs_fcst_target_row_indices,
                        desired_gfs_fcst_target_column_indices
                    ) = _read_gfs_forecast_targets_1example(
                        daily_gfs_dir_name=gfs_forecast_target_dir_name,
                        init_date_string=gfs_io.file_name_to_date(
                            gfs_file_names[gfs_file_index]
                        ),
                        target_lead_times_days=numpy.array(
                            [model_lead_time_days], dtype=int
                        ),
                        desired_row_indices=desired_gfs_fcst_target_row_indices,
                        desired_column_indices=
                        desired_gfs_fcst_target_column_indices,
                        latitude_limits_deg_n=inner_latitude_limits_deg_n,
                        longitude_limits_deg_e=inner_longitude_limits_deg_e,
                        target_field_names=target_field_names,
                        norm_param_table_xarray=None,
                        use_quantile_norm=False
                    )

                    if new_target_matrix is None:
                        full_gfs_predictor_matrix_3d = None
                        full_gfs_predictor_matrix_2d = None
                        full_laglead_target_predictor_matrix = None
                        full_baseline_prediction_matrix = None
                        full_target_matrix = None
                        full_target_matrix_with_weights = None

                        gfs_file_index, gfs_file_names = __increment_init_time(
                            current_index=gfs_file_index,
                            gfs_file_names=gfs_file_names
                        )
                        continue

                    new_target_matrix = new_target_matrix[..., 0, :]
                    full_target_matrix = numpy.concatenate(
                        [full_target_matrix, new_target_matrix], axis=-1
                    )
                    del new_target_matrix

                full_target_matrix = _pad_inner_to_outer_domain(
                    data_matrix=full_target_matrix,
                    outer_latitude_buffer_deg=outer_latitude_buffer_deg,
                    outer_longitude_buffer_deg=outer_longitude_buffer_deg,
                    is_example_axis_present=False, fill_value=0.
                )
                full_target_matrix_with_weights = numpy.concatenate(
                    [full_target_matrix, full_weight_matrix], axis=-1
                )

            # Skip 75% of patches.
            if numpy.random.uniform(low=0., high=1., size=1)[0] < 0.8:
                continue

            patch_location_dict = misc_utils.determine_patch_location(
                num_rows_in_full_grid=full_target_matrix.shape[0],
                num_columns_in_full_grid=full_target_matrix.shape[1],
                patch_size_pixels=patch_size_pixels,
                start_row=patch_metalocation_dict[PATCH_START_ROW_KEY],
                start_column=patch_metalocation_dict[PATCH_START_COLUMN_KEY]
            )
            pld = patch_location_dict

            j_start = pld[misc_utils.ROW_LIMITS_KEY][0]
            j_end = pld[misc_utils.ROW_LIMITS_KEY][1] + 1
            k_start = pld[misc_utils.COLUMN_LIMITS_KEY][0]
            k_end = pld[misc_utils.COLUMN_LIMITS_KEY][1] + 1
            i = num_examples_in_memory + 0

            if full_gfs_predictor_matrix_3d is not None:
                gfs_predictor_matrix_3d[i, ...] = (
                    full_gfs_predictor_matrix_3d[j_start:j_end, k_start:k_end, ...]
                )

            if full_gfs_predictor_matrix_2d is not None:
                gfs_predictor_matrix_2d[i, ...] = (
                    full_gfs_predictor_matrix_2d[j_start:j_end, k_start:k_end, ...]
                )

            laglead_target_predictor_matrix[i, ...] = (
                full_laglead_target_predictor_matrix[j_start:j_end, k_start:k_end, ...]
            )

            if full_era5_constant_matrix is not None:
                era5_constant_matrix[i, ...] = (
                    full_era5_constant_matrix[j_start:j_end, k_start:k_end, ...]
                )

            if do_residual_prediction:
                baseline_prediction_matrix[i, ...] = (
                    full_baseline_prediction_matrix[j_start:j_end, k_start:k_end, ...]
                )

            target_matrix_with_weights[i, ...] = (
                full_target_matrix_with_weights[j_start:j_end, k_start:k_end, ...]
            )

            num_examples_in_memory += 1

        num_buffer_rows = int(numpy.round(
            float(outer_latitude_buffer_deg) / GRID_SPACING_DEG
        ))
        num_buffer_columns = int(numpy.round(
            float(outer_longitude_buffer_deg) / GRID_SPACING_DEG
        ))
        target_matrix_with_weights[:, :num_buffer_rows, ..., -1] = 0.
        target_matrix_with_weights[:, -num_buffer_rows:, ..., -1] = 0.
        target_matrix_with_weights[:, :, :num_buffer_columns, ..., -1] = 0.
        target_matrix_with_weights[:, : -num_buffer_columns:, ..., -1] = 0.

        predictor_matrices = __report_data_properties(
            gfs_predictor_matrix_3d=gfs_predictor_matrix_3d,
            gfs_predictor_matrix_2d=gfs_predictor_matrix_2d,
            lead_time_predictors_days=lead_time_predictors_days,
            era5_constant_matrix=era5_constant_matrix,
            laglead_target_predictor_matrix=laglead_target_predictor_matrix,
            baseline_prediction_matrix=baseline_prediction_matrix,
            target_matrix_with_weights=target_matrix_with_weights,
            sentinel_value=sentinel_value
        )[0]

        print('MODEL LEAD TIME: {0:d} days'.format(model_lead_time_days))
        yield predictor_matrices, target_matrix_with_weights


def create_data(
        option_dict, init_date_string, model_lead_time_days,
        patch_start_latitude_deg_n=None, patch_start_longitude_deg_e=None):
    """Creates, rather than generates, neural-net inputs.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid

    :param option_dict: See doc for `data_generator`.
    :param init_date_string: GFS initialization date (format "yyyymmdd").  Will
        always use the 00Z model run.
    :param model_lead_time_days: Model lead time.
    :param patch_start_latitude_deg_n:
        [used only if NN was trained with patches]
        Will return only the patch starting at this latitude.  If you would
        rather return all patches for the given initialization date, make this
        argument None.
    :param patch_start_longitude_deg_e:
        [used only if NN was trained with patches]
        Will return only the patch starting at this longitude.  If you would
        rather return all patches for the given initialization date, make this
        argument None.

    :return: data_dict: Dictionary with the following keys.
    data_dict["predictor_matrices"]: Same as output from `data_generator`.
    data_dict["target_matrix_with_weights"]: Same as output from
        `data_generator`.
    data_dict["grid_latitude_matrix_deg_n"]: E-by-M numpy array of latitudes
        (deg north).
    data_dict["grid_longitude_matrix_deg_e"]: E-by-N numpy array of longitudes
        (deg east).
    data_dict["input_layer_names"]: 1-D list with same length as
        "predictor_matrices", indicating the input layer for every predictor
        matrix.
    """

    _ = time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)
    error_checking.assert_is_integer(model_lead_time_days)
    error_checking.assert_is_greater(model_lead_time_days, 0)

    option_dict[INIT_DATE_LIMITS_KEY] = [init_date_string] * 2
    option_dict[BATCH_SIZE_KEY] = 32

    all_model_leads_days = numpy.array(
        list(option_dict[MODEL_LEAD_TO_TARGET_LAGS_KEY].keys()),
        dtype=int
    )
    all_model_leads_days = numpy.unique(all_model_leads_days)
    dummy_frequencies = numpy.full(len(all_model_leads_days), 0.1)
    option_dict[MODEL_LEAD_TO_FREQ_KEY] = dict(
        zip(all_model_leads_days, dummy_frequencies)
    )

    option_dict = _check_generator_args(option_dict)

    inner_latitude_limits_deg_n = option_dict[INNER_LATITUDE_LIMITS_KEY]
    inner_longitude_limits_deg_e = option_dict[INNER_LONGITUDE_LIMITS_KEY]
    outer_latitude_buffer_deg = option_dict[OUTER_LATITUDE_BUFFER_KEY]
    outer_longitude_buffer_deg = option_dict[OUTER_LONGITUDE_BUFFER_KEY]
    gfs_predictor_field_names = option_dict[GFS_PREDICTOR_FIELDS_KEY]
    gfs_pressure_levels_mb = option_dict[GFS_PRESSURE_LEVELS_KEY]
    model_lead_days_to_gfs_pred_leads_hours = option_dict[
        MODEL_LEAD_TO_GFS_PRED_LEADS_KEY
    ]
    gfs_directory_name = option_dict[GFS_DIRECTORY_KEY]
    gfs_normalization_file_name = option_dict[GFS_NORM_FILE_KEY]
    gfs_use_quantile_norm = option_dict[GFS_USE_QUANTILE_NORM_KEY]
    era5_constant_predictor_field_names = option_dict[
        ERA5_CONSTANT_PREDICTOR_FIELDS_KEY
    ]
    era5_constant_file_name = option_dict[ERA5_CONSTANT_FILE_KEY]
    era5_normalization_file_name = option_dict[ERA5_NORM_FILE_KEY]
    era5_use_quantile_norm = option_dict[ERA5_USE_QUANTILE_NORM_KEY]
    target_field_names = option_dict[TARGET_FIELDS_KEY]
    model_lead_days_to_target_lags_days = option_dict[
        MODEL_LEAD_TO_TARGET_LAGS_KEY
    ]
    model_lead_days_to_gfs_target_leads_days = option_dict[
        MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY
    ]
    target_dir_name = option_dict[TARGET_DIRECTORY_KEY]
    gfs_forecast_target_dir_name = option_dict[GFS_FORECAST_TARGET_DIR_KEY]
    target_normalization_file_name = option_dict[TARGET_NORM_FILE_KEY]
    targets_use_quantile_norm = option_dict[TARGETS_USE_QUANTILE_NORM_KEY]
    sentinel_value = option_dict[SENTINEL_VALUE_KEY]
    do_residual_prediction = option_dict[DO_RESIDUAL_PREDICTION_KEY]
    use_lead_time_as_predictor = option_dict[USE_LEAD_TIME_AS_PRED_KEY]

    num_gfs_hours_for_interp, num_target_times_for_interp = (
        __determine_num_times_for_interp(option_dict)
    )

    gfs_pred_lead_times_hours = model_lead_days_to_gfs_pred_leads_hours[
        model_lead_time_days
    ]

    if model_lead_days_to_target_lags_days is None:
        target_lag_times_days = numpy.array([], dtype=int)
    else:
        target_lag_times_days = model_lead_days_to_target_lags_days[
            model_lead_time_days
        ]

    if model_lead_days_to_gfs_target_leads_days is None:
        gfs_target_lead_times_days = numpy.array([], dtype=int)
    else:
        gfs_target_lead_times_days = model_lead_days_to_gfs_target_leads_days[
            model_lead_time_days
        ]

    if use_lead_time_as_predictor:
        lead_time_predictors_days = numpy.array(
            [model_lead_time_days], dtype=float
        )
        lead_time_predictors_days = numpy.expand_dims(
            lead_time_predictors_days, axis=0
        )
    else:
        lead_time_predictors_days = None

    if gfs_normalization_file_name is None:
        gfs_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            gfs_normalization_file_name
        ))
        gfs_norm_param_table_xarray = gfs_io.read_normalization_file(
            gfs_normalization_file_name
        )

    if target_normalization_file_name is None:
        target_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            target_normalization_file_name
        ))
        target_norm_param_table_xarray = (
            canadian_fwi_io.read_normalization_file(
                target_normalization_file_name
            )
        )

    if era5_normalization_file_name is None:
        era5_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            era5_normalization_file_name
        ))
        era5_norm_param_table_xarray = era5_constant_io.read_normalization_file(
            era5_normalization_file_name
        )

    outer_latitude_limits_deg_n = inner_latitude_limits_deg_n + numpy.array([
        -1 * outer_latitude_buffer_deg, outer_latitude_buffer_deg
    ])
    outer_longitude_limits_deg_e = inner_longitude_limits_deg_e + numpy.array([
        -1 * outer_longitude_buffer_deg, outer_longitude_buffer_deg
    ])

    if era5_constant_predictor_field_names is None:
        era5_constant_matrix = None
    else:
        era5_constant_matrix = _get_era5_constants(
            era5_constant_file_name=era5_constant_file_name,
            latitude_limits_deg_n=outer_latitude_limits_deg_n,
            longitude_limits_deg_e=outer_longitude_limits_deg_e,
            field_names=era5_constant_predictor_field_names,
            norm_param_table_xarray=era5_norm_param_table_xarray,
            use_quantile_norm=era5_use_quantile_norm
        )
        era5_constant_matrix = numpy.expand_dims(era5_constant_matrix, axis=0)
        era5_constant_matrix = era5_constant_matrix.astype('float32')

    weight_matrix = _create_weight_matrix(
        era5_constant_file_name=era5_constant_file_name,
        inner_latitude_limits_deg_n=inner_latitude_limits_deg_n,
        inner_longitude_limits_deg_e=inner_longitude_limits_deg_e,
        outer_latitude_buffer_deg=outer_latitude_buffer_deg,
        outer_longitude_buffer_deg=outer_longitude_buffer_deg
    )
    weight_matrix = numpy.expand_dims(weight_matrix, axis=-1)
    weight_matrix = numpy.expand_dims(weight_matrix, axis=0)

    gfs_file_name = gfs_io.find_file(
        directory_name=gfs_directory_name,
        init_date_string=init_date_string,
        raise_error_if_missing=True
    )

    (
        gfs_predictor_matrix_3d,
        gfs_predictor_matrix_2d,
        desired_gfs_row_indices,
        desired_gfs_column_indices
    ) = _read_gfs_data_1example(
        gfs_file_name=gfs_file_name,
        desired_row_indices=None,
        desired_column_indices=None,
        latitude_limits_deg_n=outer_latitude_limits_deg_n,
        longitude_limits_deg_e=outer_longitude_limits_deg_e,
        lead_times_hours=gfs_pred_lead_times_hours,
        field_names=gfs_predictor_field_names,
        pressure_levels_mb=gfs_pressure_levels_mb,
        norm_param_table_xarray=gfs_norm_param_table_xarray,
        use_quantile_norm=gfs_use_quantile_norm,
        num_lead_times_for_interp=num_gfs_hours_for_interp
    )

    gfs_table_xarray = gfs_io.read_file(gfs_file_name)
    grid_latitudes_deg_n = gfs_table_xarray.coords[
        gfs_utils.LATITUDE_DIM
    ].values[desired_gfs_row_indices]
    grid_longitudes_deg_e = gfs_table_xarray.coords[
        gfs_utils.LONGITUDE_DIM
    ].values[desired_gfs_column_indices]

    if gfs_predictor_matrix_3d is not None:
        gfs_predictor_matrix_3d = numpy.expand_dims(
            gfs_predictor_matrix_3d, axis=0
        )

    if gfs_predictor_matrix_2d is not None:
        gfs_predictor_matrix_2d = numpy.expand_dims(
            gfs_predictor_matrix_2d, axis=0
        )

    (
        laglead_target_predictor_matrix,
        desired_target_row_indices,
        desired_target_column_indices
    ) = _read_lagged_targets_1example(
        gfs_init_date_string=gfs_io.file_name_to_date(gfs_file_name),
        target_dir_name=target_dir_name,
        target_lag_times_days=target_lag_times_days,
        desired_row_indices=None,
        desired_column_indices=None,
        latitude_limits_deg_n=inner_latitude_limits_deg_n,
        longitude_limits_deg_e=inner_longitude_limits_deg_e,
        target_field_names=target_field_names,
        norm_param_table_xarray=target_norm_param_table_xarray,
        use_quantile_norm=targets_use_quantile_norm
    )

    if do_residual_prediction:
        baseline_prediction_matrix, _, _ = _read_lagged_targets_1example(
            gfs_init_date_string=gfs_io.file_name_to_date(gfs_file_name),
            target_dir_name=target_dir_name,
            target_lag_times_days=numpy.array(
                [numpy.min(target_lag_times_days)]
            ),
            desired_row_indices=desired_target_row_indices,
            desired_column_indices=desired_target_column_indices,
            latitude_limits_deg_n=inner_latitude_limits_deg_n,
            longitude_limits_deg_e=inner_longitude_limits_deg_e,
            target_field_names=target_field_names,
            norm_param_table_xarray=None,
            use_quantile_norm=False
        )

        baseline_prediction_matrix = numpy.expand_dims(
            baseline_prediction_matrix, axis=0
        )
    else:
        baseline_prediction_matrix = None

    if len(gfs_target_lead_times_days) > 0:
        new_matrix = _read_gfs_forecast_targets_1example(
            daily_gfs_dir_name=gfs_forecast_target_dir_name,
            init_date_string=gfs_io.file_name_to_date(gfs_file_name),
            target_lead_times_days=gfs_target_lead_times_days,
            desired_row_indices=None,
            desired_column_indices=None,
            latitude_limits_deg_n=inner_latitude_limits_deg_n,
            longitude_limits_deg_e=inner_longitude_limits_deg_e,
            target_field_names=target_field_names,
            norm_param_table_xarray=target_norm_param_table_xarray,
            use_quantile_norm=targets_use_quantile_norm
        )[0]

        laglead_target_predictor_matrix = numpy.concatenate(
            (laglead_target_predictor_matrix, new_matrix),
            axis=-2
        )

    if num_target_times_for_interp is not None:
        source_lead_times_days = numpy.concatenate([
            numpy.sort(-1 * target_lag_times_days),
            gfs_target_lead_times_days
        ])

        laglead_target_predictor_matrix = (
            _interp_predictors_by_lead_time(
                predictor_matrix=laglead_target_predictor_matrix,
                source_lead_times_hours=
                DAYS_TO_HOURS * source_lead_times_days,
                num_target_lead_times=num_target_times_for_interp
            )
        )

        laglead_target_predictor_matrix = (
            laglead_target_predictor_matrix.astype('float32')
        )

    laglead_target_predictor_matrix = numpy.expand_dims(
        laglead_target_predictor_matrix, axis=0
    )

    target_file_name = _find_target_files_needed_1example(
        gfs_init_date_string=gfs_io.file_name_to_date(gfs_file_name),
        target_dir_name=target_dir_name,
        target_lead_times_days=
        numpy.array([model_lead_time_days], dtype=int)
    )[0]

    target_matrix = _get_target_fields(
        target_file_name=target_file_name,
        desired_row_indices=desired_target_row_indices,
        desired_column_indices=desired_target_column_indices,
        field_names=target_field_names,
        norm_param_table_xarray=None,
        use_quantile_norm=False
    )
    target_matrix = numpy.expand_dims(target_matrix, axis=0)

    baseline_prediction_matrix = _pad_inner_to_outer_domain(
        data_matrix=baseline_prediction_matrix,
        outer_latitude_buffer_deg=outer_latitude_buffer_deg,
        outer_longitude_buffer_deg=outer_longitude_buffer_deg,
        is_example_axis_present=True, fill_value=sentinel_value
    )
    baseline_prediction_matrix = baseline_prediction_matrix[..., 0, :]

    laglead_target_predictor_matrix = _pad_inner_to_outer_domain(
        data_matrix=laglead_target_predictor_matrix,
        outer_latitude_buffer_deg=outer_latitude_buffer_deg,
        outer_longitude_buffer_deg=outer_longitude_buffer_deg,
        is_example_axis_present=True, fill_value=sentinel_value
    )

    target_matrix = _pad_inner_to_outer_domain(
        data_matrix=target_matrix,
        outer_latitude_buffer_deg=outer_latitude_buffer_deg,
        outer_longitude_buffer_deg=outer_longitude_buffer_deg,
        is_example_axis_present=True, fill_value=0.
    )
    target_matrix_with_weights = numpy.concatenate(
        [target_matrix, weight_matrix], axis=-1
    )

    predictor_matrices, input_layer_names = __report_data_properties(
        gfs_predictor_matrix_3d=gfs_predictor_matrix_3d,
        gfs_predictor_matrix_2d=gfs_predictor_matrix_2d,
        lead_time_predictors_days=lead_time_predictors_days,
        era5_constant_matrix=era5_constant_matrix,
        laglead_target_predictor_matrix=laglead_target_predictor_matrix,
        baseline_prediction_matrix=baseline_prediction_matrix,
        target_matrix_with_weights=target_matrix_with_weights,
        sentinel_value=sentinel_value
    )
    print('MODEL LEAD TIME: {0:d} days'.format(model_lead_time_days))

    predictor_matrices = list(predictor_matrices)
    grid_latitude_matrix_deg_n = numpy.expand_dims(grid_latitudes_deg_n, axis=0)
    grid_longitude_matrix_deg_e = numpy.expand_dims(
        grid_longitudes_deg_e, axis=0
    )

    patch_size_deg = option_dict[OUTER_PATCH_SIZE_DEG_KEY]
    patch_overlap_size_deg = option_dict[OUTER_PATCH_OVERLAP_DEG_KEY]

    if patch_size_deg is None:
        return {
            PREDICTOR_MATRICES_KEY: predictor_matrices,
            TARGETS_AND_WEIGHTS_KEY: target_matrix_with_weights,
            GRID_LATITUDE_MATRIX_KEY: grid_latitude_matrix_deg_n,
            GRID_LONGITUDE_MATRIX_KEY: grid_longitude_matrix_deg_e,
            INPUT_LAYER_NAMES_KEY: input_layer_names
        }

    # Deal with specific patch location.
    patch_size_pixels = int(numpy.round(
        float(patch_size_deg) / GRID_SPACING_DEG
    ))
    patch_overlap_size_pixels = int(numpy.round(
        float(patch_overlap_size_deg) / GRID_SPACING_DEG
    ))

    if not (
            patch_start_latitude_deg_n is None
            or patch_start_longitude_deg_e is None
    ):
        latitude_diffs_deg = numpy.absolute(
            grid_latitudes_deg_n - patch_start_latitude_deg_n
        )
        patch_start_row = numpy.argmin(latitude_diffs_deg)

        if latitude_diffs_deg[patch_start_row] > TOLERANCE:
            error_string = (
                'Cannot find desired start latitude ({0:.4f} deg N) for '
                'patch.  Nearest grid-point latitude is {1:.4f} deg N.'
            ).format(
                patch_start_latitude_deg_n,
                grid_latitudes_deg_n[patch_start_row]
            )

            raise ValueError(error_string)

        grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            grid_longitudes_deg_e
        )
        patch_start_longitude_deg_e = (
            lng_conversion.convert_lng_positive_in_west(
                patch_start_longitude_deg_e
            )
        )
        longitude_diffs_deg = numpy.absolute(
            grid_longitudes_deg_e - patch_start_longitude_deg_e
        )
        patch_start_column = numpy.argmin(longitude_diffs_deg)

        if longitude_diffs_deg[patch_start_column] > TOLERANCE:
            error_string = (
                'Cannot find desired start longitude ({0:.4f} deg E) for '
                'patch.  Nearest grid-point longitude is {1:.4f} deg E.'
            ).format(
                patch_start_longitude_deg_e,
                grid_longitudes_deg_e[patch_start_column]
            )

            raise ValueError(error_string)

        patch_location_dict = misc_utils.determine_patch_location(
            num_rows_in_full_grid=len(grid_latitudes_deg_n),
            num_columns_in_full_grid=len(grid_longitudes_deg_e),
            patch_size_pixels=patch_size_pixels,
            start_row=patch_start_row,
            start_column=patch_start_column
        )
        pld = patch_location_dict

        j_start = pld[misc_utils.ROW_LIMITS_KEY][0]
        j_end = pld[misc_utils.ROW_LIMITS_KEY][1] + 1
        k_start = pld[misc_utils.COLUMN_LIMITS_KEY][0]
        k_end = pld[misc_utils.COLUMN_LIMITS_KEY][1] + 1

        patch_target_matrix_with_weights = (
            target_matrix_with_weights[:, j_start:j_end, k_start:k_end, ...]
        )
        patch_latitude_matrix_deg_n = (
            grid_latitude_matrix_deg_n[:, j_start:j_end]
        )
        patch_longitude_matrix_deg_e = (
            grid_longitude_matrix_deg_e[:, k_start:k_end]
        )

        num_matrices = len(predictor_matrices)
        patch_predictor_matrices = [None] * num_matrices

        for j in range(num_matrices):
            if predictor_matrices[j] is None:
                continue

            if len(predictor_matrices[j].shape) < 3:
                patch_predictor_matrices[j] = predictor_matrices[j]
                continue

            patch_predictor_matrices[j] = (
                predictor_matrices[j][:, j_start:j_end, k_start:k_end, ...]
            )

        num_buffer_rows = int(numpy.round(
            float(outer_latitude_buffer_deg) / GRID_SPACING_DEG
        ))
        num_buffer_columns = int(numpy.round(
            float(outer_longitude_buffer_deg) / GRID_SPACING_DEG
        ))
        patch_target_matrix_with_weights[:, :num_buffer_rows, ..., -1] = 0.
        patch_target_matrix_with_weights[:, -num_buffer_rows:, ..., -1] = 0.
        patch_target_matrix_with_weights[:, :, :num_buffer_columns, ..., -1] = 0.
        patch_target_matrix_with_weights[:, : -num_buffer_columns:, ..., -1] = 0.

        return {
            PREDICTOR_MATRICES_KEY: patch_predictor_matrices,
            TARGETS_AND_WEIGHTS_KEY: patch_target_matrix_with_weights,
            GRID_LATITUDE_MATRIX_KEY: patch_latitude_matrix_deg_n,
            GRID_LONGITUDE_MATRIX_KEY: patch_longitude_matrix_deg_e,
            INPUT_LAYER_NAMES_KEY: input_layer_names
        }

    # Determine number of patches in full grid.
    patch_metalocation_dict = __init_patch_metalocation_dict(
        num_rows_in_full_grid=len(grid_latitudes_deg_n),
        num_columns_in_full_grid=len(grid_longitudes_deg_e),
        patch_size_pixels=patch_size_pixels,
        patch_overlap_size_pixels=patch_overlap_size_pixels
    )

    pmld = patch_metalocation_dict
    num_patches = 0

    while True:
        pmld = __update_patch_metalocation_dict(pmld)
        if pmld[PATCH_START_ROW_KEY] < 0:
            break

        num_patches += 1

    # Initialize output arrays.
    dummy_patch_location_dict = misc_utils.determine_patch_location(
        num_rows_in_full_grid=len(grid_latitudes_deg_n),
        num_columns_in_full_grid=len(grid_longitudes_deg_e),
        patch_size_pixels=patch_size_pixels
    )
    dummy_pld = dummy_patch_location_dict
    num_rows_per_patch = (
        dummy_pld[misc_utils.ROW_LIMITS_KEY][1]
        - dummy_pld[misc_utils.ROW_LIMITS_KEY][0] + 1
    )
    num_columns_per_patch = (
        dummy_pld[misc_utils.COLUMN_LIMITS_KEY][1]
        - dummy_pld[misc_utils.COLUMN_LIMITS_KEY][0] + 1
    )

    num_matrices = len(predictor_matrices)
    patch_predictor_matrices = [None] * num_matrices

    for j in range(num_matrices):
        if predictor_matrices[j] is None:
            continue

        if len(predictor_matrices[j].shape) < 3:
            patch_predictor_matrices[j] = predictor_matrices[j]
            continue

        these_dim = (
            (num_patches, num_rows_per_patch, num_columns_per_patch) +
            predictor_matrices[j].shape[3:]
        )
        patch_predictor_matrices[j] = numpy.full(these_dim, numpy.nan)

    these_dim = (
        (num_patches, num_rows_per_patch, num_columns_per_patch) +
        target_matrix_with_weights.shape[3:]
    )
    patch_target_matrix_with_weights = numpy.full(these_dim, numpy.nan)
    patch_latitude_matrix_deg_n = numpy.full(
        (num_patches, num_rows_per_patch), numpy.nan
    )
    patch_longitude_matrix_deg_e = numpy.full(
        (num_patches, num_columns_per_patch), numpy.nan
    )

    # Populate output arrays.
    patch_metalocation_dict = __init_patch_metalocation_dict(
        num_rows_in_full_grid=len(grid_latitudes_deg_n),
        num_columns_in_full_grid=len(grid_longitudes_deg_e),
        patch_size_pixels=patch_size_pixels,
        patch_overlap_size_pixels=patch_overlap_size_pixels
    )

    for i in range(num_patches):
        patch_metalocation_dict = __update_patch_metalocation_dict(
            patch_metalocation_dict
        )
        patch_location_dict = misc_utils.determine_patch_location(
            num_rows_in_full_grid=len(grid_latitudes_deg_n),
            num_columns_in_full_grid=len(grid_longitudes_deg_e),
            patch_size_pixels=patch_size_pixels,
            start_row=patch_metalocation_dict[PATCH_START_ROW_KEY],
            start_column=patch_metalocation_dict[PATCH_START_COLUMN_KEY]
        )
        pld = patch_location_dict

        j_start = pld[misc_utils.ROW_LIMITS_KEY][0]
        j_end = pld[misc_utils.ROW_LIMITS_KEY][1] + 1
        k_start = pld[misc_utils.COLUMN_LIMITS_KEY][0]
        k_end = pld[misc_utils.COLUMN_LIMITS_KEY][1] + 1

        patch_target_matrix_with_weights[i, ...] = (
            target_matrix_with_weights[0, j_start:j_end, k_start:k_end, ...]
        )
        patch_latitude_matrix_deg_n[i, :] = (
            grid_latitude_matrix_deg_n[0, j_start:j_end]
        )
        patch_longitude_matrix_deg_e[i, :] = (
            grid_longitude_matrix_deg_e[0, k_start:k_end]
        )

        for j in range(num_matrices):
            if patch_predictor_matrices[j] is None:
                continue
            if len(patch_predictor_matrices[j].shape) < 3:
                continue

            patch_predictor_matrices[j][i, ...] = (
                predictor_matrices[j][0, j_start:j_end, k_start:k_end, ...]
            )

    num_buffer_rows = int(numpy.round(
        float(outer_latitude_buffer_deg) / GRID_SPACING_DEG
    ))
    num_buffer_columns = int(numpy.round(
        float(outer_longitude_buffer_deg) / GRID_SPACING_DEG
    ))
    patch_target_matrix_with_weights[:, :num_buffer_rows, ..., -1] = 0.
    patch_target_matrix_with_weights[:, -num_buffer_rows:, ..., -1] = 0.
    patch_target_matrix_with_weights[:, :, :num_buffer_columns, ..., -1] = 0.
    patch_target_matrix_with_weights[:, : -num_buffer_columns:, ..., -1] = 0.

    return {
        PREDICTOR_MATRICES_KEY: patch_predictor_matrices,
        TARGETS_AND_WEIGHTS_KEY: patch_target_matrix_with_weights,
        GRID_LATITUDE_MATRIX_KEY: patch_latitude_matrix_deg_n,
        GRID_LONGITUDE_MATRIX_KEY: patch_longitude_matrix_deg_e,
        INPUT_LAYER_NAMES_KEY: input_layer_names
    }


def find_metafile(model_file_name, raise_error_if_missing=True):
    """Finds metafile for neural net.

    :param model_file_name: Path to model file.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: metafile_name: Path to metafile.
    """

    error_checking.assert_is_string(model_file_name)
    metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    if raise_error_if_missing and not os.path.isfile(metafile_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            metafile_name
        )
        raise ValueError(error_string)

    return metafile_name


def write_metafile(
        pickle_file_name, num_epochs, num_training_batches_per_epoch,
        training_option_dict, num_validation_batches_per_epoch,
        validation_option_dict, loss_function_string,
        metric_function_strings, optimizer_function_string,
        chiu_net_architecture_dict, chiu_net_pp_architecture_dict,
        plateau_patience_epochs, plateau_learning_rate_multiplier,
        early_stopping_patience_epochs):
    """Writes metadata to Pickle file.

    :param pickle_file_name: Path to output file.
    :param num_epochs: See doc for `train_model`.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param loss_function_string: Same.
    :param metric_function_strings: Same.
    :param optimizer_function_string: Same.
    :param chiu_net_architecture_dict: Same.
    :param chiu_net_pp_architecture_dict: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    """

    metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        TRAINING_OPTIONS_KEY: training_option_dict,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        VALIDATION_OPTIONS_KEY: validation_option_dict,
        LOSS_FUNCTION_KEY: loss_function_string,
        METRIC_FUNCTIONS_KEY: metric_function_strings,
        OPTIMIZER_FUNCTION_KEY: optimizer_function_string,
        CHIU_NET_ARCHITECTURE_KEY: chiu_net_architecture_dict,
        CHIU_NET_PP_ARCHITECTURE_KEY: chiu_net_pp_architecture_dict,
        PLATEAU_PATIENCE_KEY: plateau_patience_epochs,
        PLATEAU_LR_MUTIPLIER_KEY: plateau_learning_rate_multiplier,
        EARLY_STOPPING_PATIENCE_KEY: early_stopping_patience_epochs
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_metafile(pickle_file_name):
    """Reads metadata from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict["num_epochs"]: See doc for `train_model`.
    metadata_dict["num_training_batches_per_epoch"]: Same.
    metadata_dict["training_option_dict"]: Same.
    metadata_dict["num_validation_batches_per_epoch"]: Same.
    metadata_dict["validation_option_dict"]: Same.
    metadata_dict["loss_function_string"]: Same.
    metadata_dict["metric_function_strings"]: Same.
    metadata_dict["optimizer_function_string"]: Same.
    metadata_dict["chiu_net_architecture_dict"]: Same.
    metadata_dict["chiu_net_pp_architecture_dict"]: Same.
    metadata_dict["plateau_patience_epochs"]: Same.
    metadata_dict["plateau_learning_rate_multiplier"]: Same.
    metadata_dict["early_stopping_patience_epochs"]: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    if METRIC_FUNCTIONS_KEY not in metadata_dict:
        metadata_dict[METRIC_FUNCTIONS_KEY] = []
    if CHIU_NET_ARCHITECTURE_KEY not in metadata_dict:
        metadata_dict[CHIU_NET_ARCHITECTURE_KEY] = None
    if CHIU_NET_PP_ARCHITECTURE_KEY not in metadata_dict:
        metadata_dict[CHIU_NET_PP_ARCHITECTURE_KEY] = None

    tod = metadata_dict[TRAINING_OPTIONS_KEY]
    vod = metadata_dict[VALIDATION_OPTIONS_KEY]

    if MODEL_LEAD_TO_GFS_PRED_LEADS_KEY not in tod:
        try:
            model_lead_time_days = tod['target_lead_time_days']

            this_dict = {
                model_lead_time_days: tod['gfs_predictor_lead_times_hours']
            }
            tod[MODEL_LEAD_TO_GFS_PRED_LEADS_KEY] = this_dict
            vod[MODEL_LEAD_TO_GFS_PRED_LEADS_KEY] = this_dict

            this_dict = {
                model_lead_time_days: tod['target_lag_times_days']
            }
            tod[MODEL_LEAD_TO_TARGET_LAGS_KEY] = this_dict
            vod[MODEL_LEAD_TO_TARGET_LAGS_KEY] = this_dict

            this_dict = {
                model_lead_time_days: tod['gfs_forecast_target_lead_times_days']
            }
            tod[MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY] = this_dict
            vod[MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY] = this_dict
        except KeyError:
            pass

    if OUTER_PATCH_SIZE_DEG_KEY not in tod:
        tod[OUTER_PATCH_SIZE_DEG_KEY] = None
        tod[OUTER_PATCH_OVERLAP_DEG_KEY] = None

        vod[OUTER_PATCH_SIZE_DEG_KEY] = None
        vod[OUTER_PATCH_OVERLAP_DEG_KEY] = None

    if (
            metadata_dict[CHIU_NET_PP_ARCHITECTURE_KEY] is not None
            and 'use_convnext_blocks' not in
            metadata_dict[CHIU_NET_PP_ARCHITECTURE_KEY]
    ):
        metadata_dict[CHIU_NET_PP_ARCHITECTURE_KEY][
            'use_convnext_blocks'
        ] = False

    metadata_dict[TRAINING_OPTIONS_KEY] = tod
    metadata_dict[VALIDATION_OPTIONS_KEY] = vod

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def read_model(hdf5_file_name):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model`.
    """

    error_checking.assert_file_exists(hdf5_file_name)

    metafile_name = find_metafile(
        model_file_name=hdf5_file_name, raise_error_if_missing=True
    )
    metadata_dict = read_metafile(metafile_name)
    print(metadata_dict[LOSS_FUNCTION_KEY])

    chiu_net_architecture_dict = metadata_dict[CHIU_NET_ARCHITECTURE_KEY]
    if chiu_net_architecture_dict is not None:
        import chiu_net_architecture

        arch_dict = chiu_net_architecture_dict

        for this_key in [
                chiu_net_architecture.LOSS_FUNCTION_KEY,
                chiu_net_architecture.OPTIMIZER_FUNCTION_KEY
        ]:
            arch_dict[this_key] = eval(arch_dict[this_key])

        for this_key in [chiu_net_architecture.METRIC_FUNCTIONS_KEY]:
            for k in range(len(arch_dict[this_key])):
                arch_dict[this_key][k] = eval(arch_dict[this_key][k])

        model_object = chiu_net_architecture.create_model(arch_dict)
        model_object.load_weights(hdf5_file_name)
        return model_object

    chiu_net_pp_architecture_dict = metadata_dict[CHIU_NET_PP_ARCHITECTURE_KEY]
    if chiu_net_pp_architecture_dict is not None:
        import \
            chiu_net_pp_architecture

        arch_dict = chiu_net_pp_architecture_dict
        if chiu_net_pp_architecture.USE_LEAD_TIME_AS_PRED_KEY not in arch_dict:
            arch_dict[chiu_net_pp_architecture.USE_LEAD_TIME_AS_PRED_KEY] = False

        for this_key in [
                chiu_net_pp_architecture.LOSS_FUNCTION_KEY,
                chiu_net_pp_architecture.OPTIMIZER_FUNCTION_KEY
        ]:
            arch_dict[this_key] = eval(arch_dict[this_key])

        for this_key in [chiu_net_pp_architecture.METRIC_FUNCTIONS_KEY]:
            for k in range(len(arch_dict[this_key])):
                arch_dict[this_key][k] = eval(arch_dict[this_key][k])

        model_object = chiu_net_pp_architecture.create_model(arch_dict)
        model_object.load_weights(hdf5_file_name)
        return model_object

    custom_object_dict = {
        'loss': eval(metadata_dict[LOSS_FUNCTION_KEY])
    }
    model_object = load_model(
        hdf5_file_name, custom_objects=custom_object_dict, compile=False
    )

    metric_function_list = [
        eval(m) for m in metadata_dict[METRIC_FUNCTIONS_KEY]
    ]
    model_object.compile(
        loss=custom_object_dict['loss'],
        optimizer=eval(metadata_dict[OPTIMIZER_FUNCTION_KEY]),
        metrics=metric_function_list
    )

    return model_object


def read_model_for_shapley(pickle_file_name):
    """Reads model from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model`.
    """

    error_checking.assert_file_exists(pickle_file_name)

    metafile_name = find_metafile(
        model_file_name=pickle_file_name, raise_error_if_missing=True
    )
    metadata_dict = read_metafile(metafile_name)

    chiu_net_pp_architecture_dict = metadata_dict[CHIU_NET_PP_ARCHITECTURE_KEY]
    assert chiu_net_pp_architecture_dict is not None

    import \
        chiu_net_pp_architecture

    arch_dict = chiu_net_pp_architecture_dict
    if chiu_net_pp_architecture.USE_LEAD_TIME_AS_PRED_KEY not in arch_dict:
        arch_dict[chiu_net_pp_architecture.USE_LEAD_TIME_AS_PRED_KEY] = False

    arch_dict[chiu_net_pp_architecture.LOSS_FUNCTION_KEY] = 'mse'
    arch_dict[chiu_net_pp_architecture.METRIC_FUNCTIONS_KEY] = []
    arch_dict[chiu_net_pp_architecture.OPTIMIZER_FUNCTION_KEY] = (
        keras.optimizers.Adam()
    )

    for this_key in [chiu_net_pp_architecture.METRIC_FUNCTIONS_KEY]:
        for k in range(len(arch_dict[this_key])):
            arch_dict[this_key][k] = eval(arch_dict[this_key][k])

    model_object = chiu_net_pp_architecture.create_model(
        option_dict=arch_dict, omit_model_summary=True
    )
    orig_model_object = chiu_net_pp_architecture.create_model(
        option_dict=arch_dict, omit_model_summary=True
    )

    pickle_file_handle = open(pickle_file_name, 'rb')
    model_weights_array_list = pickle.load(pickle_file_handle)
    pickle_file_handle.close()
    model_object.set_weights(model_weights_array_list)

    layer_names = [layer.name for layer in model_object.layers]

    for this_layer_name in layer_names:
        orig_weights_array_list = orig_model_object.get_layer(
            name=this_layer_name
        ).get_weights()

        new_weights_array_list = model_object.get_layer(
            name=this_layer_name
        ).get_weights()

        for k in range(len(orig_weights_array_list)):
            if not numpy.allclose(
                    orig_weights_array_list[k], new_weights_array_list[k],
                    atol=TOLERANCE
            ):
                continue

            warning_string = (
                'POTENTIAL MAJOR ERROR: some weight tensors in layer "{0:s}" '
                'did not change!'
            ).format(this_layer_name)

            warnings.warn(warning_string)

    return model_object


def train_model(
        model_object, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        loss_function_string, metric_function_strings,
        optimizer_function_string, chiu_net_architecture_dict,
        chiu_net_pp_architecture_dict, plateau_patience_epochs,
        plateau_learning_rate_multiplier, early_stopping_patience_epochs,
        epoch_and_lead_time_to_freq, output_dir_name):
    """Trains neural net with generator.

    :param model_object: Untrained neural net (instance of
        `keras.models.Model`).
    :param num_epochs: Number of training epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param training_option_dict: See doc for `data_generator`.  This dictionary
        will be used to generate training data.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param validation_option_dict: See doc for `data_generator`.  For validation
        only, the following values will replace corresponding values in
        `training_option_dict`:
    validation_option_dict["init_date_limit_strings"]
    validation_option_dict["gfs_directory_name"]
    validation_option_dict["target_dir_name"]
    validation_option_dict["gfs_forecast_target_dir_name"]

    :param loss_function_string: Loss function.  This string should be formatted
        such that `eval(loss_function_string)` returns the actual loss function.
    :param metric_function_strings: 1-D list with names of metrics.  Each string
        should be formatted such that `eval(metric_function_strings[i])` returns
        the actual metric function.
    :param optimizer_function_string: Optimizer.  This string should be
        formatted such that `eval(optimizer_function_string)` returns the actual
        optimizer.
    :param chiu_net_architecture_dict: Dictionary with architecture options for
        `chiu_net_architecture.create_model`.  If the model being trained is not
        a Chiu-net, make this None.
    :param chiu_net_pp_architecture_dict: Dictionary with architecture options
        for `chiu_net_pp_architecture.create_model`.  If the model being trained
        is not a Chiu-net++, make this None.
    :param plateau_patience_epochs: Training will be deemed to have reached
        "plateau" if validation loss has not decreased in the last N epochs,
        where N = plateau_patience_epochs.
    :param plateau_learning_rate_multiplier: If training reaches "plateau,"
        learning rate will be multiplied by this value in range (0, 1).
    :param early_stopping_patience_epochs: Training will be stopped early if
        validation loss has not decreased in the last N epochs, where N =
        early_stopping_patience_epochs.
    :param epoch_and_lead_time_to_freq: Dictionary returned by
        `create_learning_curriculum`.
    :param output_dir_name: Path to output directory (model and training history
        will be saved here).
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    backup_dir_name = '{0:s}/backup_and_restore'.format(output_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=backup_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 2)
    error_checking.assert_is_integer(num_validation_batches_per_epoch)
    error_checking.assert_is_geq(num_validation_batches_per_epoch, 2)
    error_checking.assert_is_integer(plateau_patience_epochs)
    error_checking.assert_is_geq(plateau_patience_epochs, 2)
    error_checking.assert_is_greater(plateau_learning_rate_multiplier, 0.)
    error_checking.assert_is_less_than(plateau_learning_rate_multiplier, 1.)
    error_checking.assert_is_integer(early_stopping_patience_epochs)
    error_checking.assert_is_geq(early_stopping_patience_epochs, 5)

    validation_keys_to_keep = [
        INIT_DATE_LIMITS_KEY, GFS_DIRECTORY_KEY, TARGET_DIRECTORY_KEY,
        GFS_FORECAST_TARGET_DIR_KEY
    ]
    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    model_file_name = '{0:s}/model.weights.h5'.format(output_dir_name)
    history_file_name = '{0:s}/history.csv'.format(output_dir_name)

    try:
        history_table_pandas = pandas.read_csv(history_file_name)
        initial_epoch = history_table_pandas['epoch'].max() + 1
        best_validation_loss = history_table_pandas['val_loss'].min()
    except:
        initial_epoch = 0
        best_validation_loss = numpy.inf

    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=True
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=True, mode='min',
        save_freq='epoch'
    )
    checkpoint_object.best = best_validation_loss

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.,
        patience=early_stopping_patience_epochs, verbose=1, mode='min'
    )
    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=plateau_learning_rate_multiplier,
        patience=plateau_patience_epochs, verbose=1, mode='min',
        min_delta=0., cooldown=0
    )
    backup_object = keras.callbacks.BackupAndRestore(
        backup_dir_name, save_freq='epoch', delete_checkpoint=False
    )

    list_of_callback_objects = [
        history_object, checkpoint_object,
        early_stopping_object, plateau_object,
        backup_object
    ]

    metafile_name = find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=False
    )
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    write_metafile(
        pickle_file_name=metafile_name,
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        loss_function_string=loss_function_string,
        metric_function_strings=metric_function_strings,
        optimizer_function_string=optimizer_function_string,
        chiu_net_architecture_dict=chiu_net_architecture_dict,
        chiu_net_pp_architecture_dict=chiu_net_pp_architecture_dict,
        plateau_patience_epochs=plateau_patience_epochs,
        plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs
    )

    epochs_in_dict = numpy.unique(numpy.array(
        [l[0] for l in list(epoch_and_lead_time_to_freq.keys())],
        dtype=int
    ))
    max_epoch_in_dict = numpy.max(epochs_in_dict)

    model_lead_times_days = numpy.unique(numpy.array(
        list(training_option_dict[MODEL_LEAD_TO_TARGET_LAGS_KEY].keys()),
        dtype=int
    ))

    for this_epoch in range(initial_epoch, num_epochs):
        epoch_in_dict = min([this_epoch + 1, max_epoch_in_dict])
        training_lead_time_freqs = numpy.array([
            epoch_and_lead_time_to_freq[epoch_in_dict, l]
            for l in model_lead_times_days
        ], dtype=float)

        training_lead_time_days_to_freq = dict(zip(
            model_lead_times_days, training_lead_time_freqs
        ))
        training_option_dict[MODEL_LEAD_TO_FREQ_KEY] = (
            training_lead_time_days_to_freq
        )

        validation_lead_time_freqs = numpy.array([
            epoch_and_lead_time_to_freq[max_epoch_in_dict, l]
            for l in model_lead_times_days
        ], dtype=float)

        validation_lead_time_days_to_freq = dict(zip(
            model_lead_times_days, validation_lead_time_freqs
        ))
        validation_option_dict[MODEL_LEAD_TO_FREQ_KEY] = (
            validation_lead_time_days_to_freq
        )

        if training_option_dict[OUTER_PATCH_SIZE_DEG_KEY] is None:
            training_generator = data_generator(training_option_dict)
            validation_generator = data_generator(validation_option_dict)
        else:
            training_generator = data_generator_fast_patches(
                training_option_dict
            )
            validation_generator = data_generator_fast_patches(
                validation_option_dict
            )

        model_object.fit(
            x=training_generator,
            steps_per_epoch=num_training_batches_per_epoch,
            epochs=this_epoch + 1,
            initial_epoch=this_epoch,
            verbose=1,
            callbacks=list_of_callback_objects,
            validation_data=validation_generator,
            validation_steps=num_validation_batches_per_epoch
        )


def apply_model(
        model_object, predictor_matrices, num_examples_per_batch, verbose=True):
    """Applies trained neural net -- inference time!

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields
    S = number of ensemble members

    :param model_object: Trained neural net (instance of `keras.models.Model`).
    :param predictor_matrices: See output doc for `data_generator`.
    :param num_examples_per_batch: Batch size.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: prediction_matrix: E-by-M-by-N-by-T-by-S numpy array of predicted
        values.
    """

    # Check input args.
    for this_matrix in predictor_matrices:
        error_checking.assert_is_numpy_array_without_nan(this_matrix)

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 1)
    num_examples = predictor_matrices[0].shape[0]
    num_examples_per_batch = min([num_examples_per_batch, num_examples])

    error_checking.assert_is_boolean(verbose)

    # Do actual stuff.
    prediction_matrix = None

    for i in range(0, num_examples, num_examples_per_batch):
        first_index = i
        last_index = min([i + num_examples_per_batch, num_examples])

        if verbose:
            print('Applying model to examples {0:d}-{1:d} of {2:d}...'.format(
                first_index + 1, last_index, num_examples
            ))

        this_prediction_matrix = model_object.predict_on_batch(
            [a[first_index:last_index, ...] for a in predictor_matrices]
        )

        if prediction_matrix is None:
            dimensions = (num_examples,) + this_prediction_matrix.shape[1:]
            prediction_matrix = numpy.full(dimensions, numpy.nan)

        prediction_matrix[first_index:last_index, ...] = this_prediction_matrix

    if verbose:
        print('Have applied model to all {0:d} examples!'.format(num_examples))

    while len(prediction_matrix.shape) < 5:
        prediction_matrix = numpy.expand_dims(prediction_matrix, axis=-1)

    return prediction_matrix


def apply_model_patchwise(
        model_object, full_predictor_matrices, num_examples_per_batch,
        model_metadata_dict, patch_overlap_size_pixels,
        use_trapezoidal_weighting=False, verbose=True):
    """Does inference for neural net trained with patchwise approach.

    :param model_object: See documentation for `apply_model`.
    :param full_predictor_matrices: See documentation for `apply_model` --
        except these matrices are on the full grid.
    :param num_examples_per_batch: Same.
    :param model_metadata_dict: Dictionary in format returned by
        `read_metafile`.
    :param patch_overlap_size_pixels: Overlap between adjacent patches,
        measured in number of pixels
    :param use_trapezoidal_weighting: Boolean flag.  If True, trapezoidal
        weighting will be used, so that predictions in the center of a given
        patch are given a higher weight than predictions at the edge.
    :param verbose: See documentation for `apply_model`.
    :return: full_prediction_matrix: See documentation for `apply_model` --
        except this matrix is on the full grid.
    """

    # Check input args.
    these_dim = model_object.layers[-1].output.shape
    num_rows_in_patch = these_dim[1]
    num_columns_in_patch = these_dim[2]
    num_target_fields = these_dim[3]

    error_checking.assert_equals(num_rows_in_patch, num_columns_in_patch)
    error_checking.assert_is_boolean(use_trapezoidal_weighting)

    error_checking.assert_is_integer(patch_overlap_size_pixels)
    error_checking.assert_is_geq(patch_overlap_size_pixels, 0)
    error_checking.assert_is_less_than(
        patch_overlap_size_pixels,
        min([num_rows_in_patch, num_columns_in_patch])
    )

    if use_trapezoidal_weighting:
        half_num_rows_in_patch = int(numpy.ceil(
            float(num_rows_in_patch) / 2
        ))
        half_num_columns_in_patch = int(numpy.ceil(
            float(num_columns_in_patch) / 2
        ))
        error_checking.assert_is_geq(
            patch_overlap_size_pixels,
            max([half_num_rows_in_patch, half_num_columns_in_patch])
        )

    error_checking.assert_is_boolean(verbose)

    # Do actual stuff.
    num_rows_in_full_grid = -1
    num_columns_in_full_grid = -1

    for this_matrix in full_predictor_matrices:
        if len(this_matrix.shape) < 2:
            continue

        num_rows_in_full_grid = this_matrix.shape[1]
        num_columns_in_full_grid = this_matrix.shape[2]
        break

    patch_metalocation_dict = __init_patch_metalocation_dict(
        num_rows_in_full_grid=num_rows_in_full_grid,
        num_columns_in_full_grid=num_columns_in_full_grid,
        patch_size_pixels=num_rows_in_patch,
        patch_overlap_size_pixels=patch_overlap_size_pixels
    )

    validation_option_dict = model_metadata_dict[VALIDATION_OPTIONS_KEY]
    outer_latitude_buffer_deg = validation_option_dict[
        OUTER_LATITUDE_BUFFER_KEY
    ]
    outer_longitude_buffer_deg = validation_option_dict[
        OUTER_LONGITUDE_BUFFER_KEY
    ]

    outer_latitude_buffer_px = int(numpy.round(
        outer_latitude_buffer_deg / GRID_SPACING_DEG
    ))
    outer_longitude_buffer_px = int(numpy.round(
        outer_longitude_buffer_deg / GRID_SPACING_DEG
    ))
    assert outer_latitude_buffer_px == outer_longitude_buffer_px

    # Buffer between inner (target) and outer (predictor) domains.
    outer_patch_buffer_px = outer_latitude_buffer_px + 0

    if use_trapezoidal_weighting:
        weight_matrix = __make_trapezoidal_weight_matrix(
            patch_size_pixels=num_rows_in_patch - 2 * outer_patch_buffer_px,
            patch_overlap_size_pixels=patch_overlap_size_pixels
        )
        weight_matrix = numpy.pad(
            weight_matrix, pad_width=outer_patch_buffer_px,
            mode='constant', constant_values=0
        )
    else:
        weight_matrix = numpy.full(
            (num_rows_in_patch, num_rows_in_patch), 1, dtype=int
        )
        weight_matrix[:outer_patch_buffer_px, :] = 0
        weight_matrix[-outer_patch_buffer_px:, :] = 0
        weight_matrix[:, :outer_patch_buffer_px] = 0
        weight_matrix[:, -outer_patch_buffer_px:] = 0

    weight_matrix = numpy.expand_dims(weight_matrix, axis=0)
    weight_matrix = numpy.expand_dims(weight_matrix, axis=-1)
    weight_matrix = numpy.expand_dims(weight_matrix, axis=-1)

    num_examples = full_predictor_matrices[0].shape[0]
    these_dim = (
        num_examples, num_rows_in_full_grid, num_columns_in_full_grid,
        num_target_fields, 1
    )
    prediction_count_matrix = numpy.full(these_dim, 0, dtype=float)
    summed_prediction_matrix = None
    # summed_prediction_matrix = numpy.full(these_dim, 0.)

    while True:
        patch_metalocation_dict = __update_patch_metalocation_dict(
            patch_metalocation_dict
        )

        patch_start_row = patch_metalocation_dict[PATCH_START_ROW_KEY]
        if patch_start_row < 0:
            break

        patch_location_dict = misc_utils.determine_patch_location(
            num_rows_in_full_grid=num_rows_in_full_grid,
            num_columns_in_full_grid=num_columns_in_full_grid,
            patch_size_pixels=num_rows_in_patch,
            start_row=patch_metalocation_dict[PATCH_START_ROW_KEY],
            start_column=patch_metalocation_dict[PATCH_START_COLUMN_KEY]
        )
        pld = patch_location_dict

        i_start = pld[misc_utils.ROW_LIMITS_KEY][0]
        i_end = pld[misc_utils.ROW_LIMITS_KEY][1] + 1
        j_start = pld[misc_utils.COLUMN_LIMITS_KEY][0]
        j_end = pld[misc_utils.COLUMN_LIMITS_KEY][1] + 1

        if verbose:
            print((
                'Applying model to rows {0:d}-{1:d} of {2:d}, and '
                'columns {3:d}-{4:d} of {5:d}, in full grid...'
            ).format(
                i_start, i_end, num_rows_in_full_grid,
                j_start, j_end, num_columns_in_full_grid
            ))

        patch_predictor_matrices = []

        for this_full_pred_matrix in full_predictor_matrices:
            if len(this_full_pred_matrix.shape) < 2:
                patch_predictor_matrices.append(this_full_pred_matrix)
                continue

            patch_predictor_matrices.append(
                this_full_pred_matrix[:, i_start:i_end, j_start:j_end, ...]
            )

        patch_prediction_matrix = apply_model(
            model_object=model_object,
            predictor_matrices=patch_predictor_matrices,
            num_examples_per_batch=num_examples_per_batch,
            verbose=False
        )

        if summed_prediction_matrix is None:
            ensemble_size = patch_prediction_matrix.shape[-1]
            these_dim = (
                num_examples, num_rows_in_full_grid, num_columns_in_full_grid,
                num_target_fields, ensemble_size
            )
            summed_prediction_matrix = numpy.full(these_dim, 0.)

        summed_prediction_matrix[:, i_start:i_end, j_start:j_end, ...] += (
            weight_matrix * patch_prediction_matrix
        )
        prediction_count_matrix[:, i_start:i_end, j_start:j_end, ...] += (
            weight_matrix
        )

    if verbose:
        print('Have applied model everywhere in full grid!')

    prediction_count_matrix = prediction_count_matrix.astype(float)
    prediction_count_matrix[prediction_count_matrix < TOLERANCE] = 0.
    return summed_prediction_matrix / prediction_count_matrix
