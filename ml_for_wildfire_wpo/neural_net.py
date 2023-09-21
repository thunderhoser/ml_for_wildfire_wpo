"""Helper methods for training a neural network to predict fire weather."""

import os
import sys
import random
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import number_rounding
import error_checking
import gfs_io
import era5_constant_io
import canadian_fwi_io
import misc_utils
import gfs_utils
import era5_constant_utils
import canadian_fwi_utils

DATE_FORMAT = '%Y%m%d'
GRID_SPACING_DEG = 0.25

INNER_LATITUDE_LIMITS_KEY = 'inner_latitude_limits_deg_n'
INNER_LONGITUDE_LIMITS_KEY = 'inner_longitude_limits_deg_e'
OUTER_LATITUDE_BUFFER_KEY = 'outer_latitude_buffer_deg'
OUTER_LONGITUDE_BUFFER_KEY = 'outer_longitude_buffer_deg'
INIT_DATE_LIMITS_KEY = 'init_date_limit_strings'
GFS_PREDICTOR_FIELDS_KEY = 'gfs_predictor_field_names'
GFS_PRESSURE_LEVELS_KEY = 'gfs_pressure_levels_mb'
GFS_PREDICTOR_LEADS_KEY = 'gfs_predictor_lead_times_hours'
GFS_DIRECTORY_KEY = 'gfs_directory_name'
ERA5_CONSTANT_PREDICTOR_FIELDS_KEY = 'era5_constant_predictor_field_names'
ERA5_CONSTANT_FILE_KEY = 'era5_constant_file_name'
TARGET_FIELD_KEY = 'target_field_name'
TARGET_LEAD_TIME_KEY = 'target_lead_time_days'
TARGET_LAG_TIMES_KEY = 'target_lag_times_days'
TARGET_CUTOFFS_KEY = 'target_cutoffs_for_classifn'
TARGET_DIRECTORY_KEY = 'target_dir_name'
BATCH_SIZE_KEY = 'num_examples_per_batch'
SENTINEL_VALUE_KEY = 'sentinel_value'

DEFAULT_GENERATOR_OPTION_DICT = {
    INNER_LATITUDE_LIMITS_KEY: numpy.array([17, 73], dtype=float),
    INNER_LONGITUDE_LIMITS_KEY: numpy.array([171, -65], dtype=float),
    OUTER_LATITUDE_BUFFER_KEY: 5.,
    OUTER_LONGITUDE_BUFFER_KEY: 5.,
    TARGET_LAG_TIMES_KEY: numpy.array([1], dtype=int),
    TARGET_CUTOFFS_KEY: None,
    # SENTINEL_VALUE_KEY: -10.
}


def _check_generator_args(option_dict):
    """Checks input arguments for generator.

    :param option_dict: See doc for `data_generator`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    # TODO(thunderhoser): If the model uses image chips from different regions,
    # the lat/long input args will be gone.
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

    error_checking.assert_is_numpy_array(
        option_dict[GFS_PREDICTOR_LEADS_KEY], num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[GFS_PREDICTOR_LEADS_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[GFS_PREDICTOR_LEADS_KEY], 0
    )
    option_dict[GFS_PREDICTOR_LEADS_KEY] = numpy.sort(
        option_dict[GFS_PREDICTOR_LEADS_KEY]
    )

    error_checking.assert_directory_exists(option_dict[GFS_DIRECTORY_KEY])

    if option_dict[ERA5_CONSTANT_PREDICTOR_FIELDS_KEY] is not None:
        error_checking.assert_is_string_list(
            option_dict[ERA5_CONSTANT_PREDICTOR_FIELDS_KEY]
        )
        for this_field_name in option_dict[ERA5_CONSTANT_PREDICTOR_FIELDS_KEY]:
            era5_constant_utils.check_field_name(this_field_name)

    error_checking.assert_file_exists(option_dict[ERA5_CONSTANT_FILE_KEY])

    canadian_fwi_utils.check_field_name(option_dict[TARGET_FIELD_KEY])

    error_checking.assert_is_integer(option_dict[TARGET_LEAD_TIME_KEY])
    error_checking.assert_is_geq(option_dict[TARGET_LEAD_TIME_KEY], 0)

    error_checking.assert_is_numpy_array(
        option_dict[TARGET_LAG_TIMES_KEY], num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[TARGET_LAG_TIMES_KEY]
    )
    error_checking.assert_is_greater_numpy_array(
        option_dict[TARGET_LAG_TIMES_KEY], 0
    )

    if option_dict[TARGET_CUTOFFS_KEY] is not None:
        error_checking.assert_is_numpy_array(
            option_dict[TARGET_CUTOFFS_KEY], num_dimensions=1
        )
        error_checking.assert_is_greater_numpy_array(
            option_dict[TARGET_CUTOFFS_KEY], 0
        )
        assert numpy.all(numpy.isfinite(option_dict[TARGET_CUTOFFS_KEY]))
        error_checking.assert_is_greater_numpy_array(
            numpy.diff(option_dict[TARGET_CUTOFFS_KEY]), 0
        )

    error_checking.assert_directory_exists(option_dict[TARGET_DIRECTORY_KEY])

    error_checking.assert_is_integer(option_dict[BATCH_SIZE_KEY])
    # error_checking.assert_is_geq(option_dict[BATCH_SIZE_KEY], 8)

    error_checking.assert_is_not_nan(option_dict[SENTINEL_VALUE_KEY])

    return option_dict


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
        gfs_init_date_unix_sec + target_lead_times_days
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
    :return: predictor_matrix: None or an M-by-N-by-L-by-P-by-F numpy array.
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


def _discretize_targets(target_matrix, cutoff_values):
    """Discretizes targets for classification task.

    E = number of data examples
    M = number of rows in grid
    N = number of columns in grid
    K = number of classes

    :param target_matrix: E-by-M-by-N numpy array of physical target values (FWI
        values).
    :param cutoff_values: length-(K - 1) numpy array of cutoff values.  0 and
        infinity will automatically be added to the beginning and end of this
        array, respectively.
    :return: discretized_target_matrix: E-by-M-by-N-by-K numpy array of ones and
        zeros.
    """

    full_cutoff_values = numpy.concatenate((
        numpy.array([0.]),
        cutoff_values,
        numpy.array([numpy.inf])
    ))
    num_classes = len(full_cutoff_values) - 1

    discretized_target_matrix = numpy.full(
        target_matrix.shape + (num_classes,), -1, dtype=int
    )

    for k in range(num_classes):
        these_flags = numpy.logical_and(
            target_matrix >= full_cutoff_values[k],
            target_matrix < full_cutoff_values[k + 1]
        )
        discretized_target_matrix[..., k][these_flags] = 1
        discretized_target_matrix[..., k][these_flags == False] = 0

    assert numpy.all(discretized_target_matrix >= 0)
    return discretized_target_matrix


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


def _get_target_field(
        target_file_name, desired_row_indices, desired_column_indices,
        field_name):
    """Reads target field from one file.

    M = number of rows in grid
    N = number of columns in grid

    :param target_file_name: Path to input file.
    :param desired_row_indices: length-M numpy array of indices.
    :param desired_column_indices: length-N numpy array of indices.
    :param field_name: Field name.
    :return: data_matrix: M-by-N numpy array of data values.
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
    return canadian_fwi_utils.get_field(
        fwi_table_xarray=fwi_table_xarray, field_name=field_name
    )


def _read_mask_from_era5(
        era5_constant_file_name, inner_latitude_limits_deg_n,
        inner_longitude_limits_deg_e, outer_latitude_buffer_deg,
        outer_longitude_buffer_deg):
    """Reads mask from ERA5 file.

    This mask is a combination of the land/sea mask and also the buffer between
    the inner and outer domains.  The mask is binary, with 1 for land areas in
    the inner domain and 0 everywhere else.

    :param era5_constant_file_name: See documentation for `data_generator`.
    :param inner_latitude_limits_deg_n: Same.
    :param inner_longitude_limits_deg_e: Same.
    :param outer_latitude_buffer_deg: Same.
    :param outer_longitude_buffer_deg: Same.
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
    ect = era5_constant_utils.subset_by_row(
        era5_constant_table_xarray=ect,
        desired_row_indices=desired_row_indices
    )
    ect = era5_constant_utils.subset_by_column(
        era5_constant_table_xarray=ect,
        desired_column_indices=desired_column_indices
    )

    mask_matrix = era5_constant_utils.get_field(
        era5_constant_table_xarray=ect,
        field_name=era5_constant_utils.LAND_SEA_MASK_NAME
    )
    mask_matrix = (mask_matrix > 0.05).astype(int)

    mask_matrix = _pad_inner_to_outer_domain(
        data_matrix=mask_matrix,
        outer_latitude_buffer_deg=outer_latitude_buffer_deg,
        outer_longitude_buffer_deg=outer_longitude_buffer_deg,
        is_example_axis_present=False, fill_value=0
    )

    return numpy.round(mask_matrix).astype(int)


def data_generator(option_dict):
    """Generates both training and validation for neural network.

    Generators should be used only at training time, not at inference time.

    This particular generator should be used for neural networks that always see
    the same geographic domain, with the bounding box for this domain specified
    in the first few input args of the dictionary.

    E = number of examples per batch = "batch size"
    M = number of grid rows in outer domain
    N = number of grid columns in outer domain
    L = number of GFS forecast hours (lead times)
    P = number of GFS pressure levels
    FFF = number of 3-D GFS predictor fields
    FF = number of 2-D predictor fields (including GFS, ERA5-constant, and
         lagged target fields)

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
    option_dict["gfs_predictor_lead_times_hours"]: 1-D numpy array with lead
        times to be used for GFS predictors.
    option_dict["gfs_directory_name"]: Name of directory with GFS data.  Files
        therein will be found by `gfs_io.find_file` and read by
        `gfs_io.read_file`.
    option_dict["era5_constant_predictor_field_names"]: 1-D list with names of
        ERA5 constant fields to be used as predictors.  If you do not want such
        predictors, make this None.
    option_dict["era5_constant_file_name"]: Path to file with ERA5 constants.
        Will be read by `era5_constant_io.read_file`.
    option_dict["target_field_name"]: Name of target field (a fire-weather
        index).
    option_dict["target_lead_time_days"]: Lead time for target field.
    option_dict["target_lag_times_days"]: 1-D numpy array with lag times to be
        used as predictors.  For example, if the target field is FFMC and the
        lag times are {1, 2, 3} days, this means that FFMC fields from {1, 2, 3}
        days ago will be in the predictors.  Just the 1-day lag time should be
        enough, though.
    option_dict["target_cutoffs_for_classifn"]: 1-D numpy array of cutoffs for
        converting regression problem to classification problem.  For example,
        if this array is [20, 30], the three classes will be 0-20, 20-30, and
        30-infinity.  If you want to do regression, make this None.
    option_dict["target_dir_name"]: Name of directory with target variable.
        Files therein will be found by `canadian_fwo_io.find_file` and read by
        `canadian_fwo_io.read_file`.
    option_dict["num_examples_per_batch"]: Number of data examples per batch,
        usually just called "batch size".
    option_dict["sentinel_value"]: All NaN will be replaced with this value.

    :return: predictor_matrices: List with the following items.  Either one may
        be missing.

    predictor_matrices[0]: E-by-M-by-N-by-P-by-L-by-FFF numpy array of 3-D
        predictors.
    predictor_matrices[1]: E-by-M-by-N-by-L-by-FF numpy array of 2-D predictors.
    """

    # TODO(thunderhoser): Allow multiple lead times for target.

    # TODO(thunderhoser): Deal with normalization.  On-the-fly normalization
    # would be easy but slow.  Pre-fab normalization (in the input files) would
    # be tricky, because the target variable y should be normalized when lagged
    # versions of y are used in the predictors but not when y itself is used as
    # the target.

    # Everywhere in the U.S. -- even parts of Alaska west of the International
    # Date Line -- has a time zone behind UTC.  Thus, in the predictors, the GFS
    # run init 00Z on day D should be matched with FWI maps on day (D - 1) or
    # earlier -- never with FWI maps on day D.

    option_dict = _check_generator_args(option_dict)

    inner_latitude_limits_deg_n = option_dict[INNER_LATITUDE_LIMITS_KEY]
    inner_longitude_limits_deg_e = option_dict[INNER_LONGITUDE_LIMITS_KEY]
    outer_latitude_buffer_deg = option_dict[OUTER_LATITUDE_BUFFER_KEY]
    outer_longitude_buffer_deg = option_dict[OUTER_LONGITUDE_BUFFER_KEY]
    init_date_limit_strings = option_dict[INIT_DATE_LIMITS_KEY]
    gfs_predictor_field_names = option_dict[GFS_PREDICTOR_FIELDS_KEY]
    gfs_pressure_levels_mb = option_dict[GFS_PRESSURE_LEVELS_KEY]
    gfs_predictor_lead_times_hours = option_dict[GFS_PREDICTOR_LEADS_KEY]
    gfs_directory_name = option_dict[GFS_DIRECTORY_KEY]
    era5_constant_predictor_field_names = option_dict[
        ERA5_CONSTANT_PREDICTOR_FIELDS_KEY
    ]
    era5_constant_file_name = option_dict[ERA5_CONSTANT_FILE_KEY]
    target_field_name = option_dict[TARGET_FIELD_KEY]
    target_lead_time_days = option_dict[TARGET_LEAD_TIME_KEY]
    target_lag_times_days = option_dict[TARGET_LAG_TIMES_KEY]
    target_cutoffs_for_classifn = option_dict[TARGET_CUTOFFS_KEY]
    target_dir_name = option_dict[TARGET_DIRECTORY_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    sentinel_value = option_dict[SENTINEL_VALUE_KEY]

    # TODO(thunderhoser): The longitude command below might fail.
    outer_latitude_limits_deg_n = inner_latitude_limits_deg_n + numpy.array([
        -1 * outer_latitude_buffer_deg, outer_latitude_buffer_deg
    ])
    outer_longitude_limits_deg_e = inner_longitude_limits_deg_e + numpy.array([
        -1 * outer_longitude_buffer_deg, outer_longitude_buffer_deg
    ])

    gfs_2d_field_names = [
        f for f in gfs_predictor_field_names
        if f in gfs_utils.ALL_2D_FIELD_NAMES
    ]
    gfs_3d_field_names = [
        f for f in gfs_predictor_field_names
        if f in gfs_utils.ALL_3D_FIELD_NAMES
    ]

    gfs_file_names = gfs_io.find_files_for_period(
        directory_name=gfs_directory_name,
        first_init_date_string=init_date_limit_strings[0],
        last_init_date_string=init_date_limit_strings[1],
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )
    random.shuffle(gfs_file_names)

    print('Reading data from: "{0:s}"...'.format(era5_constant_file_name))
    era5_constant_table_xarray = era5_constant_io.read_file(
        era5_constant_file_name
    )
    ect = era5_constant_table_xarray

    desired_era5c_row_indices = misc_utils.desired_latitudes_to_rows(
        grid_latitudes_deg_n=
        ect.coords[era5_constant_utils.LATITUDE_DIM].values,
        start_latitude_deg_n=outer_latitude_limits_deg_n[0],
        end_latitude_deg_n=outer_latitude_limits_deg_n[1]
    )
    desired_era5c_column_indices = misc_utils.desired_longitudes_to_columns(
        grid_longitudes_deg_e=
        ect.coords[era5_constant_utils.LONGITUDE_DIM].values,
        start_longitude_deg_e=outer_longitude_limits_deg_e[0],
        end_longitude_deg_e=outer_longitude_limits_deg_e[1]
    )
    ect = era5_constant_utils.subset_by_row(
        era5_constant_table_xarray=ect,
        desired_row_indices=desired_era5c_row_indices
    )
    ect = era5_constant_utils.subset_by_column(
        era5_constant_table_xarray=ect,
        desired_column_indices=desired_era5c_column_indices
    )

    if era5_constant_predictor_field_names is None:
        era5_constant_matrix = None
    else:
        era5_constant_matrix = numpy.stack([
            era5_constant_utils.get_field(
                era5_constant_table_xarray=ect, field_name=f
            )
            for f in era5_constant_predictor_field_names
        ], axis=-1)

        era5_constant_matrix = numpy.repeat(
            numpy.expand_dims(era5_constant_matrix, axis=0),
            axis=0, repeats=num_examples_per_batch
        )

    mask_matrix = _read_mask_from_era5(
        era5_constant_file_name=era5_constant_file_name,
        inner_latitude_limits_deg_n=inner_latitude_limits_deg_n,
        inner_longitude_limits_deg_e=inner_longitude_limits_deg_e,
        outer_latitude_buffer_deg=outer_latitude_buffer_deg,
        outer_longitude_buffer_deg=outer_longitude_buffer_deg
    )
    mask_matrix = numpy.repeat(
        numpy.expand_dims(mask_matrix, axis=0),
        axis=0, repeats=num_examples_per_batch
    )
    mask_matrix = numpy.expand_dims(mask_matrix, axis=-1)

    gfs_file_index = len(gfs_file_names)
    desired_gfs_row_indices = numpy.array([], dtype=int)
    desired_gfs_column_indices = numpy.array([], dtype=int)
    desired_target_row_indices = numpy.array([], dtype=int)
    desired_target_column_indices = numpy.array([], dtype=int)

    while True:
        predictor_matrix_3d = None
        gfs_predictor_matrix_2d = None
        lagged_predictor_matrix_2d = None
        target_matrix = None
        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if gfs_file_index == len(gfs_file_names):
                random.shuffle(gfs_file_names)
                gfs_file_index = 0

            print('Reading data from: "{0:s}"...'.format(
                gfs_file_names[gfs_file_index]
            ))
            gfs_table_xarray = gfs_io.read_file(gfs_file_names[gfs_file_index])

            if len(desired_gfs_row_indices) == 0:
                desired_gfs_row_indices = misc_utils.desired_latitudes_to_rows(
                    grid_latitudes_deg_n=
                    gfs_table_xarray.coords[gfs_utils.LATITUDE_DIM].values,
                    start_latitude_deg_n=outer_latitude_limits_deg_n[0],
                    end_latitude_deg_n=outer_latitude_limits_deg_n[1]
                )
                desired_gfs_column_indices = (
                    misc_utils.desired_longitudes_to_columns(
                        grid_longitudes_deg_e=
                        gfs_table_xarray.coords[gfs_utils.LONGITUDE_DIM].values,
                        start_longitude_deg_e=outer_longitude_limits_deg_e[0],
                        end_longitude_deg_e=outer_longitude_limits_deg_e[1]
                    )
                )

            # TODO(thunderhoser): I don't know if subsetting the whole table
            # first -- before extracting desired fields -- is more efficient.
            gfs_table_xarray = gfs_utils.subset_by_forecast_hour(
                gfs_table_xarray=gfs_table_xarray,
                desired_forecast_hours=gfs_predictor_lead_times_hours
            )
            gfs_table_xarray = gfs_utils.subset_by_row(
                gfs_table_xarray=gfs_table_xarray,
                desired_row_indices=desired_gfs_row_indices
            )
            gfs_table_xarray = gfs_utils.subset_by_column(
                gfs_table_xarray=gfs_table_xarray,
                desired_column_indices=desired_gfs_column_indices
            )

            this_gfs_predictor_matrix_2d = _get_2d_gfs_fields(
                gfs_table_xarray=gfs_table_xarray,
                field_names=gfs_2d_field_names
            )
            this_predictor_matrix_3d = _get_3d_gfs_fields(
                gfs_table_xarray=gfs_table_xarray,
                field_names=gfs_3d_field_names,
                pressure_levels_mb=gfs_pressure_levels_mb
            )

            target_file_names = _find_target_files_needed_1example(
                gfs_init_date_string=
                gfs_io.file_name_to_date(gfs_file_names[gfs_file_index]),
                target_dir_name=target_dir_name,
                target_lead_times_days=-1 * target_lag_times_days
            )

            if len(desired_target_row_indices) == 0:
                fwit = canadian_fwi_io.read_file(target_file_names[0])
                desired_target_row_indices = (
                    misc_utils.desired_latitudes_to_rows(
                        grid_latitudes_deg_n=
                        fwit.coords[canadian_fwi_utils.LATITUDE_DIM].values,
                        start_latitude_deg_n=inner_latitude_limits_deg_n[0],
                        end_latitude_deg_n=inner_latitude_limits_deg_n[1]
                    )
                )
                desired_target_column_indices = (
                    misc_utils.desired_longitudes_to_columns(
                        grid_longitudes_deg_e=
                        fwit.coords[canadian_fwi_utils.LONGITUDE_DIM].values,
                        start_longitude_deg_e=inner_longitude_limits_deg_e[0],
                        end_longitude_deg_e=inner_longitude_limits_deg_e[1]
                    )
                )

            this_lagged_predictor_matrix_2d = numpy.stack([
                _get_target_field(
                    target_file_name=f,
                    desired_row_indices=desired_target_row_indices,
                    desired_column_indices=desired_target_column_indices,
                    field_name=target_field_name
                )
                for f in target_file_names
            ], axis=-1)

            target_file_name = _find_target_files_needed_1example(
                gfs_init_date_string=
                gfs_io.file_name_to_date(gfs_file_names[gfs_file_index]),
                target_dir_name=target_dir_name,
                target_lead_times_days=
                numpy.array([target_lead_time_days], dtype=int)
            )[0]

            this_target_matrix = _get_target_field(
                target_file_name=target_file_name,
                desired_row_indices=desired_target_row_indices,
                desired_column_indices=desired_target_column_indices,
                field_name=target_field_name
            )
            assert not numpy.any(numpy.isnan(this_target_matrix))

            if this_predictor_matrix_3d is not None:
                if predictor_matrix_3d is None:
                    these_dim = (
                        (num_examples_per_batch,) +
                        this_predictor_matrix_3d.shape[1:]
                    )
                    predictor_matrix_3d = numpy.full(these_dim, numpy.nan)

                predictor_matrix_3d[num_examples_in_memory, ...] = (
                    this_predictor_matrix_3d
                )

            if this_gfs_predictor_matrix_2d is not None:
                if gfs_predictor_matrix_2d is None:
                    these_dim = (
                        (num_examples_per_batch,) +
                        this_gfs_predictor_matrix_2d.shape[1:]
                    )
                    gfs_predictor_matrix_2d = numpy.full(these_dim, numpy.nan)

                gfs_predictor_matrix_2d[num_examples_in_memory, ...] = (
                    this_gfs_predictor_matrix_2d
                )

            if lagged_predictor_matrix_2d is None:
                these_dim = (
                    (num_examples_per_batch,) +
                    this_lagged_predictor_matrix_2d.shape[1:]
                )
                lagged_predictor_matrix_2d = numpy.full(these_dim, numpy.nan)

            lagged_predictor_matrix_2d[num_examples_in_memory, ...] = (
                this_lagged_predictor_matrix_2d
            )

            if target_matrix is None:
                these_dim = (
                    (num_examples_per_batch,) + this_target_matrix.shape[1:]
                )
                target_matrix = numpy.full(these_dim, numpy.nan)

            target_matrix[num_examples_in_memory, ...] = this_target_matrix
            num_examples_in_memory += 1

        predictor_matrix_2d = _pad_inner_to_outer_domain(
            data_matrix=lagged_predictor_matrix_2d,
            outer_latitude_buffer_deg=outer_latitude_buffer_deg,
            outer_longitude_buffer_deg=outer_longitude_buffer_deg,
            is_example_axis_present=True, fill_value=sentinel_value
        )

        if era5_constant_matrix is not None:
            predictor_matrix_2d = numpy.concatenate(
                (predictor_matrix_2d, era5_constant_matrix), axis=-1
            )
        if gfs_predictor_matrix_2d is not None:
            predictor_matrix_2d = numpy.concatenate(
                (predictor_matrix_2d, gfs_predictor_matrix_2d), axis=-1
            )

        predictor_matrix_2d[numpy.isnan(predictor_matrix_2d)] = sentinel_value
        print('Shape of 2-D predictor matrix: {0:s}'.format(
            str(predictor_matrix_2d.shape)
        ))

        if predictor_matrix_3d is not None:
            predictor_matrix_3d[
                numpy.isnan(predictor_matrix_3d)
            ] = sentinel_value

            print('Shape of 3-D predictor matrix: {0:s}'.format(
                str(predictor_matrix_3d.shape)
            ))

        target_matrix = _pad_inner_to_outer_domain(
            data_matrix=target_matrix,
            outer_latitude_buffer_deg=outer_latitude_buffer_deg,
            outer_longitude_buffer_deg=outer_longitude_buffer_deg,
            is_example_axis_present=True, fill_value=0.
        )

        if target_cutoffs_for_classifn is None:
            target_matrix = numpy.expand_dims(target_matrix, axis=-1)
        else:
            target_matrix = _discretize_targets(
                target_matrix=target_matrix,
                cutoff_values=target_cutoffs_for_classifn
            )

        target_matrix_with_mask = numpy.concatenate(
            (target_matrix, mask_matrix), axis=-1
        )
        predictor_matrices = [
            m for m in [predictor_matrix_3d, predictor_matrix_2d]
            if m is not None
        ]

        print((
            'Shape of target matrix (including land mask as last channel): '
            '{0:s}'
        ).format(
            str(target_matrix_with_mask.shape)
        ))

        # predictor_matrices = [p.astype('float16') for p in predictor_matrices]
        yield predictor_matrices, target_matrix_with_mask
