"""Helper methods for training a neural network to predict fire weather."""

import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.io import gfs_io
from ml_for_wildfire_wpo.io import canadian_fwi_io
from ml_for_wildfire_wpo.utils import gfs_utils
from ml_for_wildfire_wpo.utils import canadian_fwi_utils

DATE_FORMAT = '%Y%m%d'

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
    outer_latitude_limits_deg_n = option_dict[INNER_LATITUDE_LIMITS_KEY] + (
        numpy.array([
            -1 * option_dict[OUTER_LATITUDE_BUFFER_KEY],
            option_dict[OUTER_LATITUDE_BUFFER_KEY]
        ])
    )
    error_checking.assert_is_valid_lat_numpy_array(outer_latitude_limits_deg_n)

    error_checking.assert_is_greater(option_dict[OUTER_LONGITUDE_BUFFER_KEY], 0)

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
            era5_utils.check_constant_field_name(this_field_name)

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
    error_checking.assert_is_geq(option_dict[BATCH_SIZE_KEY], 8)

    error_checking.assert_is_not_nan(option_dict[SENTINEL_VALUE_KEY])

    return option_dict


def data_generator(option_dict):
    """Generates both training and validation for neural network.

    Generators should be used only at training time, not at inference time.

    This particular generator should be used for neural networks that always see
    the same geographic domain, with the bounding box for this domain specified
    in the first few input args of the dictionary.

    E = number of examples per batch = "batch size"
    M = number of grid rows in outer domain
    N = number of grid rows in inner domain
    FFF = number of 3-D GFS predictor fields
    P = number of GFS pressure levels
    FF = number of 2-D predictor fields (including GFS and ERA5-constant)

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

    predictor_matrices[0]: E-by-M-by-N-by-P-by-FFF numpy array of 3-D
        predictors.
    predictor_matrices[1]: E-by-M-by-N-by-FF numpy array of 2-D predictors.
    """

    # TODO(thunderhoser): Allow multiple lead times for target.

    # TODO(thunderhoser): Deal with normalization.  On-the-fly normalization would be easy but slow.  Pre-fab normalization (in the input files) would be tricky, because the target variable y should be normalized when lagged versions of y are used in the predictors but not when y itself is used as the target.

    # TODO(thunderhoser): Deal with expanding target grid to size of predictor grid, then creating mask.

    # TODO(thunderhoser): No choice but to read land-sea mask for use in loss function (masking target domain).

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

    outer_latitude_limits_deg_n = inner_latitude_limits_deg_n + numpy.array([
        -1 * outer_latitude_buffer_deg, outer_latitude_buffer_deg
    ])

    # TODO(thunderhoser): This might fail.
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
    gfs_file_init_date_strings = [
        gfs_io.file_name_to_date(f) for f in gfs_file_names
    ]
    gfs_file_init_dates_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, gfs_io.DATE_FORMAT)
        for t in gfs_file_init_date_strings
    ], dtype=int)

    lag_dates_unix_sec = numpy.concatenate([
        d - target_lag_times_days for d in gfs_file_init_dates_unix_sec
    ])
    lead_dates_unix_sec = numpy.array(
        [d + target_lead_time_days for d in gfs_file_init_dates_unix_sec],
        dtype=int
    )
    target_file_dates_unix_sec = numpy.unique(
        numpy.concatenate([lag_dates_unix_sec, lead_dates_unix_sec])
    )
    target_file_date_strings = [
        time_conversion.unix_sec_to_string(t, canadian_fwi_io.DATE_FORMAT)
        for t in target_file_dates_unix_sec
    ]

    target_file_names = [
        canadian_fwi_io.find_file(
            directory_name=target_dir_name,
            valid_date_string=d, raise_error_if_missing=True
        )
        for d in target_file_date_strings
    ]

    shuffled_gfs_file_indices = numpy.linspace(
        0, len(gfs_file_names) - 1, num=len(gfs_file_names), dtype=int
    )
    shuffled_file_index = len(gfs_file_names)

    while True:
        predictor_matrix_3d = None
        predictor_matrix_2d = None
        target_matrix = None
        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if shuffled_file_index == len(gfs_file_names):
                shuffled_file_index = 0

            gfs_file_index = shuffled_gfs_file_indices[shuffled_file_index]

            print('Reading data from: "{0:s}"...'.format(
                gfs_file_names[gfs_file_index]
            ))
            gfs_table_xarray = gfs_io.read_file(gfs_file_names[gfs_file_index])

            # TODO(thunderhoser): I don't know if subsetting the whole table
            # first -- before extracting desired fields -- is more efficient.
            desired_row_indices = gfs_utils.desired_latitudes_to_rows(
                gfs_table_xarray=gfs_table_xarray,
                start_latitude_deg_n=outer_latitude_limits_deg_n[0],
                end_latitude_deg_n=outer_latitude_limits_deg_n[1]
            )
            desired_column_indices = gfs_utils.desired_longitudes_to_columns(
                gfs_table_xarray=gfs_table_xarray,
                start_longitude_deg_e=outer_longitude_limits_deg_e[0],
                end_longitude_deg_e=outer_longitude_limits_deg_e[1]
            )
            gfs_table_xarray = gfs_utils.subset_by_row(
                gfs_table_xarray=gfs_table_xarray,
                desired_row_indices=desired_row_indices
            )
            gfs_table_xarray = gfs_utils.subset_by_column(
                gfs_table_xarray=gfs_table_xarray,
                desired_column_indices=desired_column_indices
            )
            gfs_table_xarray = gfs_utils.subset_by_forecast_hour(
                gfs_table_xarray=gfs_table_xarray,
                desired_forecast_hours=gfs_predictor_lead_times_hours
            )

            this_predictor_matrix_2d = numpy.stack([
                gfs_utils.get_field(
                    gfs_table_xarray=gfs_table_xarray,
                    field_name=f
                )
                for f in gfs_2d_field_names
            ], axis=-1)

            this_predictor_matrix_3d = numpy.stack([
                gfs_utils.get_field(
                    gfs_table_xarray=gfs_table_xarray,
                    field_name=f,
                    pressure_levels_mb=gfs_pressure_levels_mb
                )
                for f in gfs_3d_field_names
            ], axis=-1)

            gfs_file_init_dates_unix_sec[gfs_file_index] - target_lag_times_days

            # TODO(thunderhoser): Sort times in this method.
            these_target_file_names = _find_target_files_needed_1example(
                gfs_file_name=gfs_file_names[gfs_file_index],
                target_dir_name=target_dir_name,
                target_lead_times_days=-1 * target_lag_times_days
            )

            for this_target_file_name in these_target_file_names:
                print('Reading data from: "{0:s}"...'.format(
                    this_target_file_name
                ))
                this_field_matrix = canadian_fwi_utils.get_field(
                    fwi_table_xarray=
                    canadian_fwi_io.read_file(this_target_file_name),
                    field_name=target_field_name
                )
                this_predictor_matrix_2d = numpy.concatenate((
                    this_predictor_matrix_2d,
                    numpy.expand_dims(this_field_matrix, axis=-1)
                ))
