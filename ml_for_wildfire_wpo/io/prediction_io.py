"""Input/output methods for predictions."""

import os
import copy
import numpy
import xarray
import netCDF4
from geopy.distance import geodesic
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
DATE_FORMAT = '%Y%m%d'

ROW_DIM = 'grid_row'
COLUMN_DIM = 'grid_column'
FIELD_DIM = 'field'
FIELD_CHAR_DIM = 'field_char'
ENSEMBLE_MEMBER_DIM = 'ensemble_member'
DUMMY_ENSEMBLE_MEMBER_DIM = 'dummy_ensemble_member'

MODEL_FILE_KEY = 'model_file_name'
ISOTONIC_MODEL_FILE_KEY = 'isotonic_model_file_name'
UNCERTAINTY_CALIB_MODEL_FILE_KEY = 'uncertainty_calib_model_file_name'
INIT_DATE_KEY = 'init_date_string'

TARGET_KEY = 'target'
PREDICTION_KEY = 'prediction'
WEIGHT_KEY = 'evaluation_weight'
LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'
FIELD_NAME_KEY = 'field_name'

METRES_TO_DEGREES_LAT = 1. / (60. * 1852)


def find_file(directory_name, init_date_string, raise_error_if_missing=True):
    """Finds NetCDF file with predictions initialized on one day (at 00Z).

    :param directory_name: Path to input directory.
    :param init_date_string: Initialization date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: prediction_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    _ = time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)
    error_checking.assert_is_boolean(raise_error_if_missing)

    prediction_file_name = '{0:s}/predictions_{1:s}.nc'.format(
        directory_name, init_date_string
    )

    if raise_error_if_missing and not os.path.isfile(prediction_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            prediction_file_name
        )
        raise ValueError(error_string)

    return prediction_file_name


def find_files_for_period(
        directory_name, first_init_date_string, last_init_date_string,
        raise_error_if_any_missing=False, raise_error_if_all_missing=True):
    """Finds files with predictions over a time period, one per daily model run.

    :param directory_name: Path to input directory.
    :param first_init_date_string: First date in period (format "yyyymmdd").
    :param last_init_date_string: Last date in period (format "yyyymmdd").
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :return: prediction_file_names: 1-D list of paths to NetCDF files with
        predictions, one per daily model run.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    init_date_strings = time_conversion.get_spc_dates_in_range(
        first_init_date_string, last_init_date_string
    )

    prediction_file_names = []

    for this_date_string in init_date_strings:
        this_file_name = find_file(
            directory_name=directory_name,
            init_date_string=this_date_string,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if os.path.isfile(this_file_name):
            prediction_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(prediction_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from dates {1:s} to '
            '{2:s}.'
        ).format(
            directory_name, first_init_date_string, last_init_date_string
        )
        raise ValueError(error_string)

    return prediction_file_names


def file_name_to_date(prediction_file_name):
    """Parses date from name of prediction file.

    :param prediction_file_name: File path.
    :return: init_date_string: Initialization date (format "yyyymmdd").
    """

    pathless_file_name = os.path.split(prediction_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    init_date_string = extensionless_file_name.split('_')[1]

    _ = time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)

    return init_date_string


def read_file(netcdf_file_name):
    """Reads predictions from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: prediction_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    prediction_table_xarray = xarray.open_dataset(netcdf_file_name)
    target_field_names = [
        f.decode('utf-8') for f in
        prediction_table_xarray[FIELD_NAME_KEY].values
    ]

    if ENSEMBLE_MEMBER_DIM not in prediction_table_xarray.dims:
        prediction_table_xarray = prediction_table_xarray.assign_coords({
            ENSEMBLE_MEMBER_DIM: numpy.array([0], dtype=int)
        })

        these_dim = (ROW_DIM, COLUMN_DIM, FIELD_DIM, ENSEMBLE_MEMBER_DIM)
        these_data = numpy.expand_dims(
            prediction_table_xarray[PREDICTION_KEY].values, axis=-1
        )
        prediction_table_xarray = prediction_table_xarray.assign({
            PREDICTION_KEY: (these_dim, these_data)
        })

    if ISOTONIC_MODEL_FILE_KEY not in prediction_table_xarray.attrs:
        prediction_table_xarray.attrs[ISOTONIC_MODEL_FILE_KEY] = ''
    if UNCERTAINTY_CALIB_MODEL_FILE_KEY not in prediction_table_xarray.attrs:
        prediction_table_xarray.attrs[UNCERTAINTY_CALIB_MODEL_FILE_KEY] = ''

    if prediction_table_xarray.attrs[ISOTONIC_MODEL_FILE_KEY] == '':
        prediction_table_xarray.attrs[ISOTONIC_MODEL_FILE_KEY] = None
    if prediction_table_xarray.attrs[UNCERTAINTY_CALIB_MODEL_FILE_KEY] == '':
        prediction_table_xarray.attrs[UNCERTAINTY_CALIB_MODEL_FILE_KEY] = None

    return prediction_table_xarray.assign({
        FIELD_NAME_KEY: (
            prediction_table_xarray[FIELD_NAME_KEY].dims,
            target_field_names
        )
    })


def write_file(
        netcdf_file_name, target_matrix_with_weights, prediction_matrix,
        grid_latitudes_deg_n, grid_longitudes_deg_e, field_names,
        init_date_string, model_file_name, isotonic_model_file_name,
        uncertainty_calib_model_file_name):
    """Writes predictions to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields
    S = number of ensemble members

    :param netcdf_file_name: Path to output file.
    :param target_matrix_with_weights: M-by-N-by-(T + 1) numpy array, where
        target_matrix_with_weights[..., -1] contains weights and
        target_matrix_with_weights[..., :-1] contains target values.
    :param prediction_matrix: M-by-N-by-T-by-S numpy array of predicted target
        values.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param field_names: length-T list of field names.
    :param init_date_string: Initialization date (format "yyyymmdd").
    :param model_file_name: Path to file with trained model.
    :param isotonic_model_file_name: Path to file with isotonic-regression model
        used to bias-correct ensemble means.  If N/A, make this None.
    :param uncertainty_calib_model_file_name: Path to file with uncertainty-
        calibration model used to bias-correct ensemble spreads.  If N/A, make
        this None.
    """

    # Check input args.
    _ = time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)
    error_checking.assert_is_string(model_file_name)
    if isotonic_model_file_name is not None:
        error_checking.assert_is_string(isotonic_model_file_name)
    if uncertainty_calib_model_file_name is not None:
        error_checking.assert_is_string(uncertainty_calib_model_file_name)

    error_checking.assert_is_numpy_array(
        target_matrix_with_weights, num_dimensions=3
    )
    error_checking.assert_is_geq(target_matrix_with_weights.shape[-1], 2)

    num_grid_rows = target_matrix_with_weights.shape[0]
    num_grid_columns = target_matrix_with_weights.shape[1]
    num_fields = target_matrix_with_weights.shape[2] - 1

    target_matrix = target_matrix_with_weights[..., :-1]
    error_checking.assert_is_numpy_array_without_nan(target_matrix)

    weight_matrix = target_matrix_with_weights[..., -1]
    error_checking.assert_is_geq_numpy_array(weight_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(weight_matrix, 1.)

    error_checking.assert_is_numpy_array(prediction_matrix, num_dimensions=4)

    ensemble_size = prediction_matrix.shape[3]
    expected_dim = numpy.array(
        [num_grid_rows, num_grid_columns, num_fields, ensemble_size], dtype=int
    )
    error_checking.assert_is_numpy_array(
        prediction_matrix, exact_dimensions=expected_dim
    )
    error_checking.assert_is_numpy_array_without_nan(prediction_matrix)

    error_checking.assert_is_numpy_array(
        grid_latitudes_deg_n,
        exact_dimensions=numpy.array([num_grid_rows], dtype=int)
    )
    error_checking.assert_is_valid_lat_numpy_array(
        grid_latitudes_deg_n, allow_nan=False
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(grid_latitudes_deg_n), 0.
    )

    error_checking.assert_is_numpy_array(
        grid_longitudes_deg_e,
        exact_dimensions=numpy.array([num_grid_columns], dtype=int)
    )
    grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        grid_longitudes_deg_e, allow_nan=False
    )

    if not numpy.all(numpy.diff(grid_longitudes_deg_e) > 0):
        grid_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            grid_longitudes_deg_e
        )

    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(field_names),
        exact_dimensions=numpy.array([num_fields], dtype=int)
    )

    # Do actual stuff.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    num_field_chars = max([len(f) for f in field_names])

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(
        ISOTONIC_MODEL_FILE_KEY,
        '' if isotonic_model_file_name is None else isotonic_model_file_name
    )
    dataset_object.setncattr(
        UNCERTAINTY_CALIB_MODEL_FILE_KEY,
        '' if uncertainty_calib_model_file_name is None
        else uncertainty_calib_model_file_name
    )
    dataset_object.setncattr(INIT_DATE_KEY, init_date_string)
    dataset_object.createDimension(ROW_DIM, num_grid_rows)
    dataset_object.createDimension(COLUMN_DIM, num_grid_columns)
    dataset_object.createDimension(FIELD_DIM, num_fields)
    dataset_object.createDimension(FIELD_CHAR_DIM, num_field_chars)
    dataset_object.createDimension(ENSEMBLE_MEMBER_DIM, ensemble_size)

    these_dim = (ROW_DIM, COLUMN_DIM, FIELD_DIM)
    dataset_object.createVariable(
        TARGET_KEY, datatype=numpy.float64, dimensions=these_dim
    )
    dataset_object.variables[TARGET_KEY][:] = target_matrix

    these_dim = (ROW_DIM, COLUMN_DIM, FIELD_DIM, ENSEMBLE_MEMBER_DIM)
    dataset_object.createVariable(
        PREDICTION_KEY, datatype=numpy.float64, dimensions=these_dim
    )
    dataset_object.variables[PREDICTION_KEY][:] = prediction_matrix

    these_dim = (ROW_DIM, COLUMN_DIM)
    dataset_object.createVariable(
        WEIGHT_KEY, datatype=numpy.float64, dimensions=these_dim
    )
    dataset_object.variables[WEIGHT_KEY][:] = weight_matrix

    dataset_object.createVariable(
        LATITUDE_KEY, datatype=numpy.float64, dimensions=ROW_DIM
    )
    dataset_object.variables[LATITUDE_KEY][:] = grid_latitudes_deg_n

    dataset_object.createVariable(
        LONGITUDE_KEY, datatype=numpy.float64, dimensions=COLUMN_DIM
    )
    dataset_object.variables[LONGITUDE_KEY][:] = grid_longitudes_deg_e

    this_string_format = 'S{0:d}'.format(num_field_chars)
    field_names_char_array = netCDF4.stringtochar(numpy.array(
        field_names, dtype=this_string_format
    ))

    dataset_object.createVariable(
        FIELD_NAME_KEY, datatype='S1', dimensions=(FIELD_DIM, FIELD_CHAR_DIM)
    )
    dataset_object.variables[FIELD_NAME_KEY][:] = numpy.array(
        field_names_char_array
    )

    dataset_object.close()


def take_ensemble_mean(prediction_table_xarray):
    """Takes ensemble mean for each atomic example.

    One atomic example = one init time, one valid time, one field, one pixel

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :return: prediction_table_xarray: Same but with only one prediction (the
        ensemble mean) per atomic example.
    """

    ptx = prediction_table_xarray
    ensemble_size = len(ptx.coords[ENSEMBLE_MEMBER_DIM].values)
    if ensemble_size == 1:
        return ptx

    ptx = ptx.assign_coords({
        DUMMY_ENSEMBLE_MEMBER_DIM: numpy.array([0], dtype=int)
    })

    these_dim = (ROW_DIM, COLUMN_DIM, FIELD_DIM, DUMMY_ENSEMBLE_MEMBER_DIM)

    ptx = ptx.assign({
        PREDICTION_KEY: (
            these_dim,
            numpy.mean(ptx[PREDICTION_KEY].values, axis=-1, keepdims=True)
        )
    })

    ptx = ptx.rename({DUMMY_ENSEMBLE_MEMBER_DIM: ENSEMBLE_MEMBER_DIM})
    prediction_table_xarray = ptx
    return prediction_table_xarray


def prep_for_uncertainty_calib_training(prediction_table_xarray):
    """Prepares predictions to train uncertainty calibration.

    Specifically, for every atomic example, this method replaces the full
    ensemble with the ensemble variance and replaces the target with the squared
    error of the ensemble mean.

    One atomic example = one init time, one valid time, one field, one pixel

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :return: prediction_table_xarray: Same but with the aforementioned changes.
    """

    ptx = prediction_table_xarray
    ensemble_size = len(ptx.coords[ENSEMBLE_MEMBER_DIM].values)
    assert ensemble_size > 1

    prediction_variance_matrix = numpy.var(
        ptx[PREDICTION_KEY].values, axis=-1, ddof=1, keepdims=True
    )
    squared_error_matrix = (
        numpy.mean(ptx[PREDICTION_KEY].values, axis=-1) -
        ptx[TARGET_KEY].values
    ) ** 2

    ptx = ptx.assign_coords({
        DUMMY_ENSEMBLE_MEMBER_DIM: numpy.array([0], dtype=int)
    })

    these_dim = (ROW_DIM, COLUMN_DIM, FIELD_DIM, DUMMY_ENSEMBLE_MEMBER_DIM)

    ptx = ptx.assign({
        PREDICTION_KEY: (
            these_dim,
            prediction_variance_matrix
        ),
        TARGET_KEY:(
            (ROW_DIM, COLUMN_DIM, FIELD_DIM),
            squared_error_matrix
        )
    })

    ptx = ptx.rename({DUMMY_ENSEMBLE_MEMBER_DIM: ENSEMBLE_MEMBER_DIM})
    prediction_table_xarray = ptx
    return prediction_table_xarray


def subset_by_location(
        prediction_table_xarray, desired_latitude_deg_n,
        desired_longitude_deg_e, radius_metres,
        recompute_weights_by_inverse_dist,
        recompute_weights_by_inverse_sq_dist):
    """Subsets prediction table by location.

    M = number of rows in original grid
    N = number of columns in original grid

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :param desired_latitude_deg_n: Desired latitude (deg north).
    :param desired_longitude_deg_e: Desired longitude (deg east).
    :param radius_metres: Will take all grid points within this radius around
        the desired point.
    :param recompute_weights_by_inverse_dist: Boolean flag.  If True, will
        recompute evaluation weights, scaling by inverse distance to desired
        point.
    :param recompute_weights_by_inverse_sq_dist: Boolean flag.  If True, will
        recompute evaluation weights, scaling by inverse *squared* distance to
        desired point.
    :return: new_prediction_table_xarray: Same as input but maybe with fewer
        examples.
    :return: keep_location_matrix: M-by-N numpy array of Boolean flags.
    """

    # Check input args.
    error_checking.assert_is_valid_latitude(
        desired_latitude_deg_n, allow_nan=False
    )
    desired_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg=desired_longitude_deg_e, allow_nan=False
    )

    error_checking.assert_is_greater(radius_metres, 0.)
    error_checking.assert_is_boolean(recompute_weights_by_inverse_dist)
    error_checking.assert_is_boolean(recompute_weights_by_inverse_sq_dist)
    if recompute_weights_by_inverse_dist:
        recompute_weights_by_inverse_sq_dist = False

    # Do actual stuff.
    grid_latitudes_deg_n = prediction_table_xarray[LATITUDE_KEY].values
    grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        prediction_table_xarray[LONGITUDE_KEY].values
    )
    latitude_matrix_deg_n, longitude_matrix_deg_e = (
        grids.latlng_vectors_to_matrices(
            unique_latitudes_deg=grid_latitudes_deg_n,
            unique_longitudes_deg=grid_longitudes_deg_e
        )
    )

    eval_weight_matrix = copy.deepcopy(
        prediction_table_xarray[WEIGHT_KEY].values
    )
    radius_degrees_lat = radius_metres * METRES_TO_DEGREES_LAT

    keep_location_matrix = numpy.logical_and(
        numpy.absolute(latitude_matrix_deg_n - desired_latitude_deg_n) <=
        radius_degrees_lat,
        numpy.absolute(longitude_matrix_deg_e - desired_longitude_deg_e) <=
        radius_degrees_lat
    )

    num_grid_rows = keep_location_matrix.shape[0]
    num_grid_columns = keep_location_matrix.shape[1]
    desired_point = (desired_latitude_deg_n, desired_longitude_deg_e)

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            if not keep_location_matrix[i, j]:
                continue

            this_point = (
                latitude_matrix_deg_n[i, j], longitude_matrix_deg_e[i, j]
            )
            this_distance_metres = geodesic(desired_point, this_point).meters

            if this_distance_metres > radius_metres:
                keep_location_matrix[i, j] = False
                continue

            if recompute_weights_by_inverse_dist:
                mult_factor = 1. - this_distance_metres / radius_metres
                eval_weight_matrix[i, j] *= mult_factor
                continue

            if recompute_weights_by_inverse_sq_dist:
                mult_factor = (1. - this_distance_metres / radius_metres) ** 2
                eval_weight_matrix[i, j] *= mult_factor

    eval_weight_matrix[keep_location_matrix == False] = 0.
    eval_weight_matrix[numpy.isnan(eval_weight_matrix)] = 0.
    good_rows, good_columns = numpy.where(keep_location_matrix)
    good_rows = numpy.unique(good_rows)
    good_columns = numpy.unique(good_columns)

    new_prediction_table_xarray = prediction_table_xarray.isel(
        {ROW_DIM: good_rows}
    )
    new_prediction_table_xarray = new_prediction_table_xarray.isel(
        {COLUMN_DIM: good_columns}
    )
    new_prediction_table_xarray = new_prediction_table_xarray.assign({
        WEIGHT_KEY: (
            new_prediction_table_xarray[WEIGHT_KEY].dims,
            eval_weight_matrix[good_rows, :][:, good_columns]
        )
    })

    return new_prediction_table_xarray, keep_location_matrix
