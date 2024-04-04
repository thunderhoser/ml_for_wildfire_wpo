"""Input/output methods for predictions."""

import os
import sys
import numpy
import xarray
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking

TOLERANCE = 1e-6
DATE_FORMAT = '%Y%m%d'

ROW_DIM = 'grid_row'
COLUMN_DIM = 'grid_column'
FIELD_DIM = 'field'
FIELD_CHAR_DIM = 'field_char'
ENSEMBLE_MEMBER_DIM = 'ensemble_member'

MODEL_FILE_KEY = 'model_file_name'
INIT_DATE_KEY = 'init_date_string'

TARGET_KEY = 'target'
PREDICTION_KEY = 'prediction'
WEIGHT_KEY = 'evaluation_weight'
LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'
FIELD_NAME_KEY = 'field_name'


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

    if ENSEMBLE_MEMBER_DIM not in prediction_table_xarray.coords:
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

    return prediction_table_xarray.assign({
        FIELD_NAME_KEY: (
            prediction_table_xarray[FIELD_NAME_KEY].dims,
            target_field_names
        )
    })


def write_file(
        netcdf_file_name, target_matrix_with_weights, prediction_matrix,
        grid_latitudes_deg_n, grid_longitudes_deg_e, field_names,
        init_date_string, model_file_name):
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
    """

    # Check input args.
    _ = time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)
    error_checking.assert_is_string(model_file_name)

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
