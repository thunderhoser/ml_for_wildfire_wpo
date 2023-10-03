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
CLASS_DIM = 'class'

MODEL_FILE_KEY = 'model_file_name'
INIT_DATE_KEY = 'init_date_string'

TARGET_KEY = 'target'
PREDICTION_KEY = 'prediction'
WEIGHT_KEY = 'evaluation_weight'
LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'


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

    return xarray.open_dataset(netcdf_file_name)


def write_file(
        netcdf_file_name, target_matrix_with_weights, prediction_matrix,
        grid_latitudes_deg_n, grid_longitudes_deg_e,
        init_date_string, model_file_name):
    """Writes predictions to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid
    K = number of classes

    :param netcdf_file_name: Path to output file.
    :param target_matrix_with_weights: If the model does classification, should
        be M-by-N-by-(K + 1) numpy array, where
        target_matrix_with_weights[..., -1] contains evaluation weights and
        target_matrix_with_weights[..., :-1] contains a one-hot encoding of the
        true classes.  If the model does regression, should be M-by-N-by-2 numpy
        array, where target_matrix_with_weights[..., -1] contains weights and
        target_matrix_with_weights[..., 0] contains target values.
    :param prediction_matrix: If the model does classification, should be
        M-by-N-by-K numpy array of class probabilities.  If the model does
        regression, should be M-by-N numpy array of predicted target values.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param init_date_string: Initialization date (format "yyyymmdd").
    :param model_file_name: Path to file with trained model.
    """

    # Check input args.
    _ = time_conversion.unix_sec_to_string(init_date_string, DATE_FORMAT)
    error_checking.assert_is_string(model_file_name)

    error_checking.assert_is_numpy_array(
        target_matrix_with_weights, num_dimensions=3
    )
    error_checking.assert_is_geq(target_matrix_with_weights.shape[-1], 2)

    num_grid_rows = target_matrix_with_weights.shape[0]
    num_grid_columns = target_matrix_with_weights.shape[1]
    for_classification = target_matrix_with_weights.shape[-1] > 2

    if for_classification:
        target_matrix = target_matrix_with_weights[..., :-1]

        diff_from_zero_matrix = numpy.absolute(target_matrix - 0.)
        diff_from_one_matrix = numpy.absolute(target_matrix - 1.)
        diff_matrix = numpy.minimum(diff_from_zero_matrix, diff_from_one_matrix)
        error_checking.assert_is_leq_numpy_array(diff_matrix, TOLERANCE)

        target_matrix = numpy.round(target_matrix).astype(int)
        error_checking.assert_is_geq_numpy_array(target_matrix, 0)
        error_checking.assert_is_leq_numpy_array(target_matrix, 1)
    else:
        target_matrix = target_matrix_with_weights[..., 0]
        error_checking.assert_is_numpy_array_without_nan(target_matrix)

    weight_matrix = target_matrix_with_weights[..., -1]
    error_checking.assert_is_geq_numpy_array(weight_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(weight_matrix, 1.)

    if for_classification:
        num_classes = target_matrix.shape[-1]
        expected_dim = numpy.array(
            [num_grid_rows, num_grid_columns, num_classes], dtype=int
        )
        error_checking.assert_is_numpy_array(
            prediction_matrix, exact_dimensions=expected_dim
        )

        error_checking.assert_is_geq_numpy_array(prediction_matrix, 0.)
        error_checking.assert_is_leq_numpy_array(prediction_matrix, 1.)

        diff_from_one_matrix = numpy.absolute(
            numpy.mean(prediction_matrix, axis=-1) - 1.
        )
        error_checking.assert_is_leq_numpy_array(
            diff_from_one_matrix, TOLERANCE
        )
    else:
        num_classes = 0

        expected_dim = numpy.array([num_grid_rows, num_grid_columns], dtype=int)
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

    # Do actual stuff.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(INIT_DATE_KEY, init_date_string)
    dataset_object.createDimension(ROW_DIM, num_grid_rows)
    dataset_object.createDimension(COLUMN_DIM, num_grid_columns)

    if for_classification:
        dataset_object.createDimension(CLASS_DIM, num_classes)

        these_dim = (ROW_DIM, COLUMN_DIM, CLASS_DIM)
        dataset_object.createVariable(
            TARGET_KEY, datatype=numpy.int32, dimensions=these_dim
        )
    else:
        these_dim = (ROW_DIM, COLUMN_DIM)
        dataset_object.createVariable(
            TARGET_KEY, datatype=numpy.float32, dimensions=these_dim
        )

    dataset_object.variables[TARGET_KEY][:] = target_matrix

    dataset_object.createVariable(
        PREDICTION_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[PREDICTION_KEY][:] = prediction_matrix

    these_dim = (ROW_DIM, COLUMN_DIM)
    dataset_object.createVariable(
        WEIGHT_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[WEIGHT_KEY][:] = weight_matrix

    dataset_object.createVariable(
        LATITUDE_KEY, datatype=numpy.float32, dimensions=ROW_DIM
    )
    dataset_object.variables[LATITUDE_KEY][:] = grid_latitudes_deg_n

    dataset_object.createVariable(
        LONGITUDE_KEY, datatype=numpy.float32, dimensions=COLUMN_DIM
    )
    dataset_object.variables[LONGITUDE_KEY][:] = grid_longitudes_deg_e

    dataset_object.close()
