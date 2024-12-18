"""Input/output methods for composited predictor and target fields."""

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
import neural_net

DATE_FORMAT = '%Y%m%d'

ROW_DIM = 'grid_row'
COLUMN_DIM = 'grid_column'
PRESSURE_LEVEL_DIM = 'pressure_level'
GFS_LEAD_TIME_DIM = 'gfs_lead_time'
GFS_3D_FIELD_DIM = 'gfs_3d_field'
GFS_2D_FIELD_DIM = 'gfs_2d_field'
ERA5_CONSTANT_FIELD_DIM = 'era5_constant_field'
LAGLEAD_TIME_DIM = 'laglead_time'
TARGET_FIELD_DIM = 'target_field_dim'

INIT_DATES_KEY = 'init_date_strings'
MODEL_LEAD_TIME_KEY = 'model_lead_time_days'
MODEL_FILE_KEY = 'model_file_name'

LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'
PREDICTOR_3D_GFS_KEY = 'predictor_gfs_3d'
PREDICTOR_2D_GFS_KEY = 'predictor_gfs_2d'
PREDICTOR_ERA5_KEY = 'predictor_era5'
PREDICTOR_LAGLEAD_TARGETS_KEY = 'predictor_lagged_target'
PREDICTOR_BASELINE_KEY = 'predictor_baseline'
TARGET_VALUE_KEY = 'target_value'


def write_file(
        netcdf_file_name, composite_predictor_matrices, composite_target_matrix,
        grid_latitudes_deg_n, grid_longitudes_deg_e, init_date_strings,
        model_input_layer_names, model_lead_time_days, model_file_name):
    """Writes composited predictor and target fields to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to output file.
    :param composite_predictor_matrices: 1-D list of numpy arrays with predictor
        values.
    :param composite_target_matrix: numpy array of target values.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param init_date_strings: 1-D list of forecast-init days averaged into
        composite (format "yyyymmdd").
    :param model_input_layer_names: 1-D list with names of input layers to
        model.
    :param model_lead_time_days: Model lead time.
    :param model_file_name: Path to file with trained model.
    """

    # Check input args.
    error_checking.assert_is_string_list(model_input_layer_names)
    for this_layer_name in model_input_layer_names:
        assert this_layer_name in neural_net.VALID_INPUT_LAYER_NAMES

    assert (
        neural_net.GFS_3D_LAYER_NAME in model_input_layer_names or
        neural_net.GFS_2D_LAYER_NAME in model_input_layer_names
    )
    assert len(model_input_layer_names) == len(composite_predictor_matrices)

    if neural_net.GFS_3D_LAYER_NAME in model_input_layer_names:
        lyr_idx = model_input_layer_names.index(neural_net.GFS_3D_LAYER_NAME)
    else:
        lyr_idx = model_input_layer_names.index(neural_net.GFS_2D_LAYER_NAME)

    num_grid_rows = composite_predictor_matrices[lyr_idx].shape[0]
    num_grid_columns = composite_predictor_matrices[lyr_idx].shape[1]

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

    error_checking.assert_is_string_list(init_date_strings)
    for this_date_string in init_date_strings:
        _ = time_conversion.string_to_unix_sec(
            this_date_string, DATE_FORMAT
        )

    error_checking.assert_is_integer(model_lead_time_days)
    error_checking.assert_is_greater(model_lead_time_days, 1)
    error_checking.assert_is_string(model_file_name)

    # Do actual stuff.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF4'
    )
    dataset_object.setncattr(INIT_DATES_KEY, ' '.join(init_date_strings))
    dataset_object.setncattr(MODEL_LEAD_TIME_KEY, model_lead_time_days)
    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)

    dataset_object.createDimension(ROW_DIM, num_grid_rows)
    dataset_object.createDimension(COLUMN_DIM, num_grid_columns)
    dataset_object.createDimension(
        TARGET_FIELD_DIM, composite_target_matrix.shape[2]
    )

    these_dim = (ROW_DIM, COLUMN_DIM, TARGET_FIELD_DIM)
    dataset_object.createVariable(
        TARGET_VALUE_KEY, datatype=numpy.float64, dimensions=these_dim
    )
    dataset_object.variables[TARGET_VALUE_KEY][:] = composite_target_matrix

    num_input_layers = len(model_input_layer_names)

    for k in range(num_input_layers):
        if model_input_layer_names[k] == neural_net.LEAD_TIME_LAYER_NAME:
            continue

        if model_input_layer_names[k] == neural_net.GFS_3D_LAYER_NAME:
            if PRESSURE_LEVEL_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    PRESSURE_LEVEL_DIM, composite_predictor_matrices[k].shape[2]
                )
            if GFS_LEAD_TIME_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    GFS_LEAD_TIME_DIM, composite_predictor_matrices[k].shape[3]
                )
            if GFS_3D_FIELD_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    GFS_3D_FIELD_DIM, composite_predictor_matrices[k].shape[4]
                )

            these_dim = (
                ROW_DIM, COLUMN_DIM, PRESSURE_LEVEL_DIM,
                GFS_LEAD_TIME_DIM, GFS_3D_FIELD_DIM
            )
            dataset_object.createVariable(
                PREDICTOR_3D_GFS_KEY,
                datatype=numpy.float64, dimensions=these_dim
            )
            dataset_object.variables[PREDICTOR_3D_GFS_KEY][:] = (
                composite_predictor_matrices[k]
            )

        elif model_input_layer_names[k] == neural_net.GFS_2D_LAYER_NAME:
            if GFS_LEAD_TIME_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    GFS_LEAD_TIME_DIM, composite_predictor_matrices[k].shape[2]
                )
            if GFS_2D_FIELD_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    GFS_2D_FIELD_DIM, composite_predictor_matrices[k].shape[3]
                )

            these_dim = (
                ROW_DIM, COLUMN_DIM, GFS_LEAD_TIME_DIM, GFS_2D_FIELD_DIM
            )
            dataset_object.createVariable(
                PREDICTOR_2D_GFS_KEY,
                datatype=numpy.float64, dimensions=these_dim
            )
            dataset_object.variables[PREDICTOR_2D_GFS_KEY][:] = (
                composite_predictor_matrices[k]
            )

        elif model_input_layer_names[k] == neural_net.LAGLEAD_TARGET_LAYER_NAME:
            if LAGLEAD_TIME_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    LAGLEAD_TIME_DIM, composite_predictor_matrices[k].shape[2]
                )

            these_dim = (
                ROW_DIM, COLUMN_DIM, LAGLEAD_TIME_DIM, TARGET_FIELD_DIM
            )
            dataset_object.createVariable(
                PREDICTOR_LAGLEAD_TARGETS_KEY,
                datatype=numpy.float64, dimensions=these_dim
            )
            dataset_object.variables[PREDICTOR_LAGLEAD_TARGETS_KEY][:] = (
                composite_predictor_matrices[k]
            )

        elif model_input_layer_names[k] == neural_net.ERA5_LAYER_NAME:
            if ERA5_CONSTANT_FIELD_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    ERA5_CONSTANT_FIELD_DIM,
                    composite_predictor_matrices[k].shape[2]
                )

            these_dim = (ROW_DIM, COLUMN_DIM, ERA5_CONSTANT_FIELD_DIM)
            dataset_object.createVariable(
                PREDICTOR_ERA5_KEY,
                datatype=numpy.float64, dimensions=these_dim
            )
            dataset_object.variables[PREDICTOR_ERA5_KEY][:] = (
                composite_predictor_matrices[k]
            )

        else:
            these_dim = (ROW_DIM, COLUMN_DIM, TARGET_FIELD_DIM)
            dataset_object.createVariable(
                PREDICTOR_BASELINE_KEY,
                datatype=numpy.float64, dimensions=these_dim
            )
            dataset_object.variables[PREDICTOR_BASELINE_KEY][:] = (
                composite_predictor_matrices[k]
            )

    dataset_object.createVariable(
        LATITUDE_KEY, datatype=numpy.float64, dimensions=ROW_DIM
    )
    dataset_object.variables[LATITUDE_KEY][:] = grid_latitudes_deg_n

    dataset_object.createVariable(
        LONGITUDE_KEY, datatype=numpy.float64, dimensions=COLUMN_DIM
    )
    dataset_object.variables[LONGITUDE_KEY][:] = grid_longitudes_deg_e

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads Shapley maps from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: composite_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    composite_table_xarray = xarray.open_dataset(netcdf_file_name)

    if composite_table_xarray.attrs[INIT_DATES_KEY] == '':
        composite_table_xarray.attrs[INIT_DATES_KEY] = None
    else:
        composite_table_xarray.attrs[INIT_DATES_KEY] = (
            composite_table_xarray.attrs[INIT_DATES_KEY].split(' ')
        )

    return composite_table_xarray
