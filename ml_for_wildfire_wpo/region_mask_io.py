"""Input/output methods for region masks."""

import os
import sys
import numpy
import xarray
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import longitude_conversion as lng_conversion
import file_system_utils
import error_checking

ROW_DIM = 'grid_row'
COLUMN_DIM = 'grid_column'

REGION_MASK_KEY = 'region_mask'
LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'


def write_file(
        netcdf_file_name, mask_matrix, grid_latitudes_deg_n,
        grid_longitudes_deg_e):
    """Writes region mask to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to output file.
    :param mask_matrix: M-by-N numpy array of Boolean flags, where True (False)
        means that the pixel is (not) inside the region.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    """

    error_checking.assert_is_boolean_numpy_array(mask_matrix)
    error_checking.assert_is_numpy_array(mask_matrix, num_dimensions=2)

    num_grid_rows = mask_matrix.shape[0]
    num_grid_columns = mask_matrix.shape[1]

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
        netcdf_file_name, 'w', format='NETCDF4'
    )

    dataset_object.createDimension(ROW_DIM, num_grid_rows)
    dataset_object.createDimension(COLUMN_DIM, num_grid_columns)

    these_dim = (ROW_DIM, COLUMN_DIM)
    dataset_object.createVariable(
        REGION_MASK_KEY, datatype=numpy.int32, dimensions=these_dim
    )
    dataset_object.variables[REGION_MASK_KEY][:] = mask_matrix.astype(int)

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
    """Reads predictions from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: mask_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    mask_table_xarray = xarray.open_dataset(netcdf_file_name)

    return mask_table_xarray.assign({
        REGION_MASK_KEY: (
            mask_table_xarray[REGION_MASK_KEY].dims,
            mask_table_xarray[REGION_MASK_KEY].values.astype(bool)
        )
    })
