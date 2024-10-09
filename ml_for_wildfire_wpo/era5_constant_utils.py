"""Helper methods for constants in ERA5 reanalysis."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking

LATITUDE_DIM = 'latitude_deg_n'
LONGITUDE_DIM = 'longitude_deg_e'
FIELD_DIM = 'field'
QUANTILE_LEVEL_DIM = 'quantile_level'

DATA_KEY = 'data'

MEAN_VALUE_KEY = 'mean_value'
MEAN_SQUARED_VALUE_KEY = 'mean_squared_value'
STDEV_KEY = 'standard_deviation'
QUANTILE_KEY = 'quantile'

SUBGRID_OROGRAPHY_SINE_NAME = 'subgrid_orography_angle_sine'
SUBGRID_OROGRAPHY_COSINE_NAME = 'subgrid_orography_angle_cosine'
SUBGRID_OROGRAPHY_ANISOTROPY_NAME = 'subgrid_orography_angle_anisotropy'
GEOPOTENTIAL_NAME = 'geopotential_m2_s02'
LAND_SEA_MASK_NAME = 'land_sea_mask_boolean'
SUBGRID_OROGRAPHY_SLOPE_NAME = 'subgrid_orography_angle_slope'
SUBGRID_OROGRAPHY_STDEV_NAME = 'filtered_subgrid_orography_stdev_metres'
RESOLVED_OROGRAPHY_STDEV_NAME = 'resolved_orography_stdev_metres'

ALL_FIELD_NAMES = [
    SUBGRID_OROGRAPHY_SINE_NAME, SUBGRID_OROGRAPHY_COSINE_NAME,
    SUBGRID_OROGRAPHY_ANISOTROPY_NAME, GEOPOTENTIAL_NAME,
    LAND_SEA_MASK_NAME, SUBGRID_OROGRAPHY_SLOPE_NAME,
    SUBGRID_OROGRAPHY_STDEV_NAME, RESOLVED_OROGRAPHY_STDEV_NAME
]

FIELD_NAME_TO_FANCY = {
    SUBGRID_OROGRAPHY_SINE_NAME: 'sin of subgrid orog\'y angle',
    SUBGRID_OROGRAPHY_COSINE_NAME: 'cos of subgrid orog\'y angle',
    SUBGRID_OROGRAPHY_ANISOTROPY_NAME: 'anisotropy of subgrid orog\'y angle',
    GEOPOTENTIAL_NAME: 'sfc geopotential',
    LAND_SEA_MASK_NAME: 'land/sea mask',
    SUBGRID_OROGRAPHY_SLOPE_NAME: 'slope of subgrid orog\'y angle',
    SUBGRID_OROGRAPHY_STDEV_NAME: 'stdev of subgrid orog\'y',
    RESOLVED_OROGRAPHY_STDEV_NAME: 'stdev of resolved orog\'y',
}

FIELD_NAME_TO_UNIT_STRING = {
    SUBGRID_OROGRAPHY_SINE_NAME: '',
    SUBGRID_OROGRAPHY_COSINE_NAME: '',
    SUBGRID_OROGRAPHY_ANISOTROPY_NAME: '',
    GEOPOTENTIAL_NAME: r'm$^2$ s$^{-2}$',
    LAND_SEA_MASK_NAME: '',
    SUBGRID_OROGRAPHY_SLOPE_NAME: '',
    SUBGRID_OROGRAPHY_STDEV_NAME: 'm',
    RESOLVED_OROGRAPHY_STDEV_NAME: 'm',
}


def check_field_name(field_name):
    """Ensures validity of field name.

    :param field_name: Name of weather field.
    :raises: ValueError: if `field_name not in ALL_FIELD_NAMES`.
    """

    error_checking.assert_is_string(field_name)
    if field_name in ALL_FIELD_NAMES:
        return

    error_string = (
        'Field "{0:s}" is not in the list of recognized fields (below):\n{1:s}'
    ).format(
        field_name, str(ALL_FIELD_NAMES)
    )

    raise ValueError(error_string)


def get_field(era5_constant_table_xarray, field_name):
    """Extracts one field from xarray table.

    M = number of rows in grid
    N = number of columns in grid

    :param era5_constant_table_xarray: xarray table with ERA5 constants.
    :param field_name: Field name.
    :return: data_matrix: M-by-N numpy array of data values.
    """

    error_checking.assert_is_string(field_name)

    k = numpy.where(
        era5_constant_table_xarray.coords[FIELD_DIM].values == field_name
    )[0][0]

    return era5_constant_table_xarray[DATA_KEY].values[..., k]


def subset_by_row(era5_constant_table_xarray, desired_row_indices):
    """Subsets xarray table by grid row.

    :param era5_constant_table_xarray: xarray table with ERA5 constants.
    :param desired_row_indices: 1-D numpy array with indices of desired rows.
    :return: era5_constant_table_xarray: Same as input but maybe with fewer
        rows.
    """

    error_checking.assert_is_numpy_array(desired_row_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_row_indices)
    error_checking.assert_is_geq_numpy_array(desired_row_indices, 0)

    return era5_constant_table_xarray.isel({LATITUDE_DIM: desired_row_indices})


def subset_by_column(era5_constant_table_xarray, desired_column_indices):
    """Subsets xarray table by grid column.

    :param era5_constant_table_xarray: xarray table with ERA5 constants.
    :param desired_column_indices: 1-D numpy array with indices of desired
        columns.
    :return: era5_constant_table_xarray: Same as input but maybe with fewer
        columns.
    """

    error_checking.assert_is_numpy_array(
        desired_column_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)

    return era5_constant_table_xarray.isel(
        {LONGITUDE_DIM: desired_column_indices}
    )
