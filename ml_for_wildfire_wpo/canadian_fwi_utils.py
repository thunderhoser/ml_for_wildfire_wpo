"""Helper methods for Canadian fire-weather indices."""

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
FIELD_DIM = 'field_name'
DATA_KEY = 'data'

MEAN_VALUE_KEY = 'mean_value'
MEAN_SQUARED_VALUE_KEY = 'mean_squared_value'
STDEV_KEY = 'standard_deviation'

FFMC_NAME = 'fine_fuel_moisture_code'
DMC_NAME = 'duff_moisture_code'
DC_NAME = 'drought_code'
ISI_NAME = 'initial_spread_index'
BUI_NAME = 'buildup_index'
FWI_NAME = 'fire_weather_index'
DSR_NAME = 'daily_severity_rating'

ALL_FIELD_NAMES = [
    FFMC_NAME, DMC_NAME, DC_NAME, ISI_NAME, BUI_NAME, FWI_NAME, DSR_NAME
]


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


def get_field(fwi_table_xarray, field_name):
    """Extracts one field from xarray table.

    M = number of rows in grid
    N = number of columns in grid

    :param fwi_table_xarray: xarray table with Canadian FWI variables.
    :param field_name: Field name.
    :return: data_matrix: M-by-N numpy array of data values.
    """

    error_checking.assert_is_string(field_name)

    k = numpy.where(
        fwi_table_xarray.coords[FIELD_DIM].values == field_name
    )[0][0]

    return fwi_table_xarray[DATA_KEY].values[..., k]


def subset_by_row(fwi_table_xarray, desired_row_indices):
    """Subsets xarray table by grid row.

    :param fwi_table_xarray: xarray table with Canadian FWI variables.
    :param desired_row_indices: 1-D numpy array with indices of desired rows.
    :return: fwi_table_xarray: Same as input but maybe with fewer
        rows.
    """

    error_checking.assert_is_numpy_array(desired_row_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_row_indices)
    error_checking.assert_is_geq_numpy_array(desired_row_indices, 0)

    return fwi_table_xarray.isel({LATITUDE_DIM: desired_row_indices})


def subset_by_column(fwi_table_xarray, desired_column_indices):
    """Subsets xarray table by grid column.

    :param fwi_table_xarray: xarray table with Canadian FWI variables.
    :param desired_column_indices: 1-D numpy array with indices of desired
        columns.
    :return: fwi_table_xarray: Same as input but maybe with fewer
        columns.
    """

    error_checking.assert_is_numpy_array(
        desired_column_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)

    return fwi_table_xarray.isel(
        {LONGITUDE_DIM: desired_column_indices}
    )
