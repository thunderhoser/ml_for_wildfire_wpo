"""Helper methods for daily GFS data."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import gfs_utils
import canadian_fwi_utils

RELATIVE_HUMIDITY_2METRE_NAME = 'relative_humidity_2m_agl'
TEMPERATURE_2METRE_NAME = gfs_utils.TEMPERATURE_2METRE_NAME
DEWPOINT_2METRE_NAME = gfs_utils.DEWPOINT_2METRE_NAME
U_WIND_10METRE_NAME = gfs_utils.U_WIND_10METRE_NAME
V_WIND_10METRE_NAME = gfs_utils.V_WIND_10METRE_NAME
SURFACE_PRESSURE_NAME = gfs_utils.SURFACE_PRESSURE_NAME
PRECIP_NAME = gfs_utils.PRECIP_NAME

ALL_NON_FWI_FIELD_NAMES = [
    RELATIVE_HUMIDITY_2METRE_NAME, TEMPERATURE_2METRE_NAME,
    DEWPOINT_2METRE_NAME, U_WIND_10METRE_NAME, V_WIND_10METRE_NAME,
    SURFACE_PRESSURE_NAME, PRECIP_NAME
]

FFMC_NAME = canadian_fwi_utils.FFMC_NAME
DMC_NAME = canadian_fwi_utils.DMC_NAME
DC_NAME = canadian_fwi_utils.DC_NAME
ISI_NAME = canadian_fwi_utils.ISI_NAME
BUI_NAME = canadian_fwi_utils.BUI_NAME
FWI_NAME = canadian_fwi_utils.FWI_NAME
DSR_NAME = canadian_fwi_utils.DSR_NAME

ALL_FWI_FIELD_NAMES = [
    FFMC_NAME, DMC_NAME, DC_NAME, ISI_NAME, BUI_NAME, FWI_NAME, DSR_NAME
]
ALL_FIELD_NAMES = ALL_NON_FWI_FIELD_NAMES + ALL_FWI_FIELD_NAMES

LATITUDE_DIM = gfs_utils.LATITUDE_DIM
LONGITUDE_DIM = gfs_utils.LONGITUDE_DIM
FIELD_DIM = gfs_utils.FIELD_DIM_2D
LEAD_TIME_DIM = 'lead_time_days'

DATA_KEY_2D = gfs_utils.DATA_KEY_2D


def check_field_name(field_name):
    """Ensures that field name is valid.

    :param field_name: String (must be in list `ALL_FIELD_NAMES`).
    :raises: ValueError: if `field_name not in ALL_FIELD_NAMES`.
    """

    error_checking.assert_is_string(field_name)
    if field_name in ALL_FIELD_NAMES:
        return

    error_string = (
        'Field name "{0:s}" is not in the list of accepted field names '
        '(below):\n{1:s}'
    ).format(
        field_name, str(ALL_FIELD_NAMES)
    )

    raise ValueError(error_string)


def get_field(daily_gfs_table_xarray, field_name):
    """Extracts one field from xarray table.

    M = number of rows in grid
    N = number of columns in grid

    :param daily_gfs_table_xarray: xarray table with daily GFS forecasts.
    :param field_name: Field name.
    :return: data_matrix: M-by-N-by-L numpy array of data values.
    """

    error_checking.assert_is_string(field_name)

    k = numpy.where(
        daily_gfs_table_xarray.coords[FIELD_DIM].values == field_name
    )[0][0]

    data_matrix = daily_gfs_table_xarray[DATA_KEY_2D].values[..., k]
    data_matrix = numpy.swapaxes(data_matrix, 0, 1)
    data_matrix = numpy.swapaxes(data_matrix, 1, 2)

    return data_matrix


def subset_by_row(daily_gfs_table_xarray, desired_row_indices):
    """Subsets xarray table by grid row.

    :param daily_gfs_table_xarray: xarray table with daily GFS forecasts.
    :param desired_row_indices: 1-D numpy array with indices of desired rows.
    :return: daily_gfs_table_xarray: Same as input but maybe with fewer rows.
    """

    error_checking.assert_is_numpy_array(desired_row_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_row_indices)
    error_checking.assert_is_geq_numpy_array(desired_row_indices, 0)

    return daily_gfs_table_xarray.isel({LATITUDE_DIM: desired_row_indices})


def subset_by_column(daily_gfs_table_xarray, desired_column_indices):
    """Subsets xarray table by grid column.

    :param daily_gfs_table_xarray: xarray table with daily GFS forecasts.
    :param desired_column_indices: 1-D numpy array with indices of desired
        columns.
    :return: daily_gfs_table_xarray: Same as input but maybe with fewer
        columns.
    """

    error_checking.assert_is_numpy_array(
        desired_column_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)

    return daily_gfs_table_xarray.isel(
        {LONGITUDE_DIM: desired_column_indices}
    )


def subset_by_lead_time(daily_gfs_table_xarray, lead_times_days):
    """Subsets xarray table by lead time.

    :param daily_gfs_table_xarray: xarray table with daily GFS forecasts.
    :param lead_times_days: 1-D numpy array of lead times.
    :return: daily_gfs_table_xarray: Same as input but maybe with fewer lead
        times.
    """

    error_checking.assert_is_numpy_array(lead_times_days, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(lead_times_days)
    error_checking.assert_is_greater_numpy_array(lead_times_days, 0)

    return daily_gfs_table_xarray.sel({LEAD_TIME_DIM: lead_times_days})
