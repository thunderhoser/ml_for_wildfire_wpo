"""Helper methods for time zones."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion

TOLERANCE = 1e-6

HOURS_TO_SECONDS = 3600
DATE_FORMAT = time_conversion.SPC_DATE_FORMAT

LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'
UTC_OFFSET_KEY = 'utc_offset_hours'


def find_local_noon_at_each_grid_point(valid_date_string,
                                       time_zone_table_xarray):
    """Finds local noon at each grid point.

    M = number of rows in grid
    N = number of columns in grid

    :param valid_date_string: Valid date (format "yyyy-mm-dd").
    :param time_zone_table_xarray: xarray table in format returned by
        `time_zone_io.read_file`.
    :return: valid_time_matrix_unix_sec: M-by-N numpy array of valid times (Unix
        seconds, UTC) at local noon.
    """

    valid_date_unix_sec = time_conversion.string_to_unix_sec(
        valid_date_string, DATE_FORMAT
    )
    utc_offset_matrix_sec = numpy.round(
        HOURS_TO_SECONDS *
        time_zone_table_xarray[UTC_OFFSET_KEY].values
    ).astype(int)

    return (
        valid_date_unix_sec
        + 12 * HOURS_TO_SECONDS
        - utc_offset_matrix_sec
    )
