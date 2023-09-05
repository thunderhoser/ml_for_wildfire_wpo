"""Helper methods for time zones."""

import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

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


def subset_by_longitude(time_zone_table_xarray, desired_longitudes_deg_e):
    """Subsets time-zone table by longitude.
    
    N = number of desired longitudes
    
    :param time_zone_table_xarray: xarray table in format returned by
        `time_zone_io.read_file`.
    :param desired_longitudes_deg_e: length-N numpy array of desired longitudes
        (deg east).
    :return: time_zone_table_xarray: Same as input but maybe with fewer
        longitudes.
    :return: desired_indices: length-N numpy array of selected indices.
    """

    # TODO(thunderhoser): I might want to worry about making sure that,
    # regardless of the longitude format (positive or negative in western
    # hemisphere), they are sorted monotonically.  Not sure about this yet.

    are_longitudes_positive_in_west = True
    
    try:
        error_checking.assert_is_valid_lng_numpy_array(
            longitudes_deg=desired_longitudes_deg_e,
            positive_in_west_flag=True,
            negative_in_west_flag=False,
            allow_nan=False
        )
    except:
        error_checking.assert_is_valid_lng_numpy_array(
            longitudes_deg=desired_longitudes_deg_e,
            positive_in_west_flag=False,
            negative_in_west_flag=True,
            allow_nan=False
        )

        are_longitudes_positive_in_west = False

    if are_longitudes_positive_in_west:
        grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            time_zone_table_xarray.coords[LONGITUDE_KEY].values + 0.
        )
    else:
        grid_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            time_zone_table_xarray.coords[LONGITUDE_KEY].values + 0.
        )

    desired_indices = numpy.array([
        numpy.where(
            numpy.absolute(grid_longitudes_deg_e - d) <= TOLERANCE
        )[0][0]
        for d in desired_longitudes_deg_e
    ], dtype=int)

    time_zone_table_xarray = time_zone_table_xarray.isel(
        {LONGITUDE_KEY: desired_indices}
    )

    return time_zone_table_xarray, desired_indices
