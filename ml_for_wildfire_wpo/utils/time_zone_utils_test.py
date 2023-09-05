"""Unit tests for time_zone_utils.py"""

import unittest
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from ml_for_wildfire_wpo.utils import time_zone_utils

HOURS_TO_SECONDS = 3600

# The following constants are used to test find_local_noon_at_each_grid_point.
UTC_OFFSET_MATRIX_HOURS = numpy.array([
    [-12, -11, -10, -9, -8, -7],
    [-6, -5, -4, -3, -2, -1],
    [1, 2, 3, 4, 5, 6.5],
    [7, 8, 9, 10, 11, 12]
])

DATA_DICT = {
    time_zone_utils.UTC_OFFSET_KEY: (
        (time_zone_utils.LATITUDE_KEY, time_zone_utils.LONGITUDE_KEY),
        UTC_OFFSET_MATRIX_HOURS
    )
}
TIME_ZONE_TABLE_XARRAY = xarray.Dataset(data_vars=DATA_DICT)

VALID_DATE_STRING = '20230905'
NOON_UTC_UNIX_SEC = (
    time_conversion.string_to_unix_sec(VALID_DATE_STRING, '%Y%m%d')
    + 12 * HOURS_TO_SECONDS
)

VALID_TIME_MATRIX_UNIX_SEC = NOON_UTC_UNIX_SEC + numpy.array([
    [43200, 39600, 36000, 32400, 28800, 25200],
    [21600, 18000, 14400, 10800, 7200, 3600],
    [-3600, -7200, -10800, -14400, -18000, -23400],
    [-25200, -28800, -32400, -36000, -39600, -43200]
], dtype=int)


class TimeZoneUtilsTests(unittest.TestCase):
    """Each method is a unit test for time_zone_utils.py."""

    def test_find_local_noon_at_each_grid_point(self):
        """Ensures correct output from find_local_noon_at_each_grid_point."""

        this_time_matrix_unix_sec = (
            time_zone_utils.find_local_noon_at_each_grid_point(
                valid_date_string=VALID_DATE_STRING,
                time_zone_table_xarray=TIME_ZONE_TABLE_XARRAY
            )
        )

        self.assertTrue(numpy.array_equal(
            this_time_matrix_unix_sec, VALID_TIME_MATRIX_UNIX_SEC
        ))


if __name__ == '__main__':
    unittest.main()
