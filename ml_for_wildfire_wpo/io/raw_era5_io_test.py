"""Unit tests for raw_era5_io.py"""

import unittest
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml_for_wildfire_wpo.io import raw_era5_io

HOURS_TO_SECONDS = 3600

# The following constants are used to test _precip_hour_to_grib_search_string.
PRECIP_HOUR_START_TIME_STRINGS = [
    '2023-09-05-00', '2023-09-05-01', '2023-09-05-02',
    '2023-09-05-03', '2023-09-05-04', '2023-09-05-05',
    '2023-09-05-06', '2023-09-05-07', '2023-09-05-08',
    '2023-09-05-09', '2023-09-05-10', '2023-09-05-11',
    '2023-09-05-12', '2023-09-05-13', '2023-09-05-14',
    '2023-09-05-15', '2023-09-05-16', '2023-09-05-17',
    '2023-09-05-18', '2023-09-05-19', '2023-09-05-20',
    '2023-09-05-21', '2023-09-05-22', '2023-09-05-23'
]

PRECIP_HOUR_START_TIMES_UNIX_SEC = numpy.array([
    time_conversion.string_to_unix_sec(t, '%Y-%m-%d-%H')
    for t in PRECIP_HOUR_START_TIME_STRINGS
], dtype=int)

GRIB_SEARCH_STRINGS = [
    '23090418:TP:sfc:6-7hr', '23090418:TP:sfc:7-8hr', '23090418:TP:sfc:8-9hr',
    '23090418:TP:sfc:9-10hr', '23090418:TP:sfc:10-11hr', '23090418:TP:sfc:11-12hr',
    '23090506:TP:sfc:0-1hr', '23090506:TP:sfc:1-2hr', '23090506:TP:sfc:2-3hr',
    '23090506:TP:sfc:3-4hr', '23090506:TP:sfc:4-5hr', '23090506:TP:sfc:5-6hr',
    '23090506:TP:sfc:6-7hr', '23090506:TP:sfc:7-8hr', '23090506:TP:sfc:8-9hr',
    '23090506:TP:sfc:9-10hr', '23090506:TP:sfc:10-11hr', '23090506:TP:sfc:11-12hr',
    '23090518:TP:sfc:0-1hr', '23090518:TP:sfc:1-2hr', '23090518:TP:sfc:2-3hr',
    '23090518:TP:sfc:3-4hr', '23090518:TP:sfc:4-5hr', '23090518:TP:sfc:5-6hr'
]


class RawEra5IoTests(unittest.TestCase):
    """Each method is a unit test for raw_era5_io.py."""

    def test_find_local_noon_at_each_grid_point(self):
        """Ensures correct output from find_local_noon_at_each_grid_point."""

        for i in range(len(PRECIP_HOUR_START_TIMES_UNIX_SEC)):
            this_search_string = raw_era5_io._precip_hour_to_grib_search_string(
                PRECIP_HOUR_START_TIMES_UNIX_SEC[i]
            )
            self.assertTrue(this_search_string == GRIB_SEARCH_STRINGS[i])


if __name__ == '__main__':
    unittest.main()
