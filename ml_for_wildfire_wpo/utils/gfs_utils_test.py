"""Unit tests for gfs_utils.py"""

import unittest
import numpy
import xarray
from ml_for_wildfire_wpo.utils import gfs_utils

TOLERANCE = 1e-6

# The following constants are used to test precip_from_incremental_to_full_run.
INCREMENTAL_PRECIP_VALUES = numpy.array([
    0,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4
], dtype=float)

ACCUMULATED_PRECIP_VALUES = numpy.array([
    0,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    83, 87, 90, 94, 97, 101, 104, 108, 111, 115, 118, 122
], dtype=float)

INCREMENTAL_CONVECTIVE_PRECIP_VALUES = 0.5 * INCREMENTAL_PRECIP_VALUES
ACCUMULATED_CONVECTIVE_PRECIP_VALUES = 0.5 * ACCUMULATED_PRECIP_VALUES

DATA_MATRIX_2D = numpy.stack(
    (INCREMENTAL_PRECIP_VALUES, INCREMENTAL_CONVECTIVE_PRECIP_VALUES),
    axis=-1
)
DATA_MATRIX_2D = numpy.expand_dims(DATA_MATRIX_2D, axis=-2)
DATA_MATRIX_2D = numpy.expand_dims(DATA_MATRIX_2D, axis=-2)

COORD_DICT = {
    gfs_utils.FORECAST_HOUR_DIM: gfs_utils.ALL_FORECAST_HOURS,
    gfs_utils.LATITUDE_DIM: numpy.array([40.02]),
    gfs_utils.LONGITUDE_DIM: numpy.array([254.75]),
    gfs_utils.PRESSURE_LEVEL_DIM: numpy.array([]),
    gfs_utils.FIELD_DIM_2D:
        [gfs_utils.PRECIP_NAME, gfs_utils.CONVECTIVE_PRECIP_NAME],
    gfs_utils.FIELD_DIM_3D: []
}

THESE_DIM_2D = (
    gfs_utils.FORECAST_HOUR_DIM, gfs_utils.LATITUDE_DIM,
    gfs_utils.LONGITUDE_DIM, gfs_utils.FIELD_DIM_2D
)
MAIN_DATA_DICT = {
    gfs_utils.DATA_KEY_2D: (THESE_DIM_2D, DATA_MATRIX_2D)
}

GFS_TABLE_XARRAY = xarray.Dataset(data_vars=MAIN_DATA_DICT, coords=COORD_DICT)


class GfsUtilsTests(unittest.TestCase):
    """Each method is a unit test for gfs_utils.py."""

    def test_precip_from_incremental_to_full_run(self):
        """Ensures correct output from precip_from_incremental_to_full_run."""

        new_gfs_table_xarray = gfs_utils.precip_from_incremental_to_full_run(
            GFS_TABLE_XARRAY
        )

        these_precip_values = (
            new_gfs_table_xarray[gfs_utils.DATA_KEY_2D].values[..., 0, 0, 0]
        )
        self.assertTrue(numpy.allclose(
            these_precip_values, ACCUMULATED_PRECIP_VALUES, atol=TOLERANCE
        ))

        these_precip_values = (
            new_gfs_table_xarray[gfs_utils.DATA_KEY_2D].values[..., 0, 0, 1]
        )
        self.assertTrue(numpy.allclose(
            these_precip_values, ACCUMULATED_CONVECTIVE_PRECIP_VALUES,
            atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
