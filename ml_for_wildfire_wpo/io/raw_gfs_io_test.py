"""Unit tests for raw_gfs_io.py"""

import unittest
import numpy
from ml_for_wildfire_wpo.io import raw_gfs_io

HOURS_TO_SECONDS = 3600

# The following constants are used to test desired_longitudes_to_columns.
FIRST_START_LONGITUDE_DEG_E = -164.89
FIRST_END_LONGITUDE_DEG_E = -150.2
FIRST_DESIRED_COLUMN_INDICES = numpy.linspace(780, 840, num=61, dtype=int)

SECOND_START_LONGITUDE_DEG_E = 195.11
SECOND_END_LONGITUDE_DEG_E = 209.8
SECOND_DESIRED_COLUMN_INDICES = numpy.linspace(780, 840, num=61, dtype=int)

THIRD_START_LONGITUDE_DEG_E = 160.07
THIRD_END_LONGITUDE_DEG_E = 19.81
THIRD_DESIRED_COLUMN_INDICES = numpy.concatenate((
    numpy.linspace(640, 1439, num=800, dtype=int),
    numpy.linspace(0, 80, num=81, dtype=int)
))

FOURTH_START_LONGITUDE_DEG_E = 160.07
FOURTH_END_LONGITUDE_DEG_E = 199.99
FOURTH_DESIRED_COLUMN_INDICES = numpy.linspace(640, 800, num=161, dtype=int)

FIFTH_START_LONGITUDE_DEG_E = 160.07
FIFTH_END_LONGITUDE_DEG_E = -60.21
FIFTH_DESIRED_COLUMN_INDICES = numpy.linspace(640, 1200, num=561, dtype=int)

SIXTH_START_LONGITUDE_DEG_E = -180.
SIXTH_END_LONGITUDE_DEG_E = 200.22
SIXTH_DESIRED_COLUMN_INDICES = numpy.linspace(720, 801, num=82, dtype=int)

# The following constants are used to test desired_latitudes_to_rows.
FIRST_START_LATITUDE_DEG_N = 40.004
FIRST_END_LATITUDE_DEG_N = 48.793
FIRST_DESIRED_ROW_INDICES = numpy.linspace(520, 556, num=37, dtype=int)

SECOND_START_LATITUDE_DEG_N = -89.999
SECOND_END_LATITUDE_DEG_N = 89.999
SECOND_DESIRED_ROW_INDICES = numpy.linspace(0, 720, num=721, dtype=int)


class RawGfsIoTests(unittest.TestCase):
    """Each method is a unit test for raw_gfs_io.py."""

    def test_desired_longitudes_to_columns_first(self):
        """Ensures correct output from desired_longitudes_to_columns.

        In this case, with first set of inputs.
        """

        these_column_indices = raw_gfs_io.desired_longitudes_to_columns(
            start_longitude_deg_e=FIRST_START_LONGITUDE_DEG_E,
            end_longitude_deg_e=FIRST_END_LONGITUDE_DEG_E
        )
        self.assertTrue(numpy.array_equal(
            FIRST_DESIRED_COLUMN_INDICES, these_column_indices
        ))

    def test_desired_longitudes_to_columns_second(self):
        """Ensures correct output from desired_longitudes_to_columns.

        In this case, with second set of inputs.
        """

        these_column_indices = raw_gfs_io.desired_longitudes_to_columns(
            start_longitude_deg_e=SECOND_START_LONGITUDE_DEG_E,
            end_longitude_deg_e=SECOND_END_LONGITUDE_DEG_E
        )
        self.assertTrue(numpy.array_equal(
            SECOND_DESIRED_COLUMN_INDICES, these_column_indices
        ))

    def test_desired_longitudes_to_columns_third(self):
        """Ensures correct output from desired_longitudes_to_columns.

        In this case, with third set of inputs.
        """

        these_column_indices = raw_gfs_io.desired_longitudes_to_columns(
            start_longitude_deg_e=THIRD_START_LONGITUDE_DEG_E,
            end_longitude_deg_e=THIRD_END_LONGITUDE_DEG_E
        )
        self.assertTrue(numpy.array_equal(
            THIRD_DESIRED_COLUMN_INDICES, these_column_indices
        ))

    def test_desired_longitudes_to_columns_fourth(self):
        """Ensures correct output from desired_longitudes_to_columns.

        In this case, with fourth set of inputs.
        """

        these_column_indices = raw_gfs_io.desired_longitudes_to_columns(
            start_longitude_deg_e=FOURTH_START_LONGITUDE_DEG_E,
            end_longitude_deg_e=FOURTH_END_LONGITUDE_DEG_E
        )
        self.assertTrue(numpy.array_equal(
            FOURTH_DESIRED_COLUMN_INDICES, these_column_indices
        ))

    def test_desired_longitudes_to_columns_fifth(self):
        """Ensures correct output from desired_longitudes_to_columns.

        In this case, with fifth set of inputs.
        """

        these_column_indices = raw_gfs_io.desired_longitudes_to_columns(
            start_longitude_deg_e=FIFTH_START_LONGITUDE_DEG_E,
            end_longitude_deg_e=FIFTH_END_LONGITUDE_DEG_E
        )
        self.assertTrue(numpy.array_equal(
            FIFTH_DESIRED_COLUMN_INDICES, these_column_indices
        ))

    def test_desired_longitudes_to_columns_sixth(self):
        """Ensures correct output from desired_longitudes_to_columns.

        In this case, with sixth set of inputs.
        """

        these_column_indices = raw_gfs_io.desired_longitudes_to_columns(
            start_longitude_deg_e=SIXTH_START_LONGITUDE_DEG_E,
            end_longitude_deg_e=SIXTH_END_LONGITUDE_DEG_E
        )
        self.assertTrue(numpy.array_equal(
            SIXTH_DESIRED_COLUMN_INDICES, these_column_indices
        ))

    def test_desired_latitudes_to_rows_first(self):
        """Ensures correct output from desired_latitudes_to_rows.

        In this case, with first set of inputs.
        """

        these_row_indices = raw_gfs_io.desired_latitudes_to_rows(
            start_latitude_deg_n=FIRST_START_LATITUDE_DEG_N,
            end_latitude_deg_n=FIRST_END_LATITUDE_DEG_N
        )
        self.assertTrue(numpy.array_equal(
            FIRST_DESIRED_ROW_INDICES, these_row_indices
        ))

    def test_desired_latitudes_to_rows_second(self):
        """Ensures correct output from desired_latitudes_to_rows.

        In this case, with second set of inputs.
        """

        these_row_indices = raw_gfs_io.desired_latitudes_to_rows(
            start_latitude_deg_n=SECOND_START_LATITUDE_DEG_N,
            end_latitude_deg_n=SECOND_END_LATITUDE_DEG_N
        )
        self.assertTrue(numpy.array_equal(
            SECOND_DESIRED_ROW_INDICES, these_row_indices
        ))


if __name__ == '__main__':
    unittest.main()
