"""Unit tests for era5_utils.py"""

import unittest
import numpy
from ml_for_wildfire_wpo.utils import era5_utils

HOURS_TO_SECONDS = 3600

# The following constants are used to test desired_longitudes_to_columns.
FIRST_START_LONGITUDE_DEG_E = -164.89
FIRST_END_LONGITUDE_DEG_E = -150.2
FIRST_DESIRED_COLUMN_INDICES = numpy.linspace(60, 120, num=61, dtype=int)

SECOND_START_LONGITUDE_DEG_E = 195.11
SECOND_END_LONGITUDE_DEG_E = 209.8
SECOND_DESIRED_COLUMN_INDICES = numpy.linspace(60, 120, num=61, dtype=int)

THIRD_START_LONGITUDE_DEG_E = 160.07
THIRD_END_LONGITUDE_DEG_E = 19.81
THIRD_DESIRED_COLUMN_INDICES = numpy.concatenate((
    numpy.linspace(1360, 1439, num=80, dtype=int),
    numpy.linspace(0, 800, num=801, dtype=int)
))

FOURTH_START_LONGITUDE_DEG_E = 160.07
FOURTH_END_LONGITUDE_DEG_E = 199.99
FOURTH_DESIRED_COLUMN_INDICES = numpy.concatenate((
    numpy.linspace(1360, 1439, num=80, dtype=int),
    numpy.linspace(0, 80, num=81, dtype=int)
))

FIFTH_START_LONGITUDE_DEG_E = 160.07
FIFTH_END_LONGITUDE_DEG_E = -60.21
FIFTH_DESIRED_COLUMN_INDICES = numpy.concatenate((
    numpy.linspace(1360, 1439, num=80, dtype=int),
    numpy.linspace(0, 480, num=481, dtype=int)
))


class Era5UtilsTests(unittest.TestCase):
    """Each method is a unit test for era5_utils.py."""

    def test_desired_longitudes_to_columns_first(self):
        """Ensures correct output from desired_longitudes_to_columns.

        In this case, with first set of inputs.
        """

        these_column_indices = era5_utils.desired_longitudes_to_columns(
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

        these_column_indices = era5_utils.desired_longitudes_to_columns(
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

        these_column_indices = era5_utils.desired_longitudes_to_columns(
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

        these_column_indices = era5_utils.desired_longitudes_to_columns(
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

        these_column_indices = era5_utils.desired_longitudes_to_columns(
            start_longitude_deg_e=FIFTH_START_LONGITUDE_DEG_E,
            end_longitude_deg_e=FIFTH_END_LONGITUDE_DEG_E
        )
        self.assertTrue(numpy.array_equal(
            FIFTH_DESIRED_COLUMN_INDICES, these_column_indices
        ))


if __name__ == '__main__':
    unittest.main()
