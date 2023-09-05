"""Input/output methods for raw ERA5 reanalysis data.

Each raw file should be a GRIB file downloaded from the Copernicus Climate Data
Store (https://cds.climate.copernicus.eu/cdsapp#!/dataset/
reanalysis-era5-single-levels?tab=form) with the following options:

- Variables = "10-m u-component of wind", "10-m v-component of wind",
  "2m dewpoint temperature", "2m temperature", "Surface pressure",
  and "Total precipitation"
- Year = preferably just one year
- Month = preferably all
- Day = preferably all
- Time = all
- Geographical area = all longitudes (-180...180) and latitudes 17-73 deg N
- Format = GRIB
"""

import numpy
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.utils import era5_utils

HOURS_TO_SECONDS = 3600
SECONDS_TO_HOURS = 1. / 3600
DAYS_TO_HOURS = 24

GRIB_TIME_FORMAT = '%y%m%d%H'

FIELD_NAME_TO_GRIB_NAME = {
    era5_utils.DEWPOINT_2METRE_NAME: '2D',
    era5_utils.RELATIVE_HUMIDITY_2METRE_NAME: None,
    era5_utils.TEMPERATURE_2METRE_NAME: '2T',
    era5_utils.SURFACE_PRESSURE_NAME: 'SP',
    era5_utils.U_WIND_10METRE_NAME: '10U',
    era5_utils.V_WIND_10METRE_NAME: '10V',
    era5_utils.HOURLY_PRECIP_NAME: 'TP'
}


def _precip_hour_to_grib_search_string(precip_hour_start_time_unix_sec):
    """Converts precipitation hour to search string for GRIB file.

    :param precip_hour_start_time_unix_sec: Start of the hour.
    :return: grib_search_string: String that can be used to find the relevant
        line (containing 1-hour precip accumulation) in GRIB file.
    """

    error_checking.assert_equals(
        numpy.mod(precip_hour_start_time_unix_sec, HOURS_TO_SECONDS),
        0
    )

    grib_start_time_unix_sec = number_rounding.floor_to_nearest(
        precip_hour_start_time_unix_sec + 6 * HOURS_TO_SECONDS,
        12 * HOURS_TO_SECONDS
    )
    grib_start_time_unix_sec -= 6 * HOURS_TO_SECONDS
    grib_start_time_unix_sec = int(numpy.round(grib_start_time_unix_sec))

    grib_start_time_diff_hours = int(numpy.round(
        SECONDS_TO_HOURS *
        (precip_hour_start_time_unix_sec - grib_start_time_unix_sec)
    ))

    grib_start_time_string = time_conversion.unix_sec_to_string(
        grib_start_time_unix_sec, GRIB_TIME_FORMAT
    )

    return '{0:s}:{1:s}:sfc:{2:d}-{3:d}hr'.format(
        grib_start_time_string,
        FIELD_NAME_TO_GRIB_NAME[era5_utils.HOURLY_PRECIP_NAME],
        grib_start_time_diff_hours,
        grib_start_time_diff_hours + 1
    )


def read_one_nonprecip_field(
        grib_file_name, field_name, valid_time_matrix_unix_sec,
        desired_column_indices):
    """Extracts one field (anything except precip) from GRIB file.

    M = number of rows in grid
    N = number of columns in grid

    :param grib_file_name: Path to input file.  For more details on the
        required format, see documentation at the top of this module.
    :param field_name: Name of field (weather variable) to extract.
    :param valid_time_matrix_unix_sec: M-by-N numpy array of valid times.
    :param desired_column_indices: length-N numpy array with indices of desired
        grid columns from ERA5 data.
    :return: data_matrix: M-by-N numpy array of data values.
    """

    # Check input args.
    error_checking.assert_file_exists(grib_file_name)

    error_checking.assert_is_numpy_array(
        valid_time_matrix_unix_sec, num_dimensions=2
    )
    error_checking.assert_is_integer_numpy_array(valid_time_matrix_unix_sec)

    num_grid_rows_desired = valid_time_matrix_unix_sec.shape[0]
    num_grid_columns_desired = valid_time_matrix_unix_sec.shape[1]

    error_checking.assert_is_numpy_array(
        desired_column_indices,
        exact_dimensions=numpy.array([num_grid_columns_desired], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)

    # Do actual stuff.
    grib_field_name = FIELD_NAME_TO_GRIB_NAME[field_name]
    unique_valid_times_unix_sec = numpy.unique(valid_time_matrix_unix_sec)
    data_matrix = numpy.full(
        (num_grid_rows_desired, num_grid_columns_desired), numpy.nan
    )

    for i in range(len(unique_valid_times_unix_sec)):
        these_indices = numpy.where(
            valid_time_matrix_unix_sec == unique_valid_times_unix_sec[i]
        )

        # TODO(thunderhoser): This converts any non-integer UTC offset (e.g.,
        # the half-hour thing for Newfoundland) into an integer offset.  I might
        # eventually want to handle non-integer UTC offsets more rigorously
        # (e.g., for Newfoundland, interpolate between the two nearest hours).
        this_time_string = time_conversion.unix_sec_to_string(
            unique_valid_times_unix_sec[i], GRIB_TIME_FORMAT
        )
        this_search_string = '{0:s}:{1:s}'.format(
            this_time_string, grib_field_name
        )

        print('Reading line "{0:s}" from GRIB file: "{1:s}"...'.format(
            this_search_string, grib_file_name
        ))
        this_data_matrix = grib_io.read_field_from_grib_file(
            grib_file_name=grib_file_name,
            field_name_grib1=this_search_string,
            num_grid_rows=len(era5_utils.GRID_LATITUDES_DEG_N),
            num_grid_columns=
            len(era5_utils.GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E)
        )
        this_data_matrix = this_data_matrix[:, desired_column_indices]
        this_data_matrix = numpy.flip(this_data_matrix, axis=0)

        data_matrix[these_indices] = this_data_matrix[these_indices] + 0.

    assert not numpy.any(numpy.isnan(data_matrix))

    return data_matrix


def read_24hour_precip_field(
        grib_file_name, valid_time_matrix_unix_sec, desired_column_indices):
    """Reads one field (24-hour accumulated precip) from GRIB file.

    M = number of rows in grid
    N = number of columns in grid

    :param grib_file_name: See doc for `read_one_nonprecip_field`.
    :param valid_time_matrix_unix_sec: Same.
    :param desired_column_indices: Same.
    :return: precip_matrix_24hour_metres: M-by-N numpy array of precip
        accumulations.
    """

    # Check input args.
    error_checking.assert_file_exists(grib_file_name)

    error_checking.assert_is_numpy_array(
        valid_time_matrix_unix_sec, num_dimensions=2
    )
    error_checking.assert_is_integer_numpy_array(valid_time_matrix_unix_sec)

    num_grid_rows_desired = valid_time_matrix_unix_sec.shape[0]
    num_grid_columns_desired = valid_time_matrix_unix_sec.shape[1]

    error_checking.assert_is_numpy_array(
        desired_column_indices,
        exact_dimensions=numpy.array([num_grid_columns_desired], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)

    # Do actual stuff.
    unique_valid_times_unix_sec = numpy.unique(valid_time_matrix_unix_sec)

    # TODO(thunderhoser): This converts any non-integer UTC offset (e.g.,
    # the half-hour thing for Newfoundland) into an integer offset.  I might
    # eventually want to handle non-integer UTC offsets more rigorously.
    unique_rounded_valid_times_unix_sec = number_rounding.floor_to_nearest(
        unique_valid_times_unix_sec, HOURS_TO_SECONDS
    ).astype(int)

    hour_offsets_sec = HOURS_TO_SECONDS * numpy.linspace(
        1, DAYS_TO_HOURS, num=DAYS_TO_HOURS, dtype=int
    )
    precip_hour_start_times_unix_sec = numpy.concatenate(
        [v - hour_offsets_sec for v in unique_rounded_valid_times_unix_sec]
    )
    unique_precip_hour_start_times_unix_sec = numpy.unique(
        precip_hour_start_times_unix_sec
    )

    num_start_times = len(unique_precip_hour_start_times_unix_sec)
    precip_matrix_with_time_metres = numpy.full(
        (num_start_times, num_grid_rows_desired, num_grid_columns_desired),
        numpy.nan
    )

    for k in range(num_start_times):
        this_search_string = _precip_hour_to_grib_search_string(
            unique_precip_hour_start_times_unix_sec[k]
        )

        print('Reading line "{0:s}" from GRIB file: "{1:s}"...'.format(
            this_search_string, grib_file_name
        ))
        this_data_matrix = grib_io.read_field_from_grib_file(
            grib_file_name=grib_file_name,
            field_name_grib1=this_search_string,
            num_grid_rows=len(era5_utils.GRID_LATITUDES_DEG_N),
            num_grid_columns=
            len(era5_utils.GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E)
        )
        this_data_matrix = this_data_matrix[:, desired_column_indices]
        this_data_matrix = numpy.flip(this_data_matrix, axis=0)

        precip_matrix_with_time_metres[k, ...] = this_data_matrix + 0.

    precip_matrix_24hour_metres = numpy.full(
        (num_grid_rows_desired, num_grid_columns_desired), numpy.nan
    )

    for i in range(len(unique_valid_times_unix_sec)):
        these_pixel_indices = numpy.where(
            valid_time_matrix_unix_sec == unique_valid_times_unix_sec[i]
        )

        precip_matrix_24hour_metres[these_pixel_indices] = 0.

        for k in range(len(hour_offsets_sec)):
            this_hour_index = numpy.where(
                unique_precip_hour_start_times_unix_sec ==
                unique_rounded_valid_times_unix_sec[i] - hour_offsets_sec[k]
            )[0][0]

            precip_matrix_24hour_metres[these_pixel_indices] += (
                precip_matrix_with_time_metres[this_hour_index, ...][
                    these_pixel_indices
                ]
            )

    assert numpy.all(precip_matrix_24hour_metres > 0.)

    return precip_matrix_24hour_metres
