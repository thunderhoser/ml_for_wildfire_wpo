"""Processes ERA5 data.

The output will contain, for each grid point and time step, the following daily
quantities required to compute the Canadian fire-weather indices:

- Temperature at 2 m above ground level (AGL) at noon local standard time (LST)
- Relative humidity at 2 m AGL at noon LST
- Wind speed at 10 m AGL at noon LST
- Accumulated precipitation over the 24 hours before noon LST

The main input file (argument name "input_era5_file_name") should be a grib file
downloaded from the Copernicus Climate Data Store
(https://cds.climate.copernicus.eu/cdsapp#!/dataset/
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

import argparse
import numpy
import xarray
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import moisture_conversions as moisture_conv
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.utils import time_zone_utils

# TODO(thunderhoser): This script does not consider non-integer time zones,
# i.e., where the UTC offset is not an integer.  An example is Newfoundland.
# In the future I might interpolate between the two nearest UTC times in the
# ERA5 data.

TOLERANCE = 1e-6

HOURS_TO_SECONDS = 3600
SECONDS_TO_HOURS = 1. / 3600

DATE_FORMAT = time_conversion.SPC_DATE_FORMAT
GRIB_TIME_FORMAT = '%y%m%d%H'

GRID_LATITUDES_DEG_N = numpy.linspace(17, 73, num=225, dtype=float)
GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E = numpy.linspace(
    -180, 179.75, num=1440, dtype=float
)
GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E = (
    lng_conversion.convert_lng_positive_in_west(
        GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E
    )
)

LONGITUDE_SPACING_DEG = 0.25

DEWPOINT_2METRE_NAME = 'dewpoint_2m_agl_kelvins'
RELATIVE_HUMIDITY_2METRE_NAME = 'relative_humidity_2m_agl'
TEMPERATURE_2METRE_NAME = 'temperature_2m_agl_kelvins'
SURFACE_PRESSURE_NAME = 'surface_pressure_pascals'
U_WIND_10METRE_NAME = 'u_wind_10m_agl_m_s01'
V_WIND_10METRE_NAME = 'v_wind_10m_agl_m_s01'
HOURLY_PRECIP_NAME = 'hourly_precip_metres'

ALL_FIELD_NAMES = [
    DEWPOINT_2METRE_NAME, RELATIVE_HUMIDITY_2METRE_NAME,
    TEMPERATURE_2METRE_NAME, SURFACE_PRESSURE_NAME,
    U_WIND_10METRE_NAME, V_WIND_10METRE_NAME, HOURLY_PRECIP_NAME
]

FIELD_NAME_TO_GRIB_NAME = {
    DEWPOINT_2METRE_NAME: '2D',
    RELATIVE_HUMIDITY_2METRE_NAME: None,
    TEMPERATURE_2METRE_NAME: '2T',
    SURFACE_PRESSURE_NAME: 'SP',
    U_WIND_10METRE_NAME: '10U',
    V_WIND_10METRE_NAME: '10V',
    HOURLY_PRECIP_NAME: 'TP'
}

ERA5_FILE_ARG_NAME = 'input_era5_grib_file_name'
TIME_ZONE_FILE_ARG_NAME = 'input_time_zone_file_name'
START_DATE_ARG_NAME = 'start_date_string'
END_DATE_ARG_NAME = 'end_date_string'
START_LONGITUDE_ARG_NAME = 'start_longitude_deg_e'
END_LONGITUDE_ARG_NAME = 'end_longitude_deg_e'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ERA5_FILE_HELP_STRING = (
    'Path to grib file with ERA5 data, described in the docstring at the top '
    'of this script.'
)
TIME_ZONE_FILE_HELP_STRING = (
    'Path to time-zone file.  Will be read by `time_zone_io.read_file`.'
)
START_DATE_HELP_STRING = (
    'Start date (format "yyyymmdd").  This script will process all dates in '
    'the continuous period {0:s}...{1:s}.'
).format(START_DATE_ARG_NAME, END_DATE_ARG_NAME)

END_DATE_HELP_STRING = 'Same as {0:s} but end date.'.format(START_DATE_ARG_NAME)
START_LONGITUDE_HELP_STRING = (
    'Start longitude.  This script will process all longitudes in the '
    'contiguous domain {0:s}...{1:s}.  This domain may cross the International '
    'Date Line.'
).format(START_LONGITUDE_ARG_NAME, END_LONGITUDE_ARG_NAME)

END_LONGITUDE_HELP_STRING = 'Same as {0:s} but end longitude.'.format(
    START_LONGITUDE_ARG_NAME
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Processed files will be written here (one '
    'NetCDF per day) by `era5_io.write_file`, to exact locations determined by '
    '`era5_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ERA5_FILE_ARG_NAME, type=str, required=True,
    help=ERA5_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TIME_ZONE_FILE_ARG_NAME, type=str, required=True,
    help=TIME_ZONE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + START_DATE_ARG_NAME, type=str, required=True,
    help=START_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + END_DATE_ARG_NAME, type=str, required=True,
    help=END_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + START_LONGITUDE_ARG_NAME, type=float, required=True,
    help=START_LONGITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + END_LONGITUDE_ARG_NAME, type=float, required=True,
    help=END_LONGITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_one_nonprecip_field(
        era5_grib_file_name, field_name, valid_time_matrix_unix_sec,
        desired_column_indices):
    """Extracts one field (not precipitation) from ERA5 file.

    M = number of rows in grid
    N = number of columns in grid

    :param era5_grib_file_name: See documentation at top of script.
    :param field_name: Name of field (weather variable) to extract.
    :param valid_time_matrix_unix_sec: M-by-N numpy array of valid times.
    :param desired_column_indices: length-N numpy array with indices of desired
        grid columns from ERA5 data.
    :return: data_matrix: M-by-N numpy array of data values.
    """

    grib_field_name = FIELD_NAME_TO_GRIB_NAME[field_name]
    unique_valid_times_unix_sec = numpy.unique(valid_time_matrix_unix_sec)
    data_matrix = numpy.full(valid_time_matrix_unix_sec.shape, numpy.nan)

    for i in range(len(unique_valid_times_unix_sec)):
        these_indices = numpy.where(
            valid_time_matrix_unix_sec == unique_valid_times_unix_sec[i]
        )
        this_time_string = time_conversion.unix_sec_to_string(
            unique_valid_times_unix_sec[i], GRIB_TIME_FORMAT
        )
        this_search_string = '{0:s}:{1:s}'.format(
            this_time_string, grib_field_name
        )

        print('Reading line "{0:s}" from GRIB file: "{1:s}"...'.format(
            this_search_string, era5_grib_file_name
        ))
        this_data_matrix = grib_io.read_field_from_grib_file(
            grib_file_name=era5_grib_file_name,
            field_name_grib1=this_search_string,
            num_grid_rows=len(GRID_LATITUDES_DEG_N),
            num_grid_columns=len(GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E)
        )
        this_data_matrix = this_data_matrix[:, desired_column_indices]
        this_data_matrix = numpy.flip(this_data_matrix, axis=0)

        data_matrix[these_indices] = this_data_matrix[these_indices] + 0.

    assert not numpy.any(numpy.isnan(data_matrix))

    return data_matrix


def _precip_hour_to_grib_search_string(precip_hour_start_time_unix_sec):
    """Converts precipitation hour to search string for GRIB file.

    :param precip_hour_start_time_unix_sec: Start of the hour.
    :return: grib_search_string: String that can be used to find the relevant
        line (containing 1-hour precip accumulation) in GRIB file.
    """

    # TODO(thunderhoser): Make sure that input is a multiple of 3600.

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
        FIELD_NAME_TO_GRIB_NAME[HOURLY_PRECIP_NAME],
        grib_start_time_diff_hours,
        grib_start_time_diff_hours + 1
    )


def _get_24hour_precip_field(
        era5_grib_file_name, valid_time_matrix_unix_sec,
        desired_column_indices):
    """Extracts 24-hour precipitation from ERA5 file.

    M = number of rows in grid
    N = number of columns in grid

    :param era5_grib_file_name: See documentation at top of script.
    :param valid_time_matrix_unix_sec: M-by-N numpy array of valid times (ending
        times of 24-hour accumulation period).
    :param desired_column_indices: length-N numpy array with indices of desired
        grid columns from ERA5 data.
    :return: precip_matrix_24hour_metres: M-by-N numpy array of precip
        accumulations.
    """

    unique_valid_times_unix_sec = numpy.unique(valid_time_matrix_unix_sec)
    unique_rounded_valid_times_unix_sec = number_rounding.floor_to_nearest(
        unique_valid_times_unix_sec, HOURS_TO_SECONDS
    ).astype(int)

    hour_offsets_sec = (
        HOURS_TO_SECONDS * numpy.linspace(1, 24, num=24, dtype=int)
    )
    precip_hour_start_times_unix_sec = numpy.concatenate(
        [v - hour_offsets_sec for v in unique_rounded_valid_times_unix_sec]
    )
    unique_precip_hour_start_times_unix_sec = numpy.unique(
        precip_hour_start_times_unix_sec
    )

    num_start_times = len(unique_precip_hour_start_times_unix_sec)
    num_grid_rows = valid_time_matrix_unix_sec.shape[0]
    num_grid_columns = valid_time_matrix_unix_sec.shape[1]
    precip_matrix_with_time_metres = numpy.full(
        (num_start_times, num_grid_rows, num_grid_columns), numpy.nan
    )

    for k in range(num_start_times):
        this_search_string = _precip_hour_to_grib_search_string(
            unique_precip_hour_start_times_unix_sec[k]
        )

        print('Reading line "{0:s}" from GRIB file: "{1:s}"...'.format(
            this_search_string, era5_grib_file_name
        ))
        this_data_matrix = grib_io.read_field_from_grib_file(
            grib_file_name=era5_grib_file_name,
            field_name_grib1=this_search_string,
            num_grid_rows=len(GRID_LATITUDES_DEG_N),
            num_grid_columns=len(GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E)
        )
        this_data_matrix = this_data_matrix[:, desired_column_indices]
        this_data_matrix = numpy.flip(this_data_matrix, axis=0)

        precip_matrix_with_time_metres[k, ...] = this_data_matrix + 0.

    precip_matrix_24hour_metres = numpy.full(
        (num_grid_rows, num_grid_columns), numpy.nan
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

    assert not numpy.any(numpy.isnan(precip_matrix_24hour_metres))

    return precip_matrix_24hour_metres


def _run(era5_grib_file_name, time_zone_file_name, start_date_string,
         end_date_string, start_longitude_deg_e, end_longitude_deg_e,
         output_dir_name):
    """Processes ERA5 data.

    This is effectively the main method.

    :param era5_grib_file_name: See documentation at top of file.
    :param time_zone_file_name: Same.
    :param start_date_string: Same.
    :param end_date_string: Same.
    :param start_longitude_deg_e: Same.
    :param end_longitude_deg_e: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    valid_date_strings = time_conversion.get_spc_dates_in_range(
        start_date_string, end_date_string
    )

    start_longitude_deg_e = number_rounding.floor_to_nearest(
        start_longitude_deg_e, LONGITUDE_SPACING_DEG
    )
    end_longitude_deg_e = number_rounding.ceiling_to_nearest(
        end_longitude_deg_e, LONGITUDE_SPACING_DEG
    )
    error_checking.assert_is_greater(
        numpy.absolute(start_longitude_deg_e - end_longitude_deg_e),
        TOLERANCE
    )

    start_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        start_longitude_deg_e, allow_nan=False
    )
    end_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        end_longitude_deg_e, allow_nan=False
    )

    if end_longitude_deg_e > start_longitude_deg_e:
        are_longitudes_positive_in_west = True
    else:
        start_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            start_longitude_deg_e, allow_nan=False
        )
        end_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            end_longitude_deg_e, allow_nan=False
        )
        are_longitudes_positive_in_west = False

    if are_longitudes_positive_in_west:
        grid_longitudes_deg_e = GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    else:
        grid_longitudes_deg_e = GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E

    num_longitudes = 1 + int(numpy.round(
        (end_longitude_deg_e - start_longitude_deg_e) / LONGITUDE_SPACING_DEG
    ))
    desired_longitudes_deg_e = numpy.linspace(
        start_longitude_deg_e, end_longitude_deg_e, num=num_longitudes
    )
    desired_column_indices = numpy.array([
        numpy.where(
            numpy.absolute(grid_longitudes_deg_e - d) <= TOLERANCE
        )[0][0]
        for d in desired_longitudes_deg_e
    ], dtype=int)

    # Do actual stuff.

    # TODO(thunderhoser): Modularize this!
    print('Reading time zones from: "{0:s}"...'.format(time_zone_file_name))
    time_zone_table_xarray = xarray.open_dataset(time_zone_file_name)

    assert numpy.allclose(
        time_zone_table_xarray.coords[time_zone_utils.LATITUDE_KEY].values,
        GRID_LATITUDES_DEG_N,
        atol=TOLERANCE
    )

    time_zone_table_xarray = time_zone_utils.subset_by_longitude(
        time_zone_table_xarray=time_zone_table_xarray,
        desired_longitudes_deg_e=desired_longitudes_deg_e
    )[0]

    valid_time_matrix_unix_sec = (
        time_zone_utils.find_local_noon_at_each_grid_point(
            valid_date_string=valid_date_strings[0],
            time_zone_table_xarray=time_zone_table_xarray
        )
    )

    num_grid_rows = valid_time_matrix_unix_sec.shape[0]
    num_grid_columns = valid_time_matrix_unix_sec.shape[1]
    num_fields = len(ALL_FIELD_NAMES)
    data_matrix = numpy.full(
        (num_grid_rows, num_grid_columns, num_fields), numpy.nan
    )

    for k in range(num_fields):
        if FIELD_NAME_TO_GRIB_NAME[ALL_FIELD_NAMES[k]] is None:
            continue

        if ALL_FIELD_NAMES[k] == HOURLY_PRECIP_NAME:
            data_matrix[..., k] = _get_24hour_precip_field(
                era5_grib_file_name=era5_grib_file_name,
                valid_time_matrix_unix_sec=valid_time_matrix_unix_sec,
                desired_column_indices=desired_column_indices
            )
        else:
            data_matrix[..., k] = _get_one_nonprecip_field(
                era5_grib_file_name=era5_grib_file_name,
                field_name=ALL_FIELD_NAMES[k],
                valid_time_matrix_unix_sec=valid_time_matrix_unix_sec,
                desired_column_indices=desired_column_indices
            )

    rh_index = ALL_FIELD_NAMES.index(RELATIVE_HUMIDITY_2METRE_NAME)

    data_matrix[
        ..., rh_index
    ] = moisture_conv.dewpoint_to_relative_humidity(
        dewpoints_kelvins=
        data_matrix[..., ALL_FIELD_NAMES.index(DEWPOINT_2METRE_NAME)],
        temperatures_kelvins=
        data_matrix[..., ALL_FIELD_NAMES.index(TEMPERATURE_2METRE_NAME)],
        total_pressures_pascals=
        data_matrix[..., ALL_FIELD_NAMES.index(SURFACE_PRESSURE_NAME)]
    )

    data_matrix[..., rh_index] = numpy.maximum(
        data_matrix[..., rh_index], 0.
    )
    data_matrix[..., rh_index] = numpy.minimum(
        data_matrix[..., rh_index], 1.
    )

    LATITUDE_DIM = 'latitude_deg_n'
    LONGITUDE_DIM = 'longitude_deg_e'
    FIELD_DIM = 'field_name'
    DATA_KEY = 'data'

    coord_dict = {
        LATITUDE_DIM: time_zone_table_xarray.coords['latitude_deg_n'].values,
        LONGITUDE_DIM: time_zone_table_xarray.coords['longitude_deg_e'].values,
        FIELD_DIM: ALL_FIELD_NAMES
    }
    main_data_dict = {
        DATA_KEY: (
            (LATITUDE_DIM, LONGITUDE_DIM, FIELD_DIM), data_matrix
        )
    }
    era5_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=coord_dict
    )

    # TODO(thunderhoser): Change this.

    # TODO(thunderhoser): Still need to see what happens with backwards longitude range.
    dummy_file_name = '{0:s}/era5_data.nc'.format(output_dir_name)
    print('Writing data to: "{0:s}"...'.format(dummy_file_name))
    era5_table_xarray.to_netcdf(
        path=dummy_file_name, mode='w', format='NETCDF3_64BIT'
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        era5_grib_file_name=getattr(INPUT_ARG_OBJECT, ERA5_FILE_ARG_NAME),
        time_zone_file_name=getattr(INPUT_ARG_OBJECT, TIME_ZONE_FILE_ARG_NAME),
        start_date_string=getattr(INPUT_ARG_OBJECT, START_DATE_ARG_NAME),
        end_date_string=getattr(INPUT_ARG_OBJECT, END_DATE_ARG_NAME),
        start_longitude_deg_e=getattr(INPUT_ARG_OBJECT,
                                      START_LONGITUDE_ARG_NAME),
        end_longitude_deg_e=getattr(INPUT_ARG_OBJECT, END_LONGITUDE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
