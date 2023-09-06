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
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import moisture_conversions as moisture_conv
from ml_for_wildfire_wpo.io import time_zone_io
from ml_for_wildfire_wpo.io import raw_era5_io
from ml_for_wildfire_wpo.io import era5_io
from ml_for_wildfire_wpo.utils import time_zone_utils
from ml_for_wildfire_wpo.utils import era5_utils

TOLERANCE = 1e-6
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

ALL_FIELD_NAMES = era5_utils.ALL_FIELD_NAMES

INPUT_DIR_ARG_NAME = 'input_grib_dir_name'
START_DATE_ARG_NAME = 'start_date_string'
END_DATE_ARG_NAME = 'end_date_string'
START_LONGITUDE_ARG_NAME = 'start_longitude_deg_e'
END_LONGITUDE_ARG_NAME = 'end_longitude_deg_e'
WGRIB_EXE_ARG_NAME = 'wgrib_exe_file_name'
TEMPORARY_DIR_ARG_NAME = 'temporary_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_netcdf_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing one GRIB file per year.  Files '
    'therein will be found by `raw_era5_io.find_file`.'
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
WGRIB_EXE_HELP_STRING = 'Path to wgrib executable.'
TEMPORARY_DIR_HELP_STRING = (
    'Path to temporary directory for text files created by wgrib.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Processed files will be written here (one '
    'NetCDF per day) by `era5_io.write_file`, to exact locations determined by '
    '`era5_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
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
    '--' + WGRIB_EXE_ARG_NAME, type=str, required=True,
    help=WGRIB_EXE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TEMPORARY_DIR_ARG_NAME, type=str, required=True,
    help=TEMPORARY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, start_date_string, end_date_string,
         start_longitude_deg_e, end_longitude_deg_e, wgrib_exe_name,
         temporary_dir_name, output_dir_name):
    """Processes ERA5 data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param start_date_string: Same.
    :param end_date_string: Same.
    :param start_longitude_deg_e: Same.
    :param end_longitude_deg_e: Same.
    :param wgrib_exe_name: Same.
    :param temporary_dir_name: Same.
    :param output_dir_name: Same.
    """

    valid_date_strings = time_conversion.get_spc_dates_in_range(
        start_date_string, end_date_string
    )
    desired_column_indices = era5_utils.desired_longitudes_to_columns(
        start_longitude_deg_e=start_longitude_deg_e,
        end_longitude_deg_e=end_longitude_deg_e
    )

    time_zone_table_xarray = time_zone_io.read_file()
    assert numpy.allclose(
        time_zone_table_xarray.coords[time_zone_utils.LATITUDE_KEY].values,
        era5_utils.GRID_LATITUDES_DEG_N,
        atol=TOLERANCE
    )
    assert numpy.allclose(
        time_zone_table_xarray.coords[time_zone_utils.LONGITUDE_KEY].values,
        era5_utils.GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E,
        atol=TOLERANCE
    )

    time_zone_table_xarray = time_zone_table_xarray.isel(
        {time_zone_utils.LONGITUDE_KEY: desired_column_indices}
    )

    for this_date_string in valid_date_strings:
        valid_time_matrix_unix_sec = (
            time_zone_utils.find_local_noon_at_each_grid_point(
                valid_date_string=this_date_string,
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
            if raw_era5_io.FIELD_NAME_TO_GRIB_NAME[ALL_FIELD_NAMES[k]] is None:
                continue

            if ALL_FIELD_NAMES[k] == era5_utils.HOURLY_PRECIP_NAME:
                data_matrix[..., k] = raw_era5_io.read_24hour_precip_field(
                    grib_directory_name=input_dir_name,
                    valid_time_matrix_unix_sec=valid_time_matrix_unix_sec,
                    desired_column_indices=desired_column_indices,
                    wgrib_exe_name=wgrib_exe_name,
                    temporary_dir_name=temporary_dir_name
                )
            else:
                data_matrix[..., k] = raw_era5_io.read_one_nonprecip_field(
                    grib_directory_name=input_dir_name,
                    field_name=ALL_FIELD_NAMES[k],
                    valid_time_matrix_unix_sec=valid_time_matrix_unix_sec,
                    desired_column_indices=desired_column_indices,
                    wgrib_exe_name=wgrib_exe_name,
                    temporary_dir_name=temporary_dir_name
                )

        rh_index = ALL_FIELD_NAMES.index(
            era5_utils.RELATIVE_HUMIDITY_2METRE_NAME
        )

        data_matrix[
            ..., rh_index
        ] = moisture_conv.dewpoint_to_relative_humidity(
            dewpoints_kelvins=data_matrix[
                ..., ALL_FIELD_NAMES.index(era5_utils.DEWPOINT_2METRE_NAME)
            ],
            temperatures_kelvins=data_matrix[
                ..., ALL_FIELD_NAMES.index(era5_utils.TEMPERATURE_2METRE_NAME)
            ],
            total_pressures_pascals=data_matrix[
                ..., ALL_FIELD_NAMES.index(era5_utils.SURFACE_PRESSURE_NAME)
            ]
        )

        data_matrix[..., rh_index] = numpy.maximum(
            data_matrix[..., rh_index], 0.
        )
        data_matrix[..., rh_index] = numpy.minimum(
            data_matrix[..., rh_index], 1.
        )

        this_output_file_name = era5_io.find_file(
            directory_name=output_dir_name, valid_date_string=this_date_string,
            raise_error_if_missing=False
        )
        print('\nWriting processed data to: "{0:s}"...'.format(
            this_output_file_name
        ))

        era5_io.write_file(
            netcdf_file_name=this_output_file_name,
            data_matrix=data_matrix,
            latitudes_deg_n=
            time_zone_table_xarray[time_zone_utils.LATITUDE_KEY].values,
            longitudes_deg_e=
            time_zone_table_xarray[time_zone_utils.LONGITUDE_KEY].values,
            field_names=ALL_FIELD_NAMES
        )

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        start_date_string=getattr(INPUT_ARG_OBJECT, START_DATE_ARG_NAME),
        end_date_string=getattr(INPUT_ARG_OBJECT, END_DATE_ARG_NAME),
        start_longitude_deg_e=getattr(
            INPUT_ARG_OBJECT, START_LONGITUDE_ARG_NAME
        ),
        end_longitude_deg_e=getattr(INPUT_ARG_OBJECT, END_LONGITUDE_ARG_NAME),
        wgrib_exe_name=getattr(INPUT_ARG_OBJECT, WGRIB_EXE_ARG_NAME),
        temporary_dir_name=getattr(INPUT_ARG_OBJECT, TEMPORARY_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
