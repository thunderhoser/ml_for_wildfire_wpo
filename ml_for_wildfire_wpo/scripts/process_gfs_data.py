"""Processes GFS data.

Each raw file should be a GRIB2 file downloaded from the NOAA High-performance
Storage System (HPSS) with the following options:

- One model run (init time)
- One forecast hour (valid time)
- Global domain
- 0.25-deg resolution
- 3-D variables (at 1000, 900, 800, 700, 600, 500, 400, 300, and 200 mb) =
  Temperature, specific humidity, geopotential height, u-wind, v-wind
- Variables at 2 m above ground level (AGL) = temperature, specific humidity,
  dewpoint
- Variables at 10 m AGL: u-wind, v-wind
- Variables at 0 m AGL: pressure
- Other variables: accumulated precip, accumulated convective precip,
  precipitable water, snow depth, water-equivalent snow depth, downward
  shortwave radiative flux, upward shortwave radiative flux, downward
  longwave radiative flux, upward longwave radiative flux, CAPE, soil
  temperature, volumetric soil-moisture fraction, vegetation type, soil type

The output will contain the same data, in zarr format, with one file per model
run (init time).
"""

import os
import copy
import warnings
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml_for_wildfire_wpo.io import gfs_io
from ml_for_wildfire_wpo.io import raw_gfs_io
from ml_for_wildfire_wpo.io import raw_ncar_gfs_io
from ml_for_wildfire_wpo.utils import gfs_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FORECAST_HOURS = numpy.array([
    0, 6, 12, 18, 24, 30, 36, 42, 48,
    60, 72, 84, 96, 108, 120,
    144, 168, 192, 216, 240, 264, 288, 312, 336
], dtype=int)

INPUT_DIR_ARG_NAME = 'input_grib2_dir_name'
START_DATE_ARG_NAME = 'start_date_string'
END_DATE_ARG_NAME = 'end_date_string'
START_LATITUDE_ARG_NAME = 'start_latitude_deg_n'
END_LATITUDE_ARG_NAME = 'end_latitude_deg_n'
START_LONGITUDE_ARG_NAME = 'start_longitude_deg_e'
END_LONGITUDE_ARG_NAME = 'end_longitude_deg_e'
WGRIB2_EXE_ARG_NAME = 'wgrib2_exe_file_name'
TEMPORARY_DIR_ARG_NAME = 'temporary_dir_name'
IS_NCAR_FORMAT_ARG_NAME = 'is_ncar_format'
OUTPUT_DIR_ARG_NAME = 'output_zarr_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing one GRIB2 file per model run (init '
    'time) and forecast hour (lead time).  Files therein will be found by '
    '`raw_gfs_io.find_file`.'
)
START_DATE_HELP_STRING = (
    'Start date (format "yyyymmdd").  This script will process model runs '
    'initialized at all dates in the continuous period {0:s}...{1:s}.'
).format(START_DATE_ARG_NAME, END_DATE_ARG_NAME)

END_DATE_HELP_STRING = 'Same as {0:s} but end date.'.format(START_DATE_ARG_NAME)
START_LATITUDE_HELP_STRING = (
    'Start latitude.  This script will process all latitudes in the '
    'contiguous domain {0:s}...{1:s}.'
).format(START_LATITUDE_ARG_NAME, END_LATITUDE_ARG_NAME)

END_LATITUDE_HELP_STRING = 'Same as {0:s} but end latitude.'.format(
    START_LATITUDE_ARG_NAME
)
START_LONGITUDE_HELP_STRING = (
    'Start longitude.  This script will process all longitudes in the '
    'contiguous domain {0:s}...{1:s}.  This domain may cross the International '
    'Date Line.'
).format(START_LONGITUDE_ARG_NAME, END_LONGITUDE_ARG_NAME)

END_LONGITUDE_HELP_STRING = 'Same as {0:s} but end longitude.'.format(
    START_LONGITUDE_ARG_NAME
)
WGRIB2_EXE_HELP_STRING = 'Path to wgrib2 executable.'
TEMPORARY_DIR_HELP_STRING = (
    'Path to temporary directory for text files created by wgrib2.'
)
IS_NCAR_FORMAT_HELP_STRING = (
    'Boolean flag.  If 1, will except all raw data in NCAR format, in which '
    'case this script will call raw_ncar_gfs_io.py instead of raw_gfs_io.py.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Processed files will be written here (one '
    'zarr file per model run) by `gfs_io.write_file`, to exact locations '
    'determined by `gfs_io.find_file`.'
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
    '--' + START_LATITUDE_ARG_NAME, type=float, required=True,
    help=START_LATITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + END_LATITUDE_ARG_NAME, type=float, required=True,
    help=END_LATITUDE_HELP_STRING
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
    '--' + WGRIB2_EXE_ARG_NAME, type=str, required=True,
    help=WGRIB2_EXE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TEMPORARY_DIR_ARG_NAME, type=str, required=True,
    help=TEMPORARY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + IS_NCAR_FORMAT_ARG_NAME, type=int, required=False, default=0,
    help=IS_NCAR_FORMAT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, start_date_string, end_date_string,
         start_latitude_deg_n, end_latitude_deg_n, start_longitude_deg_e,
         end_longitude_deg_e, wgrib2_exe_name, temporary_dir_name,
         is_ncar_format, output_dir_name):
    """Processes GFS data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param start_date_string: Same.
    :param end_date_string: Same.
    :param start_latitude_deg_n: Same.
    :param end_latitude_deg_n: Same.
    :param start_longitude_deg_e: Same.
    :param end_longitude_deg_e: Same.
    :param wgrib2_exe_name: Same.
    :param temporary_dir_name: Same.
    :param is_ncar_format: Same.
    :param output_dir_name: Same.
    """

    init_date_strings = time_conversion.get_spc_dates_in_range(
        start_date_string, end_date_string
    )
    desired_row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=start_latitude_deg_n,
        end_latitude_deg_n=end_latitude_deg_n
    )
    desired_column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=start_longitude_deg_e,
        end_longitude_deg_e=end_longitude_deg_e
    )

    num_forecast_hours = len(FORECAST_HOURS)

    for this_date_string in init_date_strings:
        gfs_tables_xarray = [None] * num_forecast_hours
        found_0hour_file = True

        for k in range(num_forecast_hours):
            if is_ncar_format:
                input_file_name = raw_ncar_gfs_io.find_file(
                    directory_name=input_dir_name,
                    init_date_string=this_date_string,
                    forecast_hour=FORECAST_HOURS[k],
                    raise_error_if_missing=FORECAST_HOURS[k] > 0
                )

                if FORECAST_HOURS[k] == 0:
                    found_0hour_file = os.path.isfile(input_file_name)

                if not os.path.isfile(input_file_name):
                    continue

                gfs_tables_xarray[k] = raw_ncar_gfs_io.read_file(
                    grib2_file_name=input_file_name,
                    desired_row_indices=desired_row_indices,
                    desired_column_indices=desired_column_indices,
                    wgrib2_exe_name=wgrib2_exe_name,
                    temporary_dir_name=temporary_dir_name
                )
            else:
                input_file_name = raw_gfs_io.find_file(
                    directory_name=input_dir_name,
                    init_date_string=this_date_string,
                    forecast_hour=FORECAST_HOURS[k],
                    raise_error_if_missing=True
                )
                gfs_tables_xarray[k] = raw_gfs_io.read_file(
                    grib2_file_name=input_file_name,
                    desired_row_indices=desired_row_indices,
                    desired_column_indices=desired_column_indices,
                    wgrib2_exe_name=wgrib2_exe_name,
                    temporary_dir_name=temporary_dir_name
                )

            print(SEPARATOR_STRING)

        if not found_0hour_file:
            missing_file_name = raw_ncar_gfs_io.find_file(
                directory_name=input_dir_name,
                init_date_string=this_date_string,
                forecast_hour=0, raise_error_if_missing=False
            )

            warning_string = (
                'POTENTIAL ERROR (but probably not): Could not find file at '
                'forecast hour 0.  Expected at: "{0:s}".  All values at '
                'forecast hour 0 will be NaN.'
            ).format(missing_file_name)

            warnings.warn(warning_string)

            k = numpy.where(FORECAST_HOURS == 0)[0][0]
            k_other = numpy.where(FORECAST_HOURS > 0)[0][0]
            gfs_tables_xarray[k] = copy.deepcopy(gfs_tables_xarray[k_other])

            gfs_tables_xarray[k] = gfs_tables_xarray[k].assign_coords({
                gfs_utils.FORECAST_HOUR_DIM: numpy.array([0], dtype=int)
            })

            nan_matrix = numpy.full(
                gfs_tables_xarray[k][gfs_utils.DATA_KEY_2D].values.shape,
                numpy.nan
            )

            gfs_tables_xarray[k] = gfs_tables_xarray[k].assign({
                gfs_utils.DATA_KEY_2D: (
                    gfs_tables_xarray[k][gfs_utils.DATA_KEY_2D].dims,
                    nan_matrix
                )
            })

        gfs_table_xarray = gfs_utils.concat_over_forecast_hours(
            gfs_tables_xarray
        )
        output_file_name = gfs_io.find_file(
            directory_name=output_dir_name,
            init_date_string=this_date_string,
            raise_error_if_missing=False
        )

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        gfs_io.write_file(
            zarr_file_name=output_file_name, gfs_table_xarray=gfs_table_xarray
        )
        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        start_date_string=getattr(INPUT_ARG_OBJECT, START_DATE_ARG_NAME),
        end_date_string=getattr(INPUT_ARG_OBJECT, END_DATE_ARG_NAME),
        start_latitude_deg_n=getattr(INPUT_ARG_OBJECT, START_LATITUDE_ARG_NAME),
        end_latitude_deg_n=getattr(INPUT_ARG_OBJECT, END_LATITUDE_ARG_NAME),
        start_longitude_deg_e=getattr(
            INPUT_ARG_OBJECT, START_LONGITUDE_ARG_NAME
        ),
        end_longitude_deg_e=getattr(INPUT_ARG_OBJECT, END_LONGITUDE_ARG_NAME),
        wgrib2_exe_name=getattr(INPUT_ARG_OBJECT, WGRIB2_EXE_ARG_NAME),
        temporary_dir_name=getattr(INPUT_ARG_OBJECT, TEMPORARY_DIR_ARG_NAME),
        is_ncar_format=bool(getattr(INPUT_ARG_OBJECT, IS_NCAR_FORMAT_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
