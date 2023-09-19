"""USE ONCE AND DESTROY."""

import os
import sys
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import gfs_io
import raw_gfs_io
import gfs_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_grib2_dir_name'
START_DATE_ARG_NAME = 'start_date_string'
END_DATE_ARG_NAME = 'end_date_string'
START_LATITUDE_ARG_NAME = 'start_latitude_deg_n'
END_LATITUDE_ARG_NAME = 'end_latitude_deg_n'
START_LONGITUDE_ARG_NAME = 'start_longitude_deg_e'
END_LONGITUDE_ARG_NAME = 'end_longitude_deg_e'
WGRIB2_EXE_ARG_NAME = 'wgrib2_exe_file_name'
TEMPORARY_DIR_ARG_NAME = 'temporary_dir_name'
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, start_date_string, end_date_string,
         start_latitude_deg_n, end_latitude_deg_n, start_longitude_deg_e,
         end_longitude_deg_e, wgrib2_exe_name, temporary_dir_name,
         output_dir_name):
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

    for this_date_string in init_date_strings:
        input_file_name = raw_gfs_io.find_file(
            directory_name=input_dir_name,
            init_date_string=this_date_string,
            forecast_hour=0, raise_error_if_missing=True
        )
        new_gfs_table_xarray = raw_gfs_io.read_file(
            grib2_file_name=input_file_name,
            desired_row_indices=desired_row_indices,
            desired_column_indices=desired_column_indices,
            wgrib2_exe_name=wgrib2_exe_name,
            temporary_dir_name=temporary_dir_name
        )

        print(SEPARATOR_STRING)

        output_file_name = gfs_io.find_file(
            directory_name=output_dir_name,
            init_date_string=this_date_string,
            raise_error_if_missing=True
        )
        gfs_table_xarray = gfs_io.read_file(output_file_name)

        k = numpy.where(
            gfs_table_xarray.coords[gfs_utils.FORECAST_HOUR_DIM].values == 0
        )[0][0]
        data_matrix_2d = gfs_table_xarray[gfs_utils.DATA_KEY_2D].values
        data_matrix_2d[k, ...] = (
            new_gfs_table_xarray[gfs_utils.DATA_KEY_2D].values[0, ...]
        )

        data_dict = {}
        for var_name in gfs_table_xarray.data_vars:
            if var_name == gfs_utils.DATA_KEY_2D:
                data_dict[var_name] = (
                    gfs_table_xarray[var_name].dims,
                    data_matrix_2d
                )
            else:
                data_dict[var_name] = (
                    gfs_table_xarray[var_name].dims,
                    gfs_table_xarray[var_name].values
                )

        coord_dict = {}
        for coord_name in gfs_table_xarray.coords:
            coord_dict[coord_name] = gfs_table_xarray.coords[coord_name].values

        gfs_table_xarray = xarray.Dataset(
            data_vars=data_dict, coords=coord_dict
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
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
