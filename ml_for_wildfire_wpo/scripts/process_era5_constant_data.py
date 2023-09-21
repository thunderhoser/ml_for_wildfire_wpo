"""Processes ERA5 constant fields.

The raw file should be a GRIB file downloaded from the Copernicus Climate Data
Store (https://cds.climate.copernicus.eu/cdsapp#!/dataset/
reanalysis-era5-single-levels?tab=form) with the following options:

- Variables = "angle of subgrid-scale orography,"
              "anisotropy of subgrid-scale orography,"
              "geopotential," "land-sea mask,"
              "slope of subgrid-scale orography,"
              "standard deviation of filtered subgrid-scale orography," and
              "standard deviation of orography"

- Year = any one year
- Month = any one month
- Day = any one day
- Time = any one hour
- Geographical area = full globe
- Format = GRIB

The output will contain the same data in one NetCDF file.
"""

import argparse
from ml_for_wildfire_wpo.io import raw_gfs_io
from ml_for_wildfire_wpo.io import raw_era5_constant_io
from ml_for_wildfire_wpo.io import era5_constant_io

INPUT_FILE_ARG_NAME = 'input_grib_file_name'
START_LATITUDE_ARG_NAME = 'start_latitude_deg_n'
END_LATITUDE_ARG_NAME = 'end_latitude_deg_n'
START_LONGITUDE_ARG_NAME = 'start_longitude_deg_e'
END_LONGITUDE_ARG_NAME = 'end_longitude_deg_e'
WGRIB_EXE_ARG_NAME = 'wgrib_exe_file_name'
TEMPORARY_DIR_ARG_NAME = 'temporary_dir_name'
OUTPUT_FILE_ARG_NAME = 'output_netcdf_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  The format of this file is described in the '
    'docstring at the top of this script.'
)
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
WGRIB_EXE_HELP_STRING = 'Path to wgrib executable.'
TEMPORARY_DIR_HELP_STRING = (
    'Path to temporary directory for text files created by wgrib.'
)
OUTPUT_FILE_HELP_STRING = 'Path to output (NetCDF) file.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
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
    '--' + WGRIB_EXE_ARG_NAME, type=str, required=True,
    help=WGRIB_EXE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TEMPORARY_DIR_ARG_NAME, type=str, required=True,
    help=TEMPORARY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_name, start_latitude_deg_n, end_latitude_deg_n,
         start_longitude_deg_e, end_longitude_deg_e, wgrib_exe_name,
         temporary_dir_name, output_file_name):
    """Processes ERA5 constant fields.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param start_latitude_deg_n: Same.
    :param end_latitude_deg_n: Same.
    :param start_longitude_deg_e: Same.
    :param end_longitude_deg_e: Same.
    :param wgrib_exe_name: Same.
    :param temporary_dir_name: Same.
    :param output_file_name: Same.
    """

    desired_row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=start_latitude_deg_n,
        end_latitude_deg_n=end_latitude_deg_n
    )
    desired_column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=start_longitude_deg_e,
        end_longitude_deg_e=end_longitude_deg_e
    )
    era5_constant_table_xarray = raw_era5_constant_io.read_file(
        grib_file_name=input_file_name,
        desired_row_indices=desired_row_indices,
        desired_column_indices=desired_column_indices,
        wgrib_exe_name=wgrib_exe_name,
        temporary_dir_name=temporary_dir_name
    )

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    era5_constant_io.write_file(
        netcdf_file_name=output_file_name,
        era5_constant_table_xarray=era5_constant_table_xarray
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        start_latitude_deg_n=getattr(INPUT_ARG_OBJECT, START_LATITUDE_ARG_NAME),
        end_latitude_deg_n=getattr(INPUT_ARG_OBJECT, END_LATITUDE_ARG_NAME),
        start_longitude_deg_e=getattr(
            INPUT_ARG_OBJECT, START_LONGITUDE_ARG_NAME
        ),
        end_longitude_deg_e=getattr(INPUT_ARG_OBJECT, END_LONGITUDE_ARG_NAME),
        wgrib_exe_name=getattr(INPUT_ARG_OBJECT, WGRIB_EXE_ARG_NAME),
        temporary_dir_name=getattr(INPUT_ARG_OBJECT, TEMPORARY_DIR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
