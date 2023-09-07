"""Converts Canadian fire-weather indices from TIF to NetCDF format."""

import os
import argparse
import numpy
import rioxarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.io import era5_io
from ml_for_wildfire_wpo.io import canadian_fwi_io
from ml_for_wildfire_wpo.utils import canadian_fwi_utils

FIELD_NAMES_IN_TIF_FILE = [
    '', '', '', '', '',
    canadian_fwi_utils.FFMC_NAME,
    canadian_fwi_utils.DMC_NAME,
    canadian_fwi_utils.DC_NAME,
    canadian_fwi_utils.ISI_NAME,
    canadian_fwi_utils.BUI_NAME,
    canadian_fwi_utils.FWI_NAME,
    canadian_fwi_utils.DSR_NAME
]

FIELD_NAMES_IN_NETCDF_FILE = FIELD_NAMES_IN_TIF_FILE[5:]

INPUT_DIR_ARG_NAME = 'input_tif_dir_name'
START_DATE_ARG_NAME = 'start_date_string'
END_DATE_ARG_NAME = 'end_date_string'
OUTPUT_DIR_ARG_NAME = 'output_netcdf_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing one TIF file per day.  Files '
    'therein will be found by `era5_io.find_file` (except with a slightly '
    'different end-of-file-name).'
)
START_DATE_HELP_STRING = (
    'Start date (format "yyyymmdd").  This script will process all dates in '
    'the continuous period {0:s}...{1:s}.'
).format(START_DATE_ARG_NAME, END_DATE_ARG_NAME)

END_DATE_HELP_STRING = 'Same as {0:s} but end date.'.format(START_DATE_ARG_NAME)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  NetCDF files will be written here (one per '
    'day) by `canadian_fwi_io.write_file`, to exact locations determined by '
    '`canadian_fwi_io.find_file`.'
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, start_date_string, end_date_string, output_dir_name):
    """Converts Canadian fire-weather indices from TIF to NetCDF format.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param start_date_string: Same.
    :param end_date_string: Same.
    :param output_dir_name: Same.
    """

    valid_date_strings = time_conversion.get_spc_dates_in_range(
        start_date_string, end_date_string
    )
    num_fields = len(FIELD_NAMES_IN_NETCDF_FILE)

    for this_date_string in valid_date_strings:
        input_file_name = era5_io.find_file(
            directory_name=input_dir_name, valid_date_string=this_date_string,
            raise_error_if_missing=False
        )
        input_file_name = '{0:s}_fwi.tif'.format(
            os.path.splitext(input_file_name)[0]
        )
        error_checking.assert_file_exists(input_file_name)

        print('Reading data from: "{0:s}"...'.format(input_file_name))
        orig_fwi_table_xarray = rioxarray.open_rasterio(input_file_name)

        latitudes_deg_n = orig_fwi_table_xarray.coords['y'].values
        longitudes_deg_e = orig_fwi_table_xarray.coords['x'].values

        assert len(numpy.unique(numpy.sign(numpy.diff(latitudes_deg_n)))) == 1

        num_grid_rows = len(latitudes_deg_n)
        num_grid_columns = len(longitudes_deg_e)
        data_matrix = numpy.full(
            (num_grid_rows, num_grid_columns, num_fields), numpy.nan
        )

        for j in range(num_fields):
            k = FIELD_NAMES_IN_TIF_FILE.index(FIELD_NAMES_IN_NETCDF_FILE[j])
            data_matrix[..., j] = orig_fwi_table_xarray.values[k, ...]

        if numpy.diff(latitudes_deg_n)[0] < 0:
            latitudes_deg_n = latitudes_deg_n[::-1]
            data_matrix = numpy.flip(data_matrix, axis=0)

        output_file_name = canadian_fwi_io.find_file(
            directory_name=output_dir_name, valid_date_string=this_date_string,
            raise_error_if_missing=False
        )

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        canadian_fwi_io.write_file(
            netcdf_file_name=output_file_name,
            latitudes_deg_n=latitudes_deg_n,
            longitudes_deg_e=longitudes_deg_e,
            field_names=FIELD_NAMES_IN_NETCDF_FILE,
            data_matrix=data_matrix
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        start_date_string=getattr(INPUT_ARG_OBJECT, START_DATE_ARG_NAME),
        end_date_string=getattr(INPUT_ARG_OBJECT, END_DATE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
