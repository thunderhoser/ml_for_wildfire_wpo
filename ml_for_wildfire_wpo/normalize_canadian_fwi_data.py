"""Normalizes Canadian FWI data."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import canadian_fwi_io
import normalization
import canadian_fwi_utils

INPUT_DIR_ARG_NAME = 'input_fwi_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
START_DATE_ARG_NAME = 'start_date_string'
END_DATE_ARG_NAME = 'end_date_string'
OUTPUT_DIR_ARG_NAME = 'output_fwi_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of directory with unnormalized Canadian FWI data.  Files '
    'therein will be found by `canadian_fwi_io.find_file` and read by '
    '`canadian_fwi_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params.  Will be read by '
    '`canadian_fwi_io.read_normalization_file`.'
)
START_DATE_HELP_STRING = (
    'Start date (format "yyyymmdd").  This script will process model runs '
    'initialized at all dates in the continuous period {0:s}...{1:s}.'
).format(START_DATE_ARG_NAME, END_DATE_ARG_NAME)

END_DATE_HELP_STRING = 'Same as {0:s} but end date.'.format(START_DATE_ARG_NAME)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Normalized files will be written here (one '
    'NetCDF file per day) by `canadian_fwi_io.write_file`, to exact locations '
    'determined by `canadian_fwi_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
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


def _run(input_fwi_dir_name, normalization_file_name, start_date_string,
         end_date_string, output_fwi_dir_name):
    """Normalizes Canadian FWI data.

    This is effectively the main method.

    :param input_fwi_dir_name: See documentation at top of file.
    :param normalization_file_name: Same.
    :param start_date_string: Same.
    :param end_date_string: Same.
    :param output_fwi_dir_name: Same.
    """

    init_date_strings = time_conversion.get_spc_dates_in_range(
        start_date_string, end_date_string
    )

    print('Reading normalization params from: "{0:s}"...'.format(
        normalization_file_name
    ))
    norm_param_table_xarray = canadian_fwi_io.read_normalization_file(
        normalization_file_name
    )

    for this_date_string in init_date_strings:
        input_fwi_file_name = canadian_fwi_io.find_file(
            directory_name=input_fwi_dir_name,
            valid_date_string=this_date_string,
            raise_error_if_missing=True
        )

        print('Reading data from: "{0:s}"...'.format(input_fwi_file_name))
        fwi_table_xarray = canadian_fwi_io.read_file(input_fwi_file_name)

        fwi_table_xarray = normalization.normalize_targets_to_z_scores(
            fwi_table_xarray=fwi_table_xarray,
            z_score_param_table_xarray=norm_param_table_xarray
        )

        output_fwi_file_name = canadian_fwi_io.find_file(
            directory_name=output_fwi_dir_name,
            valid_date_string=this_date_string,
            raise_error_if_missing=False
        )

        print('Writing data to: "{0:s}"...'.format(output_fwi_file_name))
        t = fwi_table_xarray

        canadian_fwi_io.write_file(
            netcdf_file_name=output_fwi_file_name,
            data_matrix=t[canadian_fwi_utils.DATA_KEY].values,
            latitudes_deg_n=t.coords[canadian_fwi_utils.LATITUDE_DIM].values,
            longitudes_deg_e=t.coords[canadian_fwi_utils.LONGITUDE_DIM].values,
            field_names=t.coords[canadian_fwi_utils.FIELD_DIM].values.tolist()
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_fwi_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        start_date_string=getattr(INPUT_ARG_OBJECT, START_DATE_ARG_NAME),
        end_date_string=getattr(INPUT_ARG_OBJECT, END_DATE_ARG_NAME),
        output_fwi_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
