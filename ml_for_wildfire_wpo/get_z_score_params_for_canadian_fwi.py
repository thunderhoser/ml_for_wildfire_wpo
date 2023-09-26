"""Computes z-score parameters for Canadian FWI data.

z-score parameters = mean and standard deviation for each variable
"""

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

INPUT_DIR_ARG_NAME = 'input_fwi_dir_name'
FIRST_DATE_ARG_NAME = 'first_valid_date_string'
LAST_DATE_ARG_NAME = 'last_valid_date_string'
OUTPUT_FILE_ARG_NAME = 'output_norm_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`canadian_fwi_io.find_file` and read by `canadian_fwi_io.read_file`.'
)
FIRST_DATE_HELP_STRING = (
    'First valid date (format "yyyymmdd").  Normalization params will be based '
    'on all valid dates in the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

LAST_DATE_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_DATE_ARG_NAME
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by '
    '`canadian_fwi_io.write_normalization_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=FIRST_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=LAST_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_dir_name, first_date_string, last_date_string, output_file_name):
    """Computes z-score parameters for Canadian FWI data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param output_file_name: Same.
    """

    valid_date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )

    fwi_file_names = [
        canadian_fwi_io.find_file(
            directory_name=input_dir_name, valid_date_string=d,
            raise_error_if_missing=True
        ) for d in valid_date_strings
    ]

    z_score_param_table_xarray = normalization.get_z_score_params_for_targets(
        fwi_file_names
    )

    print('Writing z-score parameters to: "{0:s}"...'.format(
        output_file_name
    ))
    canadian_fwi_io.write_normalization_file(
        norm_param_table_xarray=z_score_param_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
