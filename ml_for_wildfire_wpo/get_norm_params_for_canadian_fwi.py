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
FIRST_DATES_ARG_NAME = 'first_valid_date_strings'
LAST_DATES_ARG_NAME = 'last_valid_date_strings'
NUM_QUANTILES_ARG_NAME = 'num_quantiles'
NUM_SAMPLE_VALUES_ARG_NAME = 'num_sample_values_per_file'
OUTPUT_FILE_ARG_NAME = 'output_norm_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`canadian_fwi_io.find_file` and read by `canadian_fwi_io.read_file`.'
)
FIRST_DATES_HELP_STRING = (
    'List with first valid date (format "yyyymmdd") for each continuous '
    'period.  Normalization params will be based on all valid dates in all '
    'periods.'
)
LAST_DATES_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_DATES_ARG_NAME
)
NUM_QUANTILES_HELP_STRING = (
    'Number of quantiles to store for each variable.  The quantile levels will '
    'be evenly spaced from 0 to 1 (i.e., the 0th to 100th percentile).'
)
NUM_SAMPLE_VALUES_HELP_STRING = (
    'Number of sample values per file to use for computing quantiles.  This '
    'value will be applied to each variable.'
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
    '--' + FIRST_DATES_ARG_NAME, type=str, nargs='+', required=True,
    help=FIRST_DATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATES_ARG_NAME, type=str, nargs='+', required=True,
    help=LAST_DATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_QUANTILES_ARG_NAME, type=int, required=False, default=1001,
    help=NUM_QUANTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_SAMPLE_VALUES_ARG_NAME, type=int, required=True,
    help=NUM_SAMPLE_VALUES_ARG_NAME
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_dir_name, first_date_strings, last_date_strings,
         num_quantiles, num_sample_values_per_file, output_file_name):
    """Computes z-score parameters for Canadian FWI data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param first_date_strings: Same.
    :param last_date_strings: Same.
    :param num_quantiles: Same.
    :param num_sample_values_per_file: Same.
    :param output_file_name: Same.
    """

    valid_date_strings = []
    num_periods = len(first_date_strings)

    for i in range(num_periods):
        valid_date_strings += time_conversion.get_spc_dates_in_range(
            first_date_strings[i], last_date_strings[i]
        )

    fwi_file_names = [
        canadian_fwi_io.find_file(
            directory_name=input_dir_name, valid_date_string=d,
            raise_error_if_missing=True
        ) for d in valid_date_strings
    ]

    norm_param_table_xarray = (
        normalization.get_normalization_params_for_targets(
            target_file_names=fwi_file_names,
            num_quantiles=num_quantiles,
            num_sample_values_per_file=num_sample_values_per_file
        )
    )

    print('Writing normalization parameters to: "{0:s}"...'.format(
        output_file_name
    ))
    canadian_fwi_io.write_normalization_file(
        norm_param_table_xarray=norm_param_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_strings=getattr(INPUT_ARG_OBJECT, FIRST_DATES_ARG_NAME),
        last_date_strings=getattr(INPUT_ARG_OBJECT, LAST_DATES_ARG_NAME),
        num_quantiles=getattr(INPUT_ARG_OBJECT, NUM_QUANTILES_ARG_NAME),
        num_sample_values_per_file=getattr(
            INPUT_ARG_OBJECT, NUM_SAMPLE_VALUES_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
