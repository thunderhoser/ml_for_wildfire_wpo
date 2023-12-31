"""Computes z-score parameters for GFS data.

z-score parameters = mean and standard deviation for each variable
"""

import argparse
from ml_for_wildfire_wpo.io import gfs_io
from ml_for_wildfire_wpo.utils import normalization

INPUT_DIR_ARG_NAME = 'input_gfs_dir_name'
FIRST_DATES_ARG_NAME = 'first_init_date_strings'
LAST_DATES_ARG_NAME = 'last_init_date_strings'
OUTPUT_FILE_ARG_NAME = 'output_norm_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`gfs_io.find_file` and read by `gfs_io.read_file`.'
)
FIRST_DATES_HELP_STRING = (
    'List with first GFS initialization date (format "yyyymmdd") for each '
    'continuous period.  Normalization params will be based on all '
    'initialization dates in all periods.'
)
LAST_DATES_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_DATES_ARG_NAME
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by '
    '`gfs_io.write_normalization_file`.'
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
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_dir_name, first_date_strings, last_date_strings, output_file_name):
    """Computes z-score parameters for GFS data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param first_date_strings: Same.
    :param last_date_strings: Same.
    :param output_file_name: Same.
    """

    gfs_file_names = []
    num_periods = len(first_date_strings)

    for i in range(num_periods):
        gfs_file_names += gfs_io.find_files_for_period(
            directory_name=input_dir_name,
            first_init_date_string=first_date_strings[i],
            last_init_date_string=last_date_strings[i],
            raise_error_if_any_missing=False,
            raise_error_if_all_missing=False
        )

    z_score_param_table_xarray = normalization.get_z_score_params_for_gfs(
        gfs_file_names
    )

    print('Writing z-score parameters to: "{0:s}"...'.format(
        output_file_name
    ))
    gfs_io.write_normalization_file(
        norm_param_table_xarray=z_score_param_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_strings=getattr(INPUT_ARG_OBJECT, FIRST_DATES_ARG_NAME),
        last_date_strings=getattr(INPUT_ARG_OBJECT, LAST_DATES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
