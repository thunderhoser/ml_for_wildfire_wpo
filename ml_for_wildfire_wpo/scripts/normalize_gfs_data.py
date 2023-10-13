"""Normalizes GFS data."""

import argparse
from gewittergefahr.gg_utils import time_conversion
from ml_for_wildfire_wpo.io import gfs_io
from ml_for_wildfire_wpo.utils import normalization

INPUT_DIR_ARG_NAME = 'input_gfs_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
START_DATE_ARG_NAME = 'start_date_string'
END_DATE_ARG_NAME = 'end_date_string'
OUTPUT_DIR_ARG_NAME = 'output_gfs_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of directory with unnormalized GFS data (in physical units).  Files '
    'therein will be found by `gfs_io.find_file` and read by '
    '`gfs_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params.  Will be read by '
    '`gfs_io.read_normalization_file`.'
)
START_DATE_HELP_STRING = (
    'Start date (format "yyyymmdd").  This script will process model runs '
    'initialized at all dates in the continuous period {0:s}...{1:s}.'
).format(START_DATE_ARG_NAME, END_DATE_ARG_NAME)

END_DATE_HELP_STRING = 'Same as {0:s} but end date.'.format(START_DATE_ARG_NAME)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Normalized files will be written here (one '
    'zarr file per model run) by `gfs_io.write_file`, to exact locations '
    'determined by `gfs_io.find_file`.'
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


def _run(input_gfs_dir_name, normalization_file_name, start_date_string,
         end_date_string, output_gfs_dir_name):
    """Normalizes GFS data.

    This is effectively the main method.

    :param input_gfs_dir_name: See documentation at top of file.
    :param normalization_file_name: Same.
    :param start_date_string: Same.
    :param end_date_string: Same.
    :param output_gfs_dir_name: Same.
    """

    init_date_strings = time_conversion.get_spc_dates_in_range(
        start_date_string, end_date_string
    )

    print('Reading normalization params from: "{0:s}"...'.format(
        normalization_file_name
    ))
    norm_param_table_xarray = gfs_io.read_normalization_file(
        normalization_file_name
    )

    for this_date_string in init_date_strings:
        input_gfs_file_name = gfs_io.find_file(
            directory_name=input_gfs_dir_name,
            init_date_string=this_date_string,
            raise_error_if_missing=True
        )

        print('Reading data from: "{0:s}"...'.format(input_gfs_file_name))
        gfs_table_xarray = gfs_io.read_file(input_gfs_file_name)

        gfs_table_xarray = normalization.normalize_gfs_data_to_z_scores(
            gfs_table_xarray=gfs_table_xarray,
            z_score_param_table_xarray=norm_param_table_xarray
        )

        output_gfs_file_name = gfs_io.find_file(
            directory_name=output_gfs_dir_name,
            init_date_string=this_date_string,
            raise_error_if_missing=False
        )

        print('Writing data to: "{0:s}"...'.format(output_gfs_file_name))
        gfs_io.write_file(
            gfs_table_xarray=gfs_table_xarray,
            zarr_file_name=output_gfs_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_gfs_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        start_date_string=getattr(INPUT_ARG_OBJECT, START_DATE_ARG_NAME),
        end_date_string=getattr(INPUT_ARG_OBJECT, END_DATE_ARG_NAME),
        output_gfs_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
