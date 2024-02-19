"""Computes z-score parameters for ERA5 constants.

z-score parameters = mean and standard deviation for each variable
"""

import argparse
from ml_for_wildfire_wpo.io import era5_constant_io
from ml_for_wildfire_wpo.utils import normalization

INPUT_FILE_ARG_NAME = 'input_era5_constant_file_name'
OUTPUT_FILE_ARG_NAME = 'output_norm_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `era5_constant_io.read_file`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by '
    '`era5_constant_io.write_normalization_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_name, output_file_name):
    """Computes z-score parameters for ERA5 constants.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param output_file_name: Same.
    """

    z_score_param_table_xarray = (
        normalization.get_z_score_params_for_era5_constants(input_file_name)
    )

    print('Writing z-score parameters to: "{0:s}"...'.format(
        output_file_name
    ))
    era5_constant_io.write_normalization_file(
        norm_param_table_xarray=z_score_param_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
