"""Normalizes ERA5 constants."""

import argparse
from ml_for_wildfire_wpo.io import era5_constant_io
from ml_for_wildfire_wpo.utils import normalization

INPUT_FILE_ARG_NAME = 'input_era5_constant_file_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
OUTPUT_FILE_ARG_NAME = 'output_era5_constant_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (with ERA5 data in physical units).  Will be read by '
    '`era5_constant_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params.  Will be read by '
    '`era5_constant_io.read_normalization_file`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (with ERA5 data in z-score units).  Will be written '
    'by `era5_constant_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_name, normalization_file_name, output_file_name):
    """Normalizes ERA5 constants.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param normalization_file_name: Same.
    :param output_file_name: Same.
    """

    print('Reading normalization params from: "{0:s}"...'.format(
        normalization_file_name
    ))
    norm_param_table_xarray = era5_constant_io.read_normalization_file(
        normalization_file_name
    )

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    era5_constant_table_xarray = era5_constant_io.read_file(input_file_name)

    era5_constant_table_xarray = (
        normalization.normalize_era5_constants_to_z_scores(
            era5_constant_table_xarray=era5_constant_table_xarray,
            z_score_param_table_xarray=norm_param_table_xarray
        )
    )

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    era5_constant_io.write_file(
        era5_constant_table_xarray=era5_constant_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
