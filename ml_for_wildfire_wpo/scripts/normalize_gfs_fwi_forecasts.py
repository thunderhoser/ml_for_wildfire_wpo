"""Normalizes GFS-based FWI forecasts."""

import argparse
from ml_for_wildfire_wpo.io import gfs_daily_io
from ml_for_wildfire_wpo.io import canadian_fwi_io
from ml_for_wildfire_wpo.utils import normalization
from ml_for_wildfire_wpo.utils import gfs_daily_utils

INPUT_DIR_ARG_NAME = 'input_daily_gfs_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
INIT_DATE_ARG_NAME = 'init_date_string'
OUTPUT_DIR_ARG_NAME = 'output_daily_gfs_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of directory with unnormalized GFS-based FWI forecasts.  Files '
    'therein will be found by `gfs_daily_io.find_file` and read by '
    '`gfs_daily_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params.  Will be read by '
    '`canadian_fwi_io.read_normalization_file`.'
)
INIT_DATE_HELP_STRING = (
    'Initialization date (format "yyyymmdd").  This script will process the '
    'model run initialized at 0000 UTC on the given day.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Normalized files will be written here (one '
    'NetCDF file per day) by `gfs_daily_io.write_file`, to exact locations '
    'determined by `gfs_daily_io.find_file`.'
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
    '--' + INIT_DATE_ARG_NAME, type=str, required=True,
    help=INIT_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_daily_gfs_dir_name, normalization_file_name, init_date_string,
         output_daily_gfs_dir_name):
    """Normalizes GFS-based FWI forecasts.

    This is effectively the main method.

    :param input_daily_gfs_dir_name: See documentation at top of file.
    :param normalization_file_name: Same.
    :param init_date_string: Same.
    :param output_daily_gfs_dir_name: Same.
    """

    print('Reading normalization params from: "{0:s}"...'.format(
        normalization_file_name
    ))
    norm_param_table_xarray = canadian_fwi_io.read_normalization_file(
        normalization_file_name
    )

    input_daily_gfs_file_name = gfs_daily_io.find_file(
        directory_name=input_daily_gfs_dir_name,
        init_date_string=init_date_string,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(input_daily_gfs_file_name))
    daily_gfs_table_xarray = gfs_daily_io.read_file(input_daily_gfs_file_name)

    daily_gfs_table_xarray = (
        normalization.normalize_gfs_fwi_forecasts_to_z_scores(
            daily_gfs_table_xarray=daily_gfs_table_xarray,
            z_score_param_table_xarray=norm_param_table_xarray
        )
    )

    output_daily_gfs_file_name = gfs_daily_io.find_file(
        directory_name=output_daily_gfs_dir_name,
        init_date_string=init_date_string,
        raise_error_if_missing=False
    )

    print('Writing data to: "{0:s}"...'.format(output_daily_gfs_file_name))
    t = daily_gfs_table_xarray

    gfs_daily_io.write_file(
        zarr_file_name=output_daily_gfs_file_name,
        data_matrix=t[gfs_daily_utils.DATA_KEY_2D].values,
        latitudes_deg_n=t.coords[gfs_daily_utils.LATITUDE_DIM].values,
        longitudes_deg_e=t.coords[gfs_daily_utils.LONGITUDE_DIM].values,
        field_names=t.coords[gfs_daily_utils.FIELD_DIM].values,
        lead_times_days=t.coords[gfs_daily_utils.LEAD_TIME_DIM].values
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_daily_gfs_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        init_date_string=getattr(INPUT_ARG_OBJECT, INIT_DATE_ARG_NAME),
        output_daily_gfs_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
