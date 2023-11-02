"""Converts GFS forecasts of Canadian FWIs from TIF to NetCDF format.

FWI = fire-weather index
"""

import os
import sys
import argparse
import numpy
import rioxarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import error_checking
import gfs_daily_io
import gfs_daily_utils

DATE_FORMAT = '%Y%m%d'

FIELD_NAMES_IN_TIF_FILE = [
    '', '', '', '', '',
    gfs_daily_utils.FFMC_NAME,
    gfs_daily_utils.DMC_NAME,
    gfs_daily_utils.DC_NAME,
    gfs_daily_utils.ISI_NAME,
    gfs_daily_utils.BUI_NAME,
    gfs_daily_utils.FWI_NAME,
    gfs_daily_utils.DSR_NAME
]

FIELD_NAMES_IN_NETCDF_FILE = FIELD_NAMES_IN_TIF_FILE[5:]

INPUT_DIR_ARG_NAME = 'input_tif_dir_name'
INIT_DATE_ARG_NAME = 'init_date_string'
MAX_LEAD_TIME_ARG_NAME = 'max_lead_time_days'
OUTPUT_DIR_ARG_NAME = 'output_netcdf_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing one TIF file per model solution '
    '(i.e., one TIF file for every pair of init time & valid time).'
)
INIT_DATE_HELP_STRING = (
    'Initialization date (format "yyyymmdd").  This script will process FWI '
    'forecasts from the GFS run initialized at 0000 UTC on this date.'
)
MAX_LEAD_TIME_HELP_STRING = (
    'Max lead time.  For the given initialization date, this script will '
    'process FWI forecasts for lead times of 1, 2, ..., K days -- where K is {0:s}.'
).format(MAX_LEAD_TIME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  NetCDF files will be written here (one per '
    'day) by `gfs_daily_io.write_file`, to exact locations determined by '
    '`gfs_daily_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_DATE_ARG_NAME, type=str, required=True,
    help=INIT_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LEAD_TIME_ARG_NAME, type=int, required=True,
    help=MAX_LEAD_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _find_input_file(directory_name, init_date_string, lead_time_days,
                     raise_error_if_missing=True):
    """Finds TIF file with GFS-based FWI forecasts for 1 init time & 1 lead time.

    :param directory_name: Path to input directory.
    :param init_date_string: Initialization date (format "yyyymmdd").
    :param lead_time_days: Lead time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: tif_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    _ = time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)
    error_checking.assert_is_integer(lead_time_days)
    error_checking.assert_is_geq(lead_time_days, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    tif_file_name = (
        '{0:s}/init={1:s}/gfs_fwi_forecast_init={1:s}_lead={2:02d}days.tif'
    ).format(
        directory_name, init_date_string, lead_time_days
    )

    if raise_error_if_missing and not os.path.isfile(tif_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            tif_file_name
        )
        raise ValueError(error_string)

    return tif_file_name


def _run(input_dir_name, init_date_string, max_lead_time_days, output_dir_name):
    """Converts GFS forecasts of Canadian FWIs from TIF to NetCDF format.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param init_date_string: Same.
    :param max_lead_time_days: Same.
    :param output_dir_name: Same.
    """

    lead_times_days = numpy.linspace(
        1, max_lead_time_days, num=max_lead_time_days, dtype=int
    )

    input_file_names = [
        _find_input_file(
            directory_name=input_dir_name,
            init_date_string=init_date_string,
            lead_time_days=d,
            raise_error_if_missing=True
        ) for d in lead_times_days
    ]

    data_matrix = numpy.array([])
    latitudes_deg_n = numpy.array([])
    longitudes_deg_e = numpy.array([])

    num_lead_times = len(lead_times_days)
    num_fields = len(FIELD_NAMES_IN_NETCDF_FILE)

    for i in range(num_lead_times):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        orig_fwi_table_xarray = rioxarray.open_rasterio(input_file_names[i])

        latitudes_deg_n = orig_fwi_table_xarray.coords['y'].values
        longitudes_deg_e = orig_fwi_table_xarray.coords['x'].values

        assert len(numpy.unique(numpy.sign(numpy.diff(latitudes_deg_n)))) == 1

        if i == 0:
            num_grid_rows = len(latitudes_deg_n)
            num_grid_columns = len(longitudes_deg_e)
            data_matrix = numpy.full(
                (num_lead_times, num_grid_rows, num_grid_columns, num_fields),
                numpy.nan
            )

        for j in range(num_fields):
            k = FIELD_NAMES_IN_TIF_FILE.index(FIELD_NAMES_IN_NETCDF_FILE[j])
            data_matrix[i, ..., j] = orig_fwi_table_xarray.values[k, ...]

    if numpy.diff(latitudes_deg_n)[0] < 0:
        latitudes_deg_n = latitudes_deg_n[::-1]
        data_matrix = numpy.flip(data_matrix, axis=1)

    output_file_name = gfs_daily_io.find_file(
        directory_name=output_dir_name, init_date_string=init_date_string,
        raise_error_if_missing=False
    )

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    gfs_daily_io.write_file(
        zarr_file_name=output_file_name,
        data_matrix=data_matrix,
        latitudes_deg_n=latitudes_deg_n,
        longitudes_deg_e=longitudes_deg_e,
        field_names=FIELD_NAMES_IN_NETCDF_FILE,
        lead_times_days=lead_times_days
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        init_date_string=getattr(INPUT_ARG_OBJECT, INIT_DATE_ARG_NAME),
        max_lead_time_days=getattr(INPUT_ARG_OBJECT, MAX_LEAD_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
