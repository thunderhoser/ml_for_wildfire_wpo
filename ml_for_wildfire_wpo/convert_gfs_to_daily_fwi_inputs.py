"""Converts GFS data to daily FWI (fire-weather index) inputs."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import moisture_conversions as moisture_conv
import longitude_conversion as lng_conversion
import error_checking
import gfs_io
import gfs_daily_io
import time_zone_io
import gfs_utils
import gfs_daily_utils
import misc_utils
import time_zone_utils

TOLERANCE = 1e-6
DAYS_TO_SECONDS = 86400

FIELD_NAMES = gfs_daily_utils.ALL_FIELD_NAMES

INPUT_DIR_ARG_NAME = 'input_gfs_dir_name'
FIRST_INIT_DATE_ARG_NAME = 'first_init_date_string'
LAST_INIT_DATE_ARG_NAME = 'last_init_date_string'
MAX_LEAD_TIME_ARG_NAME = 'max_lead_time_days'
OUTPUT_DIR_ARG_NAME = 'output_daily_gfs_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing GFS data at the typical forecast '
    'hours (not daily at local noon).  Files therein will be found by '
    '`gfs_io.find_file` and read by `gfs_io.read_file`.'
)
FIRST_INIT_DATE_HELP_STRING = (
    'Start date (format "yyyymmdd").  This script will process GFS runs '
    'initialized at 0000 UTC on all dates in the the continuous period '
    '{0:s}...{1:s}.'
).format(
    FIRST_INIT_DATE_ARG_NAME, LAST_INIT_DATE_ARG_NAME
)

LAST_INIT_DATE_HELP_STRING = 'Same as {0:s} but end date.'.format(
    FIRST_INIT_DATE_ARG_NAME
)
MAX_LEAD_TIME_HELP_STRING = (
    'Max lead time.  Will convert data for lead times of 1, 2, ..., K days, '
    'where K = {0:s}.'
).format(MAX_LEAD_TIME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Files with daily FWI inputs (weather variables '
    'at local noon) will be written here by `gfs_daily_io.write_file`, to '
    'exact locations determined by `gfs_daily_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_INIT_DATE_ARG_NAME, type=str, required=True,
    help=FIRST_INIT_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_INIT_DATE_ARG_NAME, type=str, required=True,
    help=LAST_INIT_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LEAD_TIME_ARG_NAME, type=int, required=True,
    help=MAX_LEAD_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _convert_one_gfs_run(input_gfs_file_name, time_zone_table_xarray,
                         max_lead_time_days, output_dir_name):
    """Does the dirty work for one GFS run.

    :param input_gfs_file_name: Path to input file, with GFS data at the
        typical forecast hours.
    :param time_zone_table_xarray: xarray table in format returned by
        `time_zone_io.read_file`.
    :param max_lead_time_days: See documentation at top of file.
    :param output_dir_name: Same.
    """

    forecast_lead_times_days = numpy.linspace(
        1, max_lead_time_days, num=max_lead_time_days, dtype=int
    )

    print('Reading data from: "{0:s}"...'.format(input_gfs_file_name))
    gfs_table_xarray = gfs_io.read_file(input_gfs_file_name)

    init_date_string = gfs_io.file_name_to_date(input_gfs_file_name)
    init_date_unix_sec = time_conversion.string_to_unix_sec(
        init_date_string, gfs_io.DATE_FORMAT
    )

    valid_dates_unix_sec = (
        init_date_unix_sec + forecast_lead_times_days * DAYS_TO_SECONDS
    )
    valid_date_strings = [
        time_conversion.unix_sec_to_string(d, gfs_io.DATE_FORMAT)
        for d in valid_dates_unix_sec
    ]

    tzt = time_zone_table_xarray
    num_grid_rows = len(tzt.coords[time_zone_utils.LATITUDE_KEY].values)
    num_grid_columns = len(tzt.coords[time_zone_utils.LONGITUDE_KEY].values)
    num_fields = len(FIELD_NAMES)
    num_lead_times = len(forecast_lead_times_days)

    data_matrix = numpy.full(
        (num_lead_times, num_grid_rows, num_grid_columns, num_fields),
        numpy.nan
    )

    for i in range(num_lead_times):
        valid_time_matrix_unix_sec = (
            time_zone_utils.find_local_noon_at_each_grid_point(
                valid_date_string=valid_date_strings[i],
                time_zone_table_xarray=time_zone_table_xarray
            )
        )

        for k in range(num_fields):
            if FIELD_NAMES[k] == gfs_daily_utils.RELATIVE_HUMIDITY_2METRE_NAME:
                continue

            if FIELD_NAMES[k] == gfs_utils.PRECIP_NAME:
                data_matrix[i, ..., k] = (
                    gfs_utils.read_24hour_precip_different_times(
                        gfs_table_xarray=gfs_table_xarray,
                        init_date_string=init_date_string,
                        valid_time_matrix_unix_sec=valid_time_matrix_unix_sec
                    )
                )
            else:
                data_matrix[i, ..., k] = (
                    gfs_utils.read_nonprecip_field_different_times(
                        gfs_table_xarray=gfs_table_xarray,
                        init_date_string=init_date_string,
                        field_name=FIELD_NAMES[k],
                        valid_time_matrix_unix_sec=valid_time_matrix_unix_sec
                    )
                )

        rh_index = FIELD_NAMES.index(
            gfs_daily_utils.RELATIVE_HUMIDITY_2METRE_NAME
        )

        data_matrix[
            i, ..., rh_index
        ] = moisture_conv.dewpoint_to_relative_humidity(
            dewpoints_kelvins=data_matrix[
                i, ..., FIELD_NAMES.index(gfs_utils.DEWPOINT_2METRE_NAME)
            ],
            temperatures_kelvins=data_matrix[
                i, ..., FIELD_NAMES.index(gfs_utils.TEMPERATURE_2METRE_NAME)
            ],
            total_pressures_pascals=data_matrix[
                i, ..., FIELD_NAMES.index(gfs_utils.SURFACE_PRESSURE_NAME)
            ]
        )

        data_matrix[i, ..., rh_index] = numpy.maximum(
            data_matrix[i, ..., rh_index], 0.
        )
        data_matrix[i, ..., rh_index] = numpy.minimum(
            data_matrix[i, ..., rh_index], 1.
        )

    output_file_name = gfs_daily_io.find_file(
        directory_name=output_dir_name,
        init_date_string=init_date_string,
        raise_error_if_missing=False
    )
    print('Writing daily (local noon) GFS data to: "{0:s}"...'.format(
        output_file_name
    ))

    gfs_daily_io.write_file(
        zarr_file_name=output_file_name,
        data_matrix=data_matrix,
        latitudes_deg_n=
        time_zone_table_xarray[time_zone_utils.LATITUDE_KEY].values,
        longitudes_deg_e=
        time_zone_table_xarray[time_zone_utils.LONGITUDE_KEY].values,
        field_names=FIELD_NAMES,
        lead_times_days=forecast_lead_times_days
    )


def _run(input_dir_name, first_init_date_string, last_init_date_string,
         max_lead_time_days, output_dir_name):
    """Converts GFS data to daily FWI (fire-weather index) inputs.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param first_init_date_string: Same.
    :param last_init_date_string: Same.
    :param max_lead_time_days: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_geq(max_lead_time_days, 1)

    input_gfs_file_names = gfs_io.find_files_for_period(
        directory_name=input_dir_name,
        first_init_date_string=first_init_date_string,
        last_init_date_string=last_init_date_string,
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )

    first_gfs_table_xarray = gfs_io.read_file(input_gfs_file_names[0])
    time_zone_table_xarray = time_zone_io.read_file()

    fgfst = first_gfs_table_xarray
    tzt = time_zone_table_xarray

    desired_row_indices = misc_utils.desired_latitudes_to_rows(
        grid_latitudes_deg_n=tzt.coords[time_zone_utils.LATITUDE_KEY].values,
        start_latitude_deg_n=fgfst.coords[gfs_utils.LATITUDE_DIM].values[0],
        end_latitude_deg_n=fgfst.coords[gfs_utils.LATITUDE_DIM].values[-1]
    )
    tzt = tzt.isel(
        {time_zone_utils.LATITUDE_KEY: desired_row_indices}
    )

    desired_column_indices = misc_utils.desired_longitudes_to_columns(
        grid_longitudes_deg_e=tzt.coords[time_zone_utils.LONGITUDE_KEY].values,
        start_longitude_deg_e=fgfst.coords[gfs_utils.LONGITUDE_DIM].values[0],
        end_longitude_deg_e=fgfst.coords[gfs_utils.LONGITUDE_DIM].values[-1]
    )
    tzt = tzt.isel(
        {time_zone_utils.LONGITUDE_KEY: desired_column_indices}
    )

    assert numpy.allclose(
        tzt.coords[time_zone_utils.LATITUDE_KEY].values,
        fgfst.coords[gfs_utils.LATITUDE_DIM].values,
        atol=TOLERANCE
    )
    assert numpy.allclose(
        lng_conversion.convert_lng_positive_in_west(
            tzt.coords[time_zone_utils.LONGITUDE_KEY].values
        ),
        lng_conversion.convert_lng_positive_in_west(
            fgfst.coords[gfs_utils.LONGITUDE_DIM].values
        ),
        atol=TOLERANCE
    )

    time_zone_table_xarray = tzt

    for this_input_file_name in input_gfs_file_names:
        _convert_one_gfs_run(
            input_gfs_file_name=this_input_file_name,
            time_zone_table_xarray=time_zone_table_xarray,
            max_lead_time_days=max_lead_time_days,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_init_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_DATE_ARG_NAME
        ),
        last_init_date_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_DATE_ARG_NAME
        ),
        max_lead_time_days=getattr(INPUT_ARG_OBJECT, MAX_LEAD_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
