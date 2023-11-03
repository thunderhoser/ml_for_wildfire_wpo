"""Processes GFS data from NCAR.

Each raw file should be a GRIB2 file downloaded from the NCAR Research Data
Archive (https://rda.ucar.edu/datasets/ds084.1/) with the following
specifications:

- One model run (init time) at 0000 UTC
- One forecast hour (valid time)
- Global domain
- 0.25-deg resolution
- Four variables: accumulated precip, accumulated convective precip,
  vegetation type, soil type

The output will contain the same data, in zarr format, with one file per model
run (init time).
"""

import os
import sys
import copy
import warnings
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import gfs_io
import raw_gfs_io
import raw_ncar_gfs_io
import gfs_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FORECAST_HOURS = numpy.array([
    0, 6, 12, 18, 24, 30, 36, 42, 48,
    60, 72, 84, 96, 108, 120,
    144, 168, 192, 216, 240, 264, 288, 312, 336
], dtype=int)

FIELD_NAMES_3D = gfs_utils.ALL_3D_FIELD_NAMES
FIELD_NAMES_2D = gfs_utils.ALL_2D_FIELD_NAMES

MAIN_INPUT_DIR_ARG_NAME = 'main_input_grib2_dir_name'
INPUT_PRECIP_DIR_ARG_NAME = 'input_grib2_precip_dir_name'
START_DATE_ARG_NAME = 'start_date_string'
END_DATE_ARG_NAME = 'end_date_string'
START_LATITUDE_ARG_NAME = 'start_latitude_deg_n'
END_LATITUDE_ARG_NAME = 'end_latitude_deg_n'
START_LONGITUDE_ARG_NAME = 'start_longitude_deg_e'
END_LONGITUDE_ARG_NAME = 'end_longitude_deg_e'
WGRIB2_EXE_ARG_NAME = 'wgrib2_exe_file_name'
TEMPORARY_DIR_ARG_NAME = 'temporary_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_zarr_dir_name'

MAIN_INPUT_DIR_HELP_STRING = (
    'Name of main input directory, containing one GRIB2 file per model run '
    '(init time) and forecast hour (lead time).  Files therein will be found '
    'by `raw_ncar_gfs_io.find_file`.'
)
INPUT_PRECIP_DIR_HELP_STRING = (
    'Name of input directory for precip only.  As for the main input '
    'directory, should contain one GRIB2 file per model run and forecast hour, '
    'to be found by `raw_ncar_gfs_io.find_file`.  Unlike the main input '
    'directory, this directory must contain data for every single available '
    'GFS forecast hour, not just select ones.  If you do *not* want to try '
    'reading incremental (per-time-step) precip from a separate directory in '
    'case precip data are missing from the main directory, then just leave '
    'this argument alone.'
)
START_DATE_HELP_STRING = (
    'Start date (format "yyyymmdd").  This script will process model runs '
    'initialized at all dates in the continuous period {0:s}...{1:s}.'
).format(START_DATE_ARG_NAME, END_DATE_ARG_NAME)

END_DATE_HELP_STRING = 'Same as {0:s} but end date.'.format(START_DATE_ARG_NAME)
START_LATITUDE_HELP_STRING = (
    'Start latitude.  This script will process all latitudes in the '
    'contiguous domain {0:s}...{1:s}.'
).format(START_LATITUDE_ARG_NAME, END_LATITUDE_ARG_NAME)

END_LATITUDE_HELP_STRING = 'Same as {0:s} but end latitude.'.format(
    START_LATITUDE_ARG_NAME
)
START_LONGITUDE_HELP_STRING = (
    'Start longitude.  This script will process all longitudes in the '
    'contiguous domain {0:s}...{1:s}.  This domain may cross the International '
    'Date Line.'
).format(START_LONGITUDE_ARG_NAME, END_LONGITUDE_ARG_NAME)

END_LONGITUDE_HELP_STRING = 'Same as {0:s} but end longitude.'.format(
    START_LONGITUDE_ARG_NAME
)
WGRIB2_EXE_HELP_STRING = 'Path to wgrib2 executable.'
TEMPORARY_DIR_HELP_STRING = (
    'Path to temporary directory for text files created by wgrib2.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Processed files will be written here (one '
    'zarr file per model run) by `gfs_io.write_file`, to exact locations '
    'determined by `gfs_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MAIN_INPUT_DIR_ARG_NAME, type=str, required=True,
    help=MAIN_INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_PRECIP_DIR_ARG_NAME, type=str, required=False, default='',
    help=INPUT_PRECIP_DIR_HELP_STRING
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
    '--' + START_LATITUDE_ARG_NAME, type=float, required=True,
    help=START_LATITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + END_LATITUDE_ARG_NAME, type=float, required=True,
    help=END_LATITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + START_LONGITUDE_ARG_NAME, type=float, required=True,
    help=START_LONGITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + END_LONGITUDE_ARG_NAME, type=float, required=True,
    help=END_LONGITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WGRIB2_EXE_ARG_NAME, type=str, required=True,
    help=WGRIB2_EXE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TEMPORARY_DIR_ARG_NAME, type=str, required=True,
    help=TEMPORARY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_incremental_precip_1init(
        input_dir_name, init_date_string,
        desired_row_indices, desired_column_indices,
        wgrib2_exe_name, temporary_dir_name):
    """Reads incremental precip for one init time (i.e., one model run).

    "Incremental precip" = between two forecast hours, rather than over the
    entire model run

    :param input_dir_name: See documentation at top of file.
    :param init_date_string: Initialization date of model run (format
        "yyyymmdd").
    :param desired_row_indices: 1-D numpy array with indices of desired grid
        rows.
    :param desired_column_indices: 1-D numpy array with indices of desired grid
        columns.
    :param wgrib2_exe_name: See documentation at top of file.
    :param temporary_dir_name: Same.
    :return: gfs_table_xarray: xarray table with incremental precip data.
        Metadata and variable names should make this table self-explanatory.
    """

    forecast_hours = set(gfs_utils.ALL_FORECAST_HOURS.tolist())
    forecast_hours.remove(384)
    forecast_hours = numpy.array(list(forecast_hours), dtype=int)

    input_file_names = [
        raw_ncar_gfs_io.find_file(
            directory_name=input_dir_name,
            init_date_string=init_date_string,
            forecast_hour=h,
            raise_error_if_missing=False
        )
        for h in forecast_hours
    ]

    if not all([os.path.isfile(f) for f in input_file_names]):
        return None

    num_forecast_hours = len(forecast_hours)
    gfs_tables_xarray = [None] * num_forecast_hours

    for k in range(num_forecast_hours):
        gfs_tables_xarray[k] = raw_ncar_gfs_io.read_file(
            grib2_file_name=input_file_names[k],
            desired_row_indices=desired_row_indices,
            desired_column_indices=desired_column_indices,
            wgrib2_exe_name=wgrib2_exe_name,
            temporary_dir_name=temporary_dir_name,
            read_incremental_precip=True
        )

        data_matrix = gfs_tables_xarray[k][gfs_utils.DATA_KEY_2D].values
        if numpy.all(numpy.isnan(data_matrix)) and forecast_hours[k] > 0:
            return None

    return gfs_utils.concat_over_forecast_hours(gfs_tables_xarray)


def _run(main_input_dir_name, input_precip_dir_name,
         start_date_string, end_date_string,
         start_latitude_deg_n, end_latitude_deg_n, start_longitude_deg_e,
         end_longitude_deg_e, wgrib2_exe_name, temporary_dir_name,
         output_dir_name):
    """Processes GFS data.

    This is effectively the main method.

    :param main_input_dir_name: See documentation at top of file.
    :param input_precip_dir_name: Same.
    :param start_date_string: Same.
    :param end_date_string: Same.
    :param start_latitude_deg_n: Same.
    :param end_latitude_deg_n: Same.
    :param start_longitude_deg_e: Same.
    :param end_longitude_deg_e: Same.
    :param wgrib2_exe_name: Same.
    :param temporary_dir_name: Same.
    :param output_dir_name: Same.
    """

    if input_precip_dir_name == '':
        input_precip_dir_name = None

    init_date_strings = time_conversion.get_spc_dates_in_range(
        start_date_string, end_date_string
    )
    desired_row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=start_latitude_deg_n,
        end_latitude_deg_n=end_latitude_deg_n
    )
    desired_column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=start_longitude_deg_e,
        end_longitude_deg_e=end_longitude_deg_e
    )

    num_forecast_hours = len(FORECAST_HOURS)

    for this_date_string in init_date_strings:
        if input_precip_dir_name is None:
            precip_table_xarray = None
        else:
            precip_table_xarray = _read_incremental_precip_1init(
                input_dir_name=input_precip_dir_name,
                init_date_string=this_date_string,
                desired_row_indices=desired_row_indices,
                desired_column_indices=desired_column_indices,
                wgrib2_exe_name=wgrib2_exe_name,
                temporary_dir_name=temporary_dir_name
            )

        gfs_tables_xarray = [None] * num_forecast_hours
        found_0hour_file = True

        for k in range(num_forecast_hours):
            input_file_name = raw_ncar_gfs_io.find_file(
                directory_name=main_input_dir_name,
                init_date_string=this_date_string,
                forecast_hour=FORECAST_HOURS[k],
                raise_error_if_missing=FORECAST_HOURS[k] > 0
            )

            if FORECAST_HOURS[k] == 0:
                found_0hour_file = os.path.isfile(input_file_name)

            if not os.path.isfile(input_file_name):
                continue

            gfs_tables_xarray[k] = raw_ncar_gfs_io.read_file(
                grib2_file_name=input_file_name,
                desired_row_indices=desired_row_indices,
                desired_column_indices=desired_column_indices,
                wgrib2_exe_name=wgrib2_exe_name,
                temporary_dir_name=temporary_dir_name,
                read_incremental_precip=False
            )

            print(SEPARATOR_STRING)

        if not found_0hour_file:
            missing_file_name = raw_ncar_gfs_io.find_file(
                directory_name=main_input_dir_name,
                init_date_string=this_date_string,
                forecast_hour=0, raise_error_if_missing=False
            )

            warning_string = (
                'POTENTIAL ERROR (but probably not): Could not find file at '
                'forecast hour 0.  Expected at: "{0:s}".  All values at '
                'forecast hour 0 will be NaN.'
            ).format(missing_file_name)

            warnings.warn(warning_string)

            k = numpy.where(FORECAST_HOURS == 0)[0][0]
            k_other = numpy.where(FORECAST_HOURS > 0)[0][0]
            gfs_tables_xarray[k] = copy.deepcopy(gfs_tables_xarray[k_other])

            gfs_tables_xarray[k] = gfs_tables_xarray[k].assign_coords({
                gfs_utils.FORECAST_HOUR_DIM: numpy.array([0], dtype=int)
            })

            nan_matrix = numpy.full(
                gfs_tables_xarray[k][gfs_utils.DATA_KEY_2D].values.shape,
                numpy.nan
            )

            gfs_tables_xarray[k] = gfs_tables_xarray[k].assign({
                gfs_utils.DATA_KEY_2D: (
                    gfs_tables_xarray[k][gfs_utils.DATA_KEY_2D].dims,
                    nan_matrix
                )
            })

        gfs_table_xarray = gfs_utils.concat_over_forecast_hours(
            gfs_tables_xarray
        )

        if precip_table_xarray is not None:
            precip_table_xarray = gfs_utils.precip_from_incremental_to_full_run(
                precip_table_xarray
            )
            precip_table_xarray = gfs_utils.subset_by_forecast_hour(
                gfs_table_xarray=precip_table_xarray,
                desired_forecast_hours=FORECAST_HOURS
            )

            main_data_matrix_2d = gfs_table_xarray[gfs_utils.DATA_KEY_2D].values
            aux_data_matrix_2d = (
                precip_table_xarray[gfs_utils.DATA_KEY_2D].values
            )

            for this_field_name in [
                    gfs_utils.PRECIP_NAME, gfs_utils.CONVECTIVE_PRECIP_NAME
            ]:
                k_main = numpy.where(
                    gfs_table_xarray.coords[gfs_utils.FIELD_DIM_2D].values ==
                    this_field_name
                )[0][0]
                k_aux = numpy.where(
                    precip_table_xarray.coords[gfs_utils.FIELD_DIM_2D].values ==
                    this_field_name
                )[0][0]

                orig_num_nans = numpy.sum(numpy.isnan(
                    main_data_matrix_2d[..., k_main]
                ))
                main_data_matrix_2d[..., k_main] = (
                    aux_data_matrix_2d[..., k_aux] + 0.
                )
                num_nans = numpy.sum(numpy.isnan(
                    main_data_matrix_2d[..., k_main]
                ))

                print('Replaced {0:d} of {1:d} NaN values for {2:s}!'.format(
                    orig_num_nans - num_nans,
                    orig_num_nans,
                    this_field_name
                ))

            gfs_table_xarray = gfs_table_xarray.assign({
                gfs_utils.DATA_KEY_2D: (
                    gfs_table_xarray[gfs_utils.DATA_KEY_2D].dims,
                    main_data_matrix_2d
                )
            })

        output_file_name = gfs_io.find_file(
            directory_name=output_dir_name,
            init_date_string=this_date_string,
            raise_error_if_missing=False
        )

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        gfs_io.write_file(
            zarr_file_name=output_file_name, gfs_table_xarray=gfs_table_xarray
        )
        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        main_input_dir_name=getattr(INPUT_ARG_OBJECT, MAIN_INPUT_DIR_ARG_NAME),
        input_precip_dir_name=getattr(
            INPUT_ARG_OBJECT, INPUT_PRECIP_DIR_ARG_NAME
        ),
        start_date_string=getattr(INPUT_ARG_OBJECT, START_DATE_ARG_NAME),
        end_date_string=getattr(INPUT_ARG_OBJECT, END_DATE_ARG_NAME),
        start_latitude_deg_n=getattr(INPUT_ARG_OBJECT, START_LATITUDE_ARG_NAME),
        end_latitude_deg_n=getattr(INPUT_ARG_OBJECT, END_LATITUDE_ARG_NAME),
        start_longitude_deg_e=getattr(
            INPUT_ARG_OBJECT, START_LONGITUDE_ARG_NAME
        ),
        end_longitude_deg_e=getattr(INPUT_ARG_OBJECT, END_LONGITUDE_ARG_NAME),
        wgrib2_exe_name=getattr(INPUT_ARG_OBJECT, WGRIB2_EXE_ARG_NAME),
        temporary_dir_name=getattr(INPUT_ARG_OBJECT, TEMPORARY_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
