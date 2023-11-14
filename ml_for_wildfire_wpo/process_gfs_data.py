"""Processes GFS data.

Each raw file should be a GRIB2 file downloaded from the NOAA High-performance
Storage System (HPSS) with the following options:

- One model run (init time)
- One forecast hour (valid time)
- Global domain
- 0.25-deg resolution
- 3-D variables (at 1000, 900, 800, 700, 600, 500, 400, 300, and 200 mb) =
  Temperature, specific humidity, geopotential height, u-wind, v-wind
- Variables at 2 m above ground level (AGL) = temperature, specific humidity,
  dewpoint
- Variables at 10 m AGL: u-wind, v-wind
- Variables at 0 m AGL: pressure
- Other variables: accumulated precip, accumulated convective precip,
  precipitable water, snow depth, water-equivalent snow depth, downward
  shortwave radiative flux, upward shortwave radiative flux, downward
  longwave radiative flux, upward longwave radiative flux, CAPE, soil
  temperature, volumetric soil-moisture fraction, vegetation type, soil type

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
import error_checking
import gfs_io
import raw_gfs_io
import gfs_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FORECAST_HOURS_DEFAULT = numpy.array([
    0, 6, 12, 18, 24, 30, 36, 42, 48,
    60, 72, 84, 96, 108, 120,
    144, 168, 192, 216, 240, 264, 288, 312, 336
], dtype=int)

FORECAST_HOURS_FOR_FWI_CALC = set(gfs_utils.ALL_FORECAST_HOURS.tolist())
FORECAST_HOURS_FOR_FWI_CALC.remove(384)
FORECAST_HOURS_FOR_FWI_CALC = numpy.array(
    list(FORECAST_HOURS_FOR_FWI_CALC), dtype=int
)

FIELD_NAMES_3D_DEFAULT = gfs_utils.ALL_3D_FIELD_NAMES
FIELD_NAMES_2D_DEFAULT = gfs_utils.ALL_2D_FIELD_NAMES

FIELD_NAMES_3D_FOR_FWI_CALC = []
FIELD_NAMES_2D_FOR_FWI_CALC = [
    gfs_utils.TEMPERATURE_2METRE_NAME,
    gfs_utils.DEWPOINT_2METRE_NAME,
    gfs_utils.SPECIFIC_HUMIDITY_2METRE_NAME,
    gfs_utils.U_WIND_10METRE_NAME,
    gfs_utils.V_WIND_10METRE_NAME,
    gfs_utils.SURFACE_PRESSURE_NAME,
    gfs_utils.PRECIP_NAME
]

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
FOR_DIRECT_FWI_CALC_ARG_NAME = 'for_direct_fwi_calc'
ALLOW_N_MISSING_HOURS_ARG_NAME = 'allow_n_missing_forecast_hours'
MAX_FORECAST_HOUR_ARG_NAME = 'max_forecast_hour'
OUTPUT_DIR_ARG_NAME = 'output_zarr_dir_name'

MAIN_INPUT_DIR_HELP_STRING = (
    'Name of main input directory, containing one GRIB2 file per model run '
    '(init time) and forecast hour (lead time).  Files therein will be found '
    'by `raw_gfs_io.find_file`.'
)
INPUT_PRECIP_DIR_HELP_STRING = (
    'Name of input directory for precip only.  As for the main input '
    'directory, should contain one GRIB2 file per model run and forecast hour, '
    'to be found by `raw_gfs_io.find_file`.  Unlike the main input directory, '
    'this directory must contain data for every single available GFS forecast '
    'hour, not just select ones.  If you do *not* want to try reading '
    'incremental (per-time-step) precip from a separate directory in case '
    'precip data are missing from the main directory, then just leave this '
    'argument alone.'
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
FOR_DIRECT_FWI_CALC_HELP_STRING = (
    'Boolean flag.  If True, will process only variables needed for direct FWI '
    'calculation (i.e., to compute GFS forecasts of fire-weather indices).  If '
    'False, will process all variables.'
)
ALLOW_N_MISSING_HOURS_HELP_STRING = (
    '[used only if {0:s} == 1] Will allow this number of missing forecast '
    'hours.'
).format(FOR_DIRECT_FWI_CALC_ARG_NAME)

MAX_FORECAST_HOUR_HELP_STRING = (
    '[used only if {0:s} == 1] Max forecast hour to process.'
).format(FOR_DIRECT_FWI_CALC_ARG_NAME)

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
    '--' + FOR_DIRECT_FWI_CALC_ARG_NAME, type=int, required=False, default=0,
    help=FOR_DIRECT_FWI_CALC_ARG_NAME
)
INPUT_ARG_PARSER.add_argument(
    '--' + ALLOW_N_MISSING_HOURS_ARG_NAME, type=int, required=False, default=0,
    help=ALLOW_N_MISSING_HOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_FORECAST_HOUR_ARG_NAME, type=int, required=True,
    help=MAX_FORECAST_HOUR_HELP_STRING
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

    forecast_hours = FORECAST_HOURS_FOR_FWI_CALC

    input_file_names = [
        raw_gfs_io.find_file(
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
        gfs_tables_xarray[k] = raw_gfs_io.read_file(
            grib2_file_name=input_file_names[k],
            desired_row_indices=desired_row_indices,
            desired_column_indices=desired_column_indices,
            wgrib2_exe_name=wgrib2_exe_name,
            temporary_dir_name=temporary_dir_name,
            field_names_2d=
            [gfs_utils.PRECIP_NAME, gfs_utils.CONVECTIVE_PRECIP_NAME],
            field_names_3d=[],
            read_incremental_precip=True
        )

        incremental_precip_matrix_metres = (
            gfs_tables_xarray[k][gfs_utils.DATA_KEY_2D].values
        )
        if not (
                numpy.all(numpy.isnan(incremental_precip_matrix_metres))
                and forecast_hours[k] > 0
        ):
            continue

        current_gfs_table_xarray = raw_gfs_io.read_file(
            grib2_file_name=input_file_names[k],
            desired_row_indices=desired_row_indices,
            desired_column_indices=desired_column_indices,
            wgrib2_exe_name=wgrib2_exe_name,
            temporary_dir_name=temporary_dir_name,
            field_names_2d=
            [gfs_utils.PRECIP_NAME, gfs_utils.CONVECTIVE_PRECIP_NAME],
            field_names_3d=[],
            read_incremental_precip=False
        )

        previous_gfs_table_xarray = raw_gfs_io.read_file(
            grib2_file_name=input_file_names[k - 1],
            desired_row_indices=desired_row_indices,
            desired_column_indices=desired_column_indices,
            wgrib2_exe_name=wgrib2_exe_name,
            temporary_dir_name=temporary_dir_name,
            field_names_2d=
            [gfs_utils.PRECIP_NAME, gfs_utils.CONVECTIVE_PRECIP_NAME],
            field_names_3d=[],
            read_incremental_precip=False
        )

        incremental_precip_matrix_metres = (
            current_gfs_table_xarray[gfs_utils.DATA_KEY_2D].values -
            previous_gfs_table_xarray[gfs_utils.DATA_KEY_2D].values
        )
        if (
                numpy.all(numpy.isnan(incremental_precip_matrix_metres))
                and forecast_hours[k] > 0
        ):
            return None

        gfs_tables_xarray[k] = current_gfs_table_xarray.assign({
            gfs_utils.DATA_KEY_2D: (
                gfs_tables_xarray[k][gfs_utils.DATA_KEY_2D].dims,
                incremental_precip_matrix_metres
            )
        })

    return gfs_utils.concat_over_forecast_hours(gfs_tables_xarray)


def _run(main_input_dir_name, input_precip_dir_name,
         start_date_string, end_date_string,
         start_latitude_deg_n, end_latitude_deg_n, start_longitude_deg_e,
         end_longitude_deg_e, wgrib2_exe_name, temporary_dir_name,
         for_direct_fwi_calc, allow_n_missing_forecast_hours,
         max_forecast_hour, output_dir_name):
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
    :param for_direct_fwi_calc: Same.
    :param allow_n_missing_forecast_hours: Same.
    :param max_forecast_hour: Same.
    :param output_dir_name: Same.
    """

    if not for_direct_fwi_calc:
        allow_n_missing_forecast_hours = 0
        max_forecast_hour = 1000

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

    if for_direct_fwi_calc:
        forecast_hours = FORECAST_HOURS_FOR_FWI_CALC + 0
        field_names_2d = FIELD_NAMES_2D_FOR_FWI_CALC
        field_names_3d = FIELD_NAMES_3D_FOR_FWI_CALC
    else:
        forecast_hours = FORECAST_HOURS_DEFAULT + 0
        field_names_2d = FIELD_NAMES_2D_DEFAULT
        field_names_3d = FIELD_NAMES_3D_DEFAULT

    error_checking.assert_is_greater(max_forecast_hour, 0)
    forecast_hours = forecast_hours[forecast_hours <= max_forecast_hour]

    num_forecast_hours = len(forecast_hours)
    error_checking.assert_is_geq(allow_n_missing_forecast_hours, 0)
    error_checking.assert_is_less_than(
        allow_n_missing_forecast_hours, num_forecast_hours
    )

    for this_date_string in init_date_strings:
        input_file_names = [
            raw_gfs_io.find_file(
                directory_name=main_input_dir_name,
                init_date_string=this_date_string,
                forecast_hour=h,
                raise_error_if_missing=False
            )
            for h in forecast_hours
        ]

        missing_hour_flags = numpy.array(
            [not os.path.isfile(f) for f in input_file_names], dtype=bool
        )
        num_missing_forecast_hours = numpy.sum(missing_hour_flags)

        if num_missing_forecast_hours > allow_n_missing_forecast_hours:
            error_string = (
                'Cannot find {0:d} forecast hours for init date {1:s}.  Files '
                'expected at:\n{2:s}'
            ).format(
                num_missing_forecast_hours,
                this_date_string,
                str(numpy.array(input_file_names)[missing_hour_flags])
            )

            raise ValueError(error_string)

        missing_hour_indices = numpy.where(missing_hour_flags)[0]
        found_hour_index = numpy.where(numpy.invert(missing_hour_flags))[0][0]

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

        for k in range(num_forecast_hours):
            if os.path.isfile(input_file_names[k]):
                gfs_tables_xarray[k] = raw_gfs_io.read_file(
                    grib2_file_name=input_file_names[k],
                    desired_row_indices=desired_row_indices,
                    desired_column_indices=desired_column_indices,
                    wgrib2_exe_name=wgrib2_exe_name,
                    temporary_dir_name=temporary_dir_name,
                    field_names_2d=field_names_2d,
                    field_names_3d=field_names_3d,
                    read_incremental_precip=False
                )
            else:
                warning_string = (
                    'POTENTIAL ERROR: Cannot find file for forecast hour '
                    '{0:d}.  Expected at: "{1:s}"'
                ).format(
                    forecast_hours[k], input_file_names[k]
                )

                warnings.warn(warning_string)

            print(SEPARATOR_STRING)

        for k in missing_hour_indices:
            k_other = found_hour_index
            gfs_tables_xarray[k] = copy.deepcopy(gfs_tables_xarray[k_other])

            gfs_tables_xarray[k] = gfs_tables_xarray[k].assign_coords({
                gfs_utils.FORECAST_HOUR_DIM: forecast_hours[[k]]
            })

            nan_matrix_2d = numpy.full(
                gfs_tables_xarray[k][gfs_utils.DATA_KEY_2D].values.shape,
                numpy.nan
            )
            nan_matrix_3d = numpy.full(
                gfs_tables_xarray[k][gfs_utils.DATA_KEY_3D].values.shape,
                numpy.nan
            )

            gfs_tables_xarray[k] = gfs_tables_xarray[k].assign({
                gfs_utils.DATA_KEY_2D: (
                    gfs_tables_xarray[k][gfs_utils.DATA_KEY_2D].dims,
                    nan_matrix_2d
                ),
                gfs_utils.DATA_KEY_3D: (
                    gfs_tables_xarray[k][gfs_utils.DATA_KEY_3D].dims,
                    nan_matrix_3d
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
                desired_forecast_hours=forecast_hours
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
                )[0]

                if (
                        this_field_name == gfs_utils.CONVECTIVE_PRECIP_NAME and
                        len(k_main) == 0
                ):
                    continue

                k_main = k_main[0]
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

        gfs_table_xarray = gfs_utils.remove_negative_precip(gfs_table_xarray)

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
        for_direct_fwi_calc=bool(
            getattr(INPUT_ARG_OBJECT, FOR_DIRECT_FWI_CALC_ARG_NAME)
        ),
        allow_n_missing_forecast_hours=getattr(
            INPUT_ARG_OBJECT, ALLOW_N_MISSING_HOURS_ARG_NAME
        ),
        max_forecast_hour=getattr(INPUT_ARG_OBJECT, MAX_FORECAST_HOUR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
