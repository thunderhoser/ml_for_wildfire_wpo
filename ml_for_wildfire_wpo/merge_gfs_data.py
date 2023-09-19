"""Merges processed GFS files from two sources: NOAA HPSS and NCAR RDA.

NOAA HPSS data are processed by process_gfs_data.py and contain almost
everything needed.  NCAR RDA data are processed by process_ncar_gfs_data.py and
contain only 4 variables: accumulated precip, accumulated convective precip,
vegetation fraction, and soil type.
"""

import os
import sys
import warnings
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import gfs_io
import raw_ncar_gfs_io
import gfs_utils

TOLERANCE = 1e-6
FIELD_NAMES_TO_MERGE = raw_ncar_gfs_io.FIELD_NAMES_TO_READ

MAIN_GFS_DIR_ARG_NAME = 'input_main_gfs_dir_name'
NCAR_GFS_DIR_ARG_NAME = 'input_ncar_gfs_dir_name'
START_DATE_ARG_NAME = 'start_date_string'
END_DATE_ARG_NAME = 'end_date_string'
ALLOW_MISSING_NCAR_ARG_NAME = 'allow_missing_ncar_files'
MERGED_GFS_DIR_ARG_NAME = 'output_merged_gfs_dir_name'

MAIN_GFS_DIR_HELP_STRING = (
    'Name of directory with main (from NOAA HPSS) GFS dataset, containing most '
    'of the variables.  Files therein will be found by `gfs_io.find_file` and '
    'read by `gfs_io.read_file`.'
)
NCAR_GFS_DIR_HELP_STRING = (
    'Name of directory with NCAR GFS dataset, containing a few of the '
    'variables.  Files therein will be found by `gfs_io.find_file` and read by '
    '`gfs_io.read_file`.'
)
START_DATE_HELP_STRING = (
    'Start date (format "yyyymmdd").  This script will process model runs '
    'initialized at all dates in the continuous period {0:s}...{1:s}.'
).format(START_DATE_ARG_NAME, END_DATE_ARG_NAME)

END_DATE_HELP_STRING = 'Same as {0:s} but end date.'.format(START_DATE_ARG_NAME)
ALLOW_MISSING_NCAR_HELP_STRING = (
    'Boolean flag.  If 1, when an NCAR file is missing, this script will just '
    'copy the main GFS file into the "merged" directory.'
)
MERGED_GFS_DIR_HELP_STRING = (
    'Name of output directory.  Merged files will be written here (one '
    'zarr file per model run) by `gfs_io.write_file`, to exact locations '
    'determined by `gfs_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MAIN_GFS_DIR_ARG_NAME, type=str, required=True,
    help=MAIN_GFS_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NCAR_GFS_DIR_ARG_NAME, type=str, required=True,
    help=NCAR_GFS_DIR_HELP_STRING
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
    '--' + ALLOW_MISSING_NCAR_ARG_NAME, type=int, required=False, default=0,
    help=ALLOW_MISSING_NCAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MERGED_GFS_DIR_ARG_NAME, type=str, required=True,
    help=MERGED_GFS_DIR_HELP_STRING
)


def _merge_data_1field_1init(
        main_gfs_table_xarray, ncar_gfs_table_xarray, field_name):
    """Merges data for one field and one init time (model run).

    F = number of forecast hours
    M = number of rows in grid
    N = number of columns in grid

    :param main_gfs_table_xarray: xarray table with main GFS data.
    :param ncar_gfs_table_xarray: xarray table with NCAR GFS data.
    :param field_name: Field name.
    :return: merged_data_matrix: F-by-M-by-N numpy array of data values.
    :return: main_field_index: Index of the given field in the main GFS table.
    """

    ncar_field_index = numpy.where(
        ncar_gfs_table_xarray.coords[gfs_utils.FIELD_DIM_2D].values ==
        field_name
    )[0][0]
    ncar_data_matrix = ncar_gfs_table_xarray[gfs_utils.DATA_KEY_2D].values[
        ..., ncar_field_index
    ]

    main_field_index = numpy.where(
        main_gfs_table_xarray.coords[gfs_utils.FIELD_DIM_2D].values ==
        field_name
    )[0][0]
    main_data_matrix = main_gfs_table_xarray[gfs_utils.DATA_KEY_2D].values[
        ..., main_field_index
    ]
    orig_num_missing_values = numpy.sum(numpy.isnan(main_data_matrix))

    nan_flag_by_forecast_hour = numpy.all(
        numpy.isnan(main_data_matrix), axis=(1, 2)
    )
    nan_indices = numpy.where(nan_flag_by_forecast_hour)[0]
    if len(nan_indices) == 0:
        print((
            'Main GFS table has no forecast hour with all NaN for {0:s}.'
        ).format(
            field_name
        ))
        return main_data_matrix, main_field_index

    main_data_matrix[nan_indices, ...] = ncar_data_matrix[nan_indices, ...]
    num_missing_values = numpy.sum(numpy.isnan(main_data_matrix))

    print('Replaced {0:d} of {1:d} missing values for {2:s}.'.format(
        orig_num_missing_values - num_missing_values,
        orig_num_missing_values,
        field_name
    ))

    return main_data_matrix, main_field_index


def _run(main_gfs_dir_name, ncar_gfs_dir_name, start_date_string,
         end_date_string, allow_missing_ncar_files, merged_gfs_dir_name):
    """Merges processed GFS files from two sources: NOAA HPSS and NCAR RDA.

    This is effectively the main method.

    :param main_gfs_dir_name: See documentation at top of file.
    :param ncar_gfs_dir_name: Same.
    :param start_date_string: Same.
    :param end_date_string: Same.
    :param allow_missing_ncar_files: Same.
    :param merged_gfs_dir_name: Same.
    """

    init_date_strings = time_conversion.get_spc_dates_in_range(
        start_date_string, end_date_string
    )
    num_fields_to_merge = len(FIELD_NAMES_TO_MERGE)

    for this_date_string in init_date_strings:
        main_gfs_file_name = gfs_io.find_file(
            directory_name=main_gfs_dir_name,
            init_date_string=this_date_string,
            raise_error_if_missing=True
        )
        ncar_gfs_file_name = gfs_io.find_file(
            directory_name=ncar_gfs_dir_name,
            init_date_string=this_date_string,
            raise_error_if_missing=not allow_missing_ncar_files
        )

        print('Reading data from: "{0:s}"...'.format(main_gfs_file_name))
        main_gfs_table_xarray = gfs_io.read_file(main_gfs_file_name)

        data_matrix_2d = main_gfs_table_xarray[gfs_utils.DATA_KEY_2D].values
        field_names_2d = (
            main_gfs_table_xarray.coords[gfs_utils.FIELD_DIM_2D].values
        )

        if os.path.isfile(ncar_gfs_file_name):
            print('Reading data from: "{0:s}"...'.format(ncar_gfs_file_name))
            ncar_gfs_table_xarray = gfs_io.read_file(ncar_gfs_file_name)

            assert numpy.array_equal(
                main_gfs_table_xarray.coords[gfs_utils.FORECAST_HOUR_DIM].values,
                ncar_gfs_table_xarray.coords[gfs_utils.FORECAST_HOUR_DIM].values
            )
            assert numpy.allclose(
                main_gfs_table_xarray.coords[gfs_utils.LATITUDE_DIM].values,
                ncar_gfs_table_xarray.coords[gfs_utils.LATITUDE_DIM].values,
                atol=TOLERANCE
            )
            assert numpy.allclose(
                main_gfs_table_xarray.coords[gfs_utils.LONGITUDE_DIM].values,
                ncar_gfs_table_xarray.coords[gfs_utils.LONGITUDE_DIM].values,
                atol=TOLERANCE
            )

            for k in range(num_fields_to_merge):
                this_merged_data_matrix, this_index = _merge_data_1field_1init(
                    main_gfs_table_xarray=main_gfs_table_xarray,
                    ncar_gfs_table_xarray=ncar_gfs_table_xarray,
                    field_name=FIELD_NAMES_TO_MERGE[k]
                )
                data_matrix_2d[..., this_index] = this_merged_data_matrix
        else:
            warning_string = (
                'POTENTIAL ERROR: Could not find NCAR GFS file.  Expected at: '
                '"{0:s}"'
            ).format(ncar_gfs_file_name)

            warnings.warn(warning_string)

        data_dict = {}
        for var_name in main_gfs_table_xarray.data_vars:
            if var_name == gfs_utils.DATA_KEY_2D:
                data_dict[var_name] = (
                    main_gfs_table_xarray[var_name].dims,
                    data_matrix_2d
                )
            else:
                data_dict[var_name] = (
                    main_gfs_table_xarray[var_name].dims,
                    main_gfs_table_xarray[var_name].values
                )

        coord_dict = {}
        for coord_name in main_gfs_table_xarray.coords:
            if coord_name == gfs_utils.FIELD_DIM_2D:
                coord_dict[coord_name] = field_names_2d
            else:
                coord_dict[coord_name] = (
                    main_gfs_table_xarray.coords[coord_name].values
                )

        merged_gfs_table_xarray = xarray.Dataset(
            data_vars=data_dict, coords=coord_dict
        )
        merged_gfs_file_name = gfs_io.find_file(
            directory_name=merged_gfs_dir_name,
            init_date_string=this_date_string,
            raise_error_if_missing=False
        )

        print('Writing merged data to: "{0:s}"...'.format(merged_gfs_file_name))
        gfs_io.write_file(
            gfs_table_xarray=merged_gfs_table_xarray,
            zarr_file_name=merged_gfs_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        main_gfs_dir_name=getattr(INPUT_ARG_OBJECT, MAIN_GFS_DIR_ARG_NAME),
        ncar_gfs_dir_name=getattr(INPUT_ARG_OBJECT, NCAR_GFS_DIR_ARG_NAME),
        start_date_string=getattr(INPUT_ARG_OBJECT, START_DATE_ARG_NAME),
        end_date_string=getattr(INPUT_ARG_OBJECT, END_DATE_ARG_NAME),
        allow_missing_ncar_files=bool(
            getattr(INPUT_ARG_OBJECT, ALLOW_MISSING_NCAR_ARG_NAME)
        ),
        merged_gfs_dir_name=getattr(INPUT_ARG_OBJECT, MERGED_GFS_DIR_ARG_NAME)
    )
