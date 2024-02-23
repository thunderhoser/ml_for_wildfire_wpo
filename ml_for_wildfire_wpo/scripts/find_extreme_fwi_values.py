"""Finds extreme FWI values."""

import os
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.io import canadian_fwi_io
from ml_for_wildfire_wpo.utils import canadian_fwi_utils

INPUT_DIR_ARG_NAME = 'input_fwi_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
FIRST_DATE_ARG_NAME = 'first_valid_date_string'
LAST_DATE_ARG_NAME = 'last_valid_date_string'
QUANTILE_LEVEL_ARG_NAME = 'quantile_level'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing Canadian FWI data.  Files therein '
    'will be found by `canadian_fwi_io.find_file` and read by '
    '`canadian_fwi_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file, containing (among other things) various '
    'quantiles for each FWI.  Will be read by '
    '`canadian_fwi_io.read_normalization_file`.'
)
FIRST_DATE_HELP_STRING = (
    'First valid date in period (format "yyyymmdd").  This script will look '
    'for extreme values in the period {0:s}...{1:s}.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

LAST_DATE_HELP_STRING = 'Same as {0:s} but last date in period.'.format(
    FIRST_DATE_ARG_NAME
)
QUANTILE_LEVEL_HELP_STRING = (
    'Threshold quantile level.  Values above this level will be considered '
    'extreme.'
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
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=FIRST_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=LAST_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + QUANTILE_LEVEL_ARG_NAME, type=float, required=True,
    help=QUANTILE_LEVEL_HELP_STRING
)


def _run(fwi_dir_name, normalization_file_name, first_valid_date_string,
         last_valid_date_string, quantile_level):
    """Finds extreme FWI values.

    This is effectively the main method.

    :param fwi_dir_name: See documentation at top of this script.
    :param normalization_file_name: Same.
    :param first_valid_date_string: Same.
    :param last_valid_date_string: Same.
    :param quantile_level: Same.
    """

    error_checking.assert_is_greater(quantile_level, 0.)
    error_checking.assert_is_leq(quantile_level, 1.)

    valid_date_strings = time_conversion.get_spc_dates_in_range(
        first_valid_date_string, last_valid_date_string
    )
    fwi_file_names = [
        canadian_fwi_io.find_file(
            directory_name=fwi_dir_name,
            valid_date_string=d,
            raise_error_if_missing=False
        )
        for d in valid_date_strings
    ]
    fwi_file_names = [f for f in fwi_file_names if os.path.isfile(f)]

    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    norm_param_table_xarray = canadian_fwi_io.read_normalization_file(
        normalization_file_name
    )
    npt = norm_param_table_xarray

    q = numpy.argmin(numpy.absolute(
        npt.coords[canadian_fwi_utils.QUANTILE_LEVEL_DIM].values -
        quantile_level
    ))
    extreme_threshold_by_field = (
        npt[canadian_fwi_utils.QUANTILE_KEY].values[:, q]
    )

    field_names = npt.coords[canadian_fwi_utils.FIELD_DIM].values
    num_fields = len(field_names)

    for j in range(num_fields):
        print((
            '{0:.3f}th percentile (extreme-value threshold) for {1:s} = {2:.4g}'
        ).format(
            100 * quantile_level,
            field_names[j],
            extreme_threshold_by_field[j]
        ))

    num_files = len(fwi_file_names)
    extreme_times_by_field_unix_sec = [numpy.array([], dtype=int)] * num_fields
    extreme_latitudes_by_field_deg_n = [numpy.array([])] * num_fields
    extreme_longitudes_by_field_deg_e = [numpy.array([])] * num_fields

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(fwi_file_names[i]))
        fwi_table_xarray = canadian_fwi_io.read_file(fwi_file_names[i])

        current_date_string = canadian_fwi_io.file_name_to_date(
            fwi_file_names[i]
        )
        current_date_unix_sec = time_conversion.string_to_unix_sec(
            current_date_string, canadian_fwi_io.DATE_FORMAT
        )

        for j in range(num_fields):
            this_data_matrix = canadian_fwi_utils.get_field(
                fwi_table_xarray=fwi_table_xarray, field_name=field_names[j]
            )
            row_indices, column_indices = numpy.where(
                this_data_matrix >= extreme_threshold_by_field[j]
            )

            if len(row_indices) == 0:
                continue

            fwit = fwi_table_xarray

            extreme_times_by_field_unix_sec[j] = numpy.concatenate([
                extreme_times_by_field_unix_sec[j],
                numpy.full(len(row_indices), current_date_unix_sec, dtype=int)
            ])
            extreme_latitudes_by_field_deg_n[j] = numpy.concatenate([
                extreme_latitudes_by_field_deg_n[j],
                fwit.coords[canadian_fwi_utils.LATITUDE_DIM].values[row_indices]
            ])
            extreme_longitudes_by_field_deg_e[j] = numpy.concatenate([
                extreme_longitudes_by_field_deg_e[j],
                fwit.coords[canadian_fwi_utils.LONGITUDE_DIM].values[
                    column_indices
                ]
            ])

    for j in range(num_fields):
        unique_times_unix_sec = numpy.unique(extreme_times_by_field_unix_sec[j])
        unique_time_strings = [
            time_conversion.unix_sec_to_string(t, '%Y%m%d')
            for t in unique_times_unix_sec
        ]
        unique_times_array_string = ','.join([
            '{0:s}'.format(t) for t in unique_time_strings
        ])
        print('Extreme-value times for {0:s}: {1:s}'.format(
            field_names[j], unique_times_array_string
        ))

        for this_time_unix_sec in unique_times_unix_sec:
            indices = numpy.where(
                extreme_times_by_field_unix_sec[j] == this_time_unix_sec
            )[0]
            these_latitudes_deg_n = extreme_latitudes_by_field_deg_n[j][indices]
            these_longitudes_deg_e = (
                extreme_longitudes_by_field_deg_e[j][indices]
            )

            coords_array_string = '; '.join([
                '{0:.2f} deg N, {1:.2f} deg E'.format(n, e)
                for n, e in zip(these_latitudes_deg_n, these_longitudes_deg_e)
            ])

            print('Extreme-value coords for {0:s} at {1:s}: {2:s}'.format(
                field_names[j],
                time_conversion.unix_sec_to_string(
                    this_time_unix_sec, '%Y%m%d'
                ),
                coords_array_string
            ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        fwi_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        first_valid_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_valid_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        quantile_level=getattr(INPUT_ARG_OBJECT, QUANTILE_LEVEL_ARG_NAME)
    )
