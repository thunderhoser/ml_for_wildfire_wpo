"""Determines weights for loss function.

There is one weight per target variable; the goal is to make every target
variable have an equal contribution to the overall loss.  This script assumes
that the loss function is dual-weighted mean squared error (DWMSE).
"""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import error_checking
import canadian_fwi_io
import canadian_fwi_utils

# TODO(thunderhoser): Adding the land-sea mask as input could make the weights
# a bit better.

INPUT_DIR_ARG_NAME = 'input_target_dir_name'
TARGET_FIELDS_ARG_NAME = 'target_field_names'
FIRST_DATES_ARG_NAME = 'first_valid_date_strings'
LAST_DATES_ARG_NAME = 'last_valid_date_strings'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
QUANTILE_LEVEL_ARG_NAME = 'quantile_level'
NUM_SAMPLE_VALUES_ARG_NAME = 'num_sample_values_per_file'

INPUT_DIR_HELP_STRING = (
    'Path to directory with target fields.  Files therein will be found by '
    '`canadian_fwi_io.find_file` and read by `canadian_fwi_io.read_file`.'
)
TARGET_FIELDS_HELP_STRING = '1-D list with names of target fields to predict.'
FIRST_DATES_HELP_STRING = (
    'List with first valid date (format "yyyymmdd") for each continuous '
    'period.  Weights will be based on all valid dates in all periods.'
)
LAST_DATES_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_DATES_ARG_NAME
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file.  Quantiles will be read from here, to '
    'determine which target values will be ignored because they are too high.'
)
QUANTILE_LEVEL_HELP_STRING = (
    'Threshold quantile level.  Values above this level, for each target '
    'variable, will be considered too extreme and thus ignored.'
)
NUM_SAMPLE_VALUES_HELP_STRING = (
    'Number of sample values per file to use for computing weights.  This '
    'value will be applied to each variable.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=TARGET_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATES_ARG_NAME, type=str, nargs='+', required=True,
    help=FIRST_DATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATES_ARG_NAME, type=str, nargs='+', required=True,
    help=LAST_DATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + QUANTILE_LEVEL_ARG_NAME, type=float, required=True,
    help=QUANTILE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_SAMPLE_VALUES_ARG_NAME, type=int, required=True,
    help=NUM_SAMPLE_VALUES_HELP_STRING
)


def _increment_dwmse_one_field(fwi_table_xarray, field_name, extreme_threshold,
                               climo_mean, num_sample_values):
    """Increments DWMSE (dual-weighted mean squared error) for one field.

    :param fwi_table_xarray: xarray table in format returned by
        `canadian_fwi_io.read_file`.
    :param field_name: Name of field.
    :param extreme_threshold: Extreme-value threshold for given field.
    :param climo_mean: Climatological mean for given field.
    :param num_sample_values: Number of sample values to use from
        `fwi_table_xarray`.
    :return: new_dwmse: DWMSE for new data batch.
    :return: new_num_values: Number of values in new data batch.
    """

    data_matrix = canadian_fwi_utils.get_field(
        fwi_table_xarray=fwi_table_xarray, field_name=field_name
    )
    data_matrix = numpy.minimum(data_matrix, extreme_threshold)

    real_data_values = data_matrix[numpy.invert(numpy.isnan(data_matrix))]
    numpy.random.shuffle(real_data_values)
    real_data_values = real_data_values[:num_sample_values]

    sample_weights = numpy.maximum(
        numpy.absolute(climo_mean),
        numpy.absolute(real_data_values)
    )
    error_values = sample_weights * (climo_mean - real_data_values) ** 2

    return numpy.mean(error_values), len(error_values)


def _run(input_dir_name, target_field_names,
         first_date_strings, last_date_strings,
         normalization_file_name, quantile_level, num_sample_values_per_file):
    """Determines weights for loss function.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param target_field_names: Same.
    :param first_date_strings: Same.
    :param last_date_strings: Same.
    :param normalization_file_name: Same.
    :param quantile_level: Same.
    :param num_sample_values_per_file: Same.
    """

    # Check input args.
    num_periods = len(first_date_strings)
    assert len(last_date_strings) == num_periods

    error_checking.assert_is_greater(quantile_level, 0.)
    error_checking.assert_is_leq(quantile_level, 1.)

    # Determine extreme-value thresholds.
    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    norm_param_table_xarray = canadian_fwi_io.read_normalization_file(
        normalization_file_name
    )
    npt = norm_param_table_xarray

    q = numpy.argmin(numpy.absolute(
        npt.coords[canadian_fwi_utils.QUANTILE_LEVEL_DIM].values -
        quantile_level
    ))

    num_fields = len(target_field_names)
    extreme_threshold_by_field = numpy.full(num_fields, numpy.nan)
    climo_mean_by_field = numpy.full(num_fields, numpy.nan)

    for j in range(num_fields):
        j_new = numpy.where(
            npt.coords[canadian_fwi_utils.FIELD_DIM].values ==
            target_field_names[j]
        )[0][0]

        extreme_threshold_by_field[j] = (
            npt[canadian_fwi_utils.QUANTILE_KEY].values[j_new, q]
        )
        climo_mean_by_field[j] = (
            npt[canadian_fwi_utils.MEAN_VALUE_KEY].values[j_new]
        )

    # Find the weights.
    valid_date_strings = []
    for i in range(num_periods):
        valid_date_strings += time_conversion.get_spc_dates_in_range(
            first_date_strings[i], last_date_strings[i]
        )

    target_file_names = [
        canadian_fwi_io.find_file(
            directory_name=input_dir_name, valid_date_string=d,
            raise_error_if_missing=True
        ) for d in valid_date_strings
    ]

    num_files = len(target_file_names)
    dwmse_by_field = numpy.full(num_fields, numpy.nan)
    num_values_by_field = numpy.full(num_fields, 0, dtype=int)

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(target_file_names[i]))
        fwi_table_xarray = canadian_fwi_io.read_file(target_file_names[i])

        for j in range(num_fields):
            new_dwmse, new_num_values = _increment_dwmse_one_field(
                fwi_table_xarray=fwi_table_xarray,
                field_name=target_field_names[j],
                extreme_threshold=extreme_threshold_by_field[j],
                climo_mean=climo_mean_by_field[j],
                num_sample_values=num_sample_values_per_file
            )

            these_means = numpy.array([dwmse_by_field[j], new_dwmse])
            these_weights = numpy.array([
                num_values_by_field[j], new_num_values
            ])
            dwmse_by_field[j] = numpy.average(
                these_means, weights=these_weights
            )

            num_values_by_field[j] += new_num_values

    unnorm_weight_by_field = numpy.sum(dwmse_by_field) / dwmse_by_field
    weight_by_field = unnorm_weight_by_field / numpy.sum(unnorm_weight_by_field)

    for j in range(num_fields):
        print((
            '{0:s} ... DWMSE = {1:.2f} ... climo mean = {2:.4f} ... '
            'extreme-value threshold = {3:.4f} ... '
            'unnormalized weight = {4:.4f} ... normalized weight = {5:.4f}'
        ).format(
            target_field_names[j],
            dwmse_by_field[j],
            climo_mean_by_field[j],
            extreme_threshold_by_field[j],
            unnorm_weight_by_field[j],
            weight_by_field[j]
        ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        target_field_names=getattr(INPUT_ARG_OBJECT, TARGET_FIELDS_ARG_NAME),
        first_date_strings=getattr(INPUT_ARG_OBJECT, FIRST_DATES_ARG_NAME),
        last_date_strings=getattr(INPUT_ARG_OBJECT, LAST_DATES_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        quantile_level=getattr(INPUT_ARG_OBJECT, QUANTILE_LEVEL_ARG_NAME),
        num_sample_values_per_file=getattr(
            INPUT_ARG_OBJECT, NUM_SAMPLE_VALUES_ARG_NAME
        )
    )
