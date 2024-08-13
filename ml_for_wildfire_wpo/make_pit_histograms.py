"""Computes PIT (prob integral transform) histogram for each target field."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io
import pit_histogram_utils as pith_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
INIT_DATE_LIMITS_ARG_NAME = 'init_date_limit_strings'
TARGET_FIELDS_ARG_NAME = 'target_field_names'
NUM_BINS_ARG_NAME = 'num_bins'
ISOTONIC_MODEL_FILE_ARG_NAME = 'isotonic_model_file_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing one prediction file per '
    'initialization date (i.e., per GFS model run).  Files therein will be '
    'found by `prediction_io.find_file` and read by `prediction_io.read_file`.'
)
INIT_DATE_LIMITS_HELP_STRING = (
    'List of two GFS-init dates, specifying the beginning and end of the '
    'evaluation period.  Format of each date should be "yyyymmdd".'
)
TARGET_FIELDS_HELP_STRING = 'List of target fields to be evaluated.'
NUM_BINS_HELP_STRING = 'Number of bins per histogram.'
ISOTONIC_MODEL_FILE_HELP_STRING = (
    'Path to file with isotonic-regression model, which will be used to bias-'
    'correct predictions before evaluation.  If you do not want IR, leave this '
    'argument alone.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Evaluation scores will be written here by '
    '`pit_histogram_utils.write_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_DATE_LIMITS_ARG_NAME, type=str, nargs=2, required=True,
    help=INIT_DATE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=TARGET_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BINS_ARG_NAME, type=int, required=True, help=NUM_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ISOTONIC_MODEL_FILE_ARG_NAME, type=str, required=False, default='',
    help=ISOTONIC_MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_dir_name, init_date_limit_strings, target_field_names,
         num_bins, isotonic_model_file_name, output_file_name):
    """Computes PIT (prob integral transform) histogram for each target field.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of this script.
    :param init_date_limit_strings: Same.
    :param target_field_names: Same.
    :param num_bins: Same.
    :param isotonic_model_file_name: Same.
    :param output_file_name: Same.
    """

    if isotonic_model_file_name == '':
        isotonic_model_file_name = None

    prediction_file_names = prediction_io.find_files_for_period(
        directory_name=prediction_dir_name,
        first_init_date_string=init_date_limit_strings[0],
        last_init_date_string=init_date_limit_strings[1],
        raise_error_if_any_missing=False, raise_error_if_all_missing=True
    )

    result_table_xarray = pith_utils.compute_pit_histograms(
        prediction_file_names=prediction_file_names,
        isotonic_model_file_name=isotonic_model_file_name,
        target_field_names=target_field_names,
        num_bins=num_bins
    )
    print(SEPARATOR_STRING)

    rtx = result_table_xarray
    target_field_names = rtx.coords[pith_utils.FIELD_DIM].values.tolist()

    for j in range(len(target_field_names)):
        print((
            'Variable = {0:s} ... PITD = {1:f} ... low-PIT freq bias = {2:f} '
            '... medium-PIT freq bias = {3:f} ... high-PIT freq bias = {4:f}'
        ).format(
            target_field_names[j],
            rtx[pith_utils.PIT_DEVIATION_KEY].values[j],
            rtx[pith_utils.LOW_BIN_BIAS_KEY].values[j],
            rtx[pith_utils.MIDDLE_BIN_BIAS_KEY].values[j],
            rtx[pith_utils.HIGH_BIN_BIAS_KEY].values[j]
        ))

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    pith_utils.write_results(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        init_date_limit_strings=getattr(
            INPUT_ARG_OBJECT, INIT_DATE_LIMITS_ARG_NAME
        ),
        target_field_names=getattr(
            INPUT_ARG_OBJECT, TARGET_FIELDS_ARG_NAME
        ),
        num_bins=getattr(INPUT_ARG_OBJECT, NUM_BINS_ARG_NAME),
        isotonic_model_file_name=getattr(
            INPUT_ARG_OBJECT, ISOTONIC_MODEL_FILE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
