"""Computes spread-skill relationship for multiple target fields."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io
import spread_skill_utils as ss_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SENTINEL_VALUE = -1e6

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
INIT_DATE_LIMITS_ARG_NAME = 'init_date_limit_strings'
TARGET_FIELDS_ARG_NAME = 'target_field_names'
NUM_BINS_ARG_NAME = 'num_bins_by_target'
MIN_BIN_EDGES_ARG_NAME = 'min_bin_edge_by_target'
MAX_BIN_EDGES_ARG_NAME = 'max_bin_edge_by_target'
MIN_BIN_EDGES_PRCTILE_ARG_NAME = 'min_bin_edge_prctile_by_target'
MAX_BIN_EDGES_PRCTILE_ARG_NAME = 'max_bin_edge_prctile_by_target'
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
NUM_BINS_HELP_STRING = (
    'length-T list with number of spread bins for each target, where '
    'T = number of target fields (length of {0:s}).'
).format(
    TARGET_FIELDS_ARG_NAME
)
MIN_BIN_EDGES_HELP_STRING = (
    'length-T list with minimum spread values in spread-skill plot.  If you '
    'instead want minimum values to be percentiles over the data, leave this '
    'argument alone and use {0:s}.'
).format(
    MIN_BIN_EDGES_PRCTILE_ARG_NAME
)
MAX_BIN_EDGES_HELP_STRING = 'Same as {0:s} but for maximum'.format(
    MIN_BIN_EDGES_ARG_NAME
)
MIN_BIN_EDGES_PRCTILE_HELP_STRING = (
    'length-T list with percentile level used to determine minimum spread '
    'value in plot for each target.  If you instead want to specify raw '
    'values, leave this argument alone and use {0:s}.'
).format(
    MIN_BIN_EDGES_ARG_NAME
)
MAX_BIN_EDGES_PRCTILE_HELP_STRING = 'Same as {0:s} but for maximum'.format(
    MIN_BIN_EDGES_PRCTILE_ARG_NAME
)
ISOTONIC_MODEL_FILE_HELP_STRING = (
    'Path to file with isotonic-regression model, which will be used to bias-'
    'correct predictions before evaluation.  If you do not want IR, leave this '
    'argument alone.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Evaluation scores will be written here by '
    '`spread_skill_utils.write_results`.'
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
    '--' + NUM_BINS_ARG_NAME, type=int, nargs='+', required=True,
    help=NUM_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_BIN_EDGES_ARG_NAME, type=float, nargs='+', required=False,
    default=[SENTINEL_VALUE - 1], help=MIN_BIN_EDGES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_BIN_EDGES_ARG_NAME, type=float, nargs='+', required=False,
    default=[SENTINEL_VALUE - 1], help=MAX_BIN_EDGES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_BIN_EDGES_PRCTILE_ARG_NAME, type=float, nargs='+',
    required=False, default=[SENTINEL_VALUE - 1],
    help=MIN_BIN_EDGES_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_BIN_EDGES_PRCTILE_ARG_NAME, type=float, nargs='+',
    required=False, default=[SENTINEL_VALUE - 1],
    help=MAX_BIN_EDGES_PRCTILE_HELP_STRING
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
         num_bins_by_target, min_bin_edge_by_target, max_bin_edge_by_target,
         min_bin_edge_prctile_by_target, max_bin_edge_prctile_by_target,
         isotonic_model_file_name, output_file_name):
    """Computes spread-skill relationship for multiple target fields.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of file.
    :param init_date_limit_strings: Same.
    :param target_field_names: Same.
    :param num_bins_by_target: Same.
    :param min_bin_edge_by_target: Same.
    :param max_bin_edge_by_target: Same.
    :param min_bin_edge_prctile_by_target: Same.
    :param max_bin_edge_prctile_by_target: Same.
    :param isotonic_model_file_name: Same.
    :param output_file_name: Same.
    """

    if isotonic_model_file_name == '':
        isotonic_model_file_name = None

    if (
            (len(min_bin_edge_by_target) == 1 and
             min_bin_edge_by_target[0] <= SENTINEL_VALUE) or
            (len(max_bin_edge_by_target) == 1 and
             max_bin_edge_by_target[0] <= SENTINEL_VALUE)
    ):
        min_bin_edge_by_target = None
        max_bin_edge_by_target = None

    if (
            (len(min_bin_edge_prctile_by_target) == 1 and
             min_bin_edge_prctile_by_target[0] <= SENTINEL_VALUE) or
            (len(max_bin_edge_prctile_by_target) == 1 and
             max_bin_edge_prctile_by_target[0] <= SENTINEL_VALUE)
    ):
        min_bin_edge_prctile_by_target = None
        max_bin_edge_prctile_by_target = None

    prediction_file_names = prediction_io.find_files_for_period(
        directory_name=prediction_dir_name,
        first_init_date_string=init_date_limit_strings[0],
        last_init_date_string=init_date_limit_strings[1],
        raise_error_if_any_missing=False, raise_error_if_all_missing=True
    )

    result_table_xarray = ss_utils.get_spread_vs_skill(
        prediction_file_names=prediction_file_names,
        isotonic_model_file_name=isotonic_model_file_name,
        target_field_names=target_field_names,
        num_bins_by_target=num_bins_by_target,
        min_bin_edge_by_target=min_bin_edge_by_target,
        max_bin_edge_by_target=max_bin_edge_by_target,
        min_bin_edge_prctile_by_target=min_bin_edge_prctile_by_target,
        max_bin_edge_prctile_by_target=max_bin_edge_prctile_by_target
    )
    print(SEPARATOR_STRING)

    rtx = result_table_xarray
    target_field_names = rtx.coords[ss_utils.FIELD_DIM].values.tolist()

    for j in range(len(target_field_names)):
        print('Variable = {0:s} ... SSREL = {1:f} ... SSRAT = {2:f}'.format(
            target_field_names[j],
            rtx[ss_utils.SSREL_KEY].values[j],
            rtx[ss_utils.SSRAT_KEY].values[j]
        ))

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    ss_utils.write_results(
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
        num_bins_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, NUM_BINS_ARG_NAME), dtype=int
        ),
        min_bin_edge_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_BIN_EDGES_ARG_NAME), dtype=float
        ),
        max_bin_edge_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_BIN_EDGES_ARG_NAME), dtype=float
        ),
        min_bin_edge_prctile_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_BIN_EDGES_PRCTILE_ARG_NAME),
            dtype=float
        ),
        max_bin_edge_prctile_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_BIN_EDGES_PRCTILE_ARG_NAME),
            dtype=float
        ),
        isotonic_model_file_name=getattr(
            INPUT_ARG_OBJECT, ISOTONIC_MODEL_FILE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
