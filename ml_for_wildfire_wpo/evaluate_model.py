"""Evaluates model."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import prediction_io
import regression_evaluation as regression_eval

# TODO(thunderhoser): Add input arg for regression vs. classification.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
INIT_DATE_LIMITS_ARG_NAME = 'init_date_limit_strings'
NUM_BOOTSTRAP_REPS_ARG_NAME = 'num_bootstrap_reps'
NUM_RELIABILITY_BINS_ARG_NAME = 'num_reliability_bins'
RELIA_BIN_LIMITS_ARG_NAME = 'reliability_bin_edge_limits'
RELIA_BIN_LIMITS_PRCTILE_ARG_NAME = 'reliability_bin_edge_limits_percentile'
PER_GRID_CELL_ARG_NAME = 'per_grid_cell'
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
NUM_BOOTSTRAP_REPS_HELP_STRING = 'Number of bootstrap replicates.'
NUM_RELIABILITY_BINS_HELP_STRING = 'Number of bins for reliability curves.'
RELIA_BIN_LIMITS_HELP_STRING = (
    'List of two values: the min and max predicted/target value for '
    'reliability curves.  If you want to specify min/max by percentiles '
    'instead, leave this argument alone.'
)
RELIA_BIN_LIMITS_PRCTILE_HELP_STRING = (
    'See documentation for {0:s}.  These two percentiles must be in the range '
    '0...100.'
).format(RELIA_BIN_LIMITS_ARG_NAME)

PER_GRID_CELL_HELP_STRING = (
    'Boolean flag.  If 1, will compute a separate set of scores at each grid '
    'cell.  If 0, will compute one set of scores for the whole domain.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Evaluation scores will be written here by '
    '`regression_evaluation.write_file`.'
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
    '--' + NUM_BOOTSTRAP_REPS_ARG_NAME, type=int, required=True,
    help=NUM_BOOTSTRAP_REPS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_RELIABILITY_BINS_ARG_NAME, type=int, required=False, default=100,
    help=NUM_RELIABILITY_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RELIA_BIN_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=RELIA_BIN_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RELIA_BIN_LIMITS_PRCTILE_ARG_NAME, type=float, nargs=2,
    required=False, default=[0.5, 99.5],
    help=RELIA_BIN_LIMITS_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PER_GRID_CELL_ARG_NAME, type=int, required=True,
    help=PER_GRID_CELL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_dir_name, init_date_limit_strings,
         num_bootstrap_reps, num_reliability_bins,
         reliability_bin_edge_limits, reliability_bin_edge_limits_percentile,
         per_grid_cell, output_file_name):
    """Evaluates model.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of file.
    :param init_date_limit_strings: Same.
    :param num_bootstrap_reps: Same.
    :param num_reliability_bins: Same.
    :param reliability_bin_edge_limits: Same.
    :param reliability_bin_edge_limits_percentile: Same.
    :param per_grid_cell: Same.
    :param output_file_name: Same.
    """

    min_bin_edge = reliability_bin_edge_limits[0]
    max_bin_edge = reliability_bin_edge_limits[1]
    min_bin_edge_percentile = reliability_bin_edge_limits_percentile[0]
    max_bin_edge_percentile = reliability_bin_edge_limits_percentile[1]

    if min_bin_edge >= max_bin_edge:
        min_bin_edge = None
        max_bin_edge = None

        error_checking.assert_is_leq(min_bin_edge_percentile, 10.)
        error_checking.assert_is_geq(max_bin_edge_percentile, 90.)
    else:
        min_bin_edge_percentile = None
        max_bin_edge_percentile = None

    prediction_file_names = prediction_io.find_files_for_period(
        directory_name=prediction_dir_name,
        first_init_date_string=init_date_limit_strings[0],
        last_init_date_string=init_date_limit_strings[1],
        raise_error_if_any_missing=False, raise_error_if_all_missing=True
    )

    result_table_xarray = regression_eval.get_scores_with_bootstrapping(
        prediction_file_names=prediction_file_names,
        num_bootstrap_reps=num_bootstrap_reps,
        num_reliability_bins=num_reliability_bins,
        min_reliability_bin_edge=min_bin_edge,
        max_reliability_bin_edge=max_bin_edge,
        min_reliability_bin_edge_percentile=min_bin_edge_percentile,
        max_reliability_bin_edge_percentile=max_bin_edge_percentile,
        per_grid_cell=per_grid_cell
    )
    print(SEPARATOR_STRING)

    t = result_table_xarray
    print((
        'Stdev of target and predicted values = {0:f}, {1:f} ... '
        'MSE and skill score = {2:f}, {3:f} ... '
        'DWMSE and skill score = {4:f}, {5:f} ... '
        'MAE and skill score = {6:f}, {7:f} ... '
        'bias = {8:f} ... correlation = {9:f} ... KGE = {10:f}'
    ).format(
        numpy.nanmean(t[regression_eval.TARGET_STDEV_KEY].values),
        numpy.nanmean(t[regression_eval.PREDICTION_STDEV_KEY].values),
        numpy.nanmean(t[regression_eval.MSE_KEY].values),
        numpy.nanmean(t[regression_eval.MSE_SKILL_SCORE_KEY].values),
        numpy.nanmean(t[regression_eval.DWMSE_KEY].values),
        numpy.nanmean(t[regression_eval.DWMSE_SKILL_SCORE_KEY].values),
        numpy.nanmean(t[regression_eval.MAE_KEY].values),
        numpy.nanmean(t[regression_eval.MAE_SKILL_SCORE_KEY].values),
        numpy.nanmean(t[regression_eval.BIAS_KEY].values),
        numpy.nanmean(t[regression_eval.CORRELATION_KEY].values),
        numpy.nanmean(t[regression_eval.KGE_KEY].values)
    ))

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    regression_eval.write_file(
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
        num_bootstrap_reps=getattr(
            INPUT_ARG_OBJECT, NUM_BOOTSTRAP_REPS_ARG_NAME
        ),
        num_reliability_bins=getattr(
            INPUT_ARG_OBJECT, NUM_RELIABILITY_BINS_ARG_NAME
        ),
        reliability_bin_edge_limits=numpy.array(
            getattr(INPUT_ARG_OBJECT, RELIA_BIN_LIMITS_ARG_NAME),
            dtype=float
        ),
        reliability_bin_edge_limits_percentile=numpy.array(
            getattr(INPUT_ARG_OBJECT, RELIA_BIN_LIMITS_PRCTILE_ARG_NAME),
            dtype=float
        ),
        per_grid_cell=bool(getattr(INPUT_ARG_OBJECT, PER_GRID_CELL_ARG_NAME)),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )