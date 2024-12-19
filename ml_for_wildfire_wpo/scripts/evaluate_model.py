"""Evaluates model."""

import argparse
import numpy
from ml_for_wildfire_wpo.io import prediction_io
from ml_for_wildfire_wpo.utils import regression_evaluation as regression_eval

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SENTINEL_VALUE = -1e6

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
INIT_DATE_LIMITS_ARG_NAME = 'init_date_limit_strings'
NUM_BOOTSTRAP_REPS_ARG_NAME = 'num_bootstrap_reps'
TARGET_FIELDS_ARG_NAME = 'target_field_names'
NUM_RELIA_BINS_ARG_NAME = 'num_relia_bins_by_target'
MIN_RELIA_BIN_EDGES_ARG_NAME = 'min_relia_bin_edge_by_target'
MAX_RELIA_BIN_EDGES_ARG_NAME = 'max_relia_bin_edge_by_target'
MIN_RELIA_BIN_EDGES_PRCTILE_ARG_NAME = 'min_relia_bin_edge_prctile_by_target'
MAX_RELIA_BIN_EDGES_PRCTILE_ARG_NAME = 'max_relia_bin_edge_prctile_by_target'
PER_GRID_CELL_ARG_NAME = 'per_grid_cell'
KEEP_IT_SIMPLE_ARG_NAME = 'keep_it_simple'
COMPUTE_SSRAT_ARG_NAME = 'compute_ssrat'
LATITUDE_LIMITS_ARG_NAME = 'latitude_limits_deg_n'
LONGITUDE_LIMITS_ARG_NAME = 'longitude_limits_deg_e'
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
NUM_BOOTSTRAP_REPS_HELP_STRING = 'Number of bootstrap replicates.'
TARGET_FIELDS_HELP_STRING = 'List of target fields to be evaluated.'
NUM_RELIA_BINS_HELP_STRING = (
    'length-T numpy array with number of bins in reliability curve for each '
    'target, where T = number of target fields.'
)
MIN_RELIA_BIN_EDGES_HELP_STRING = (
    'length-T numpy array with minimum target/predicted value in reliability '
    'curve for each target.  If you instead want minimum values to be '
    'percentiles over the data, leave this argument alone and use {0:s}.'
).format(MIN_RELIA_BIN_EDGES_PRCTILE_ARG_NAME)

MAX_RELIA_BIN_EDGES_HELP_STRING = 'Same as {0:s} but for maximum'.format(
    MIN_RELIA_BIN_EDGES_ARG_NAME
)
MIN_RELIA_BIN_EDGES_PRCTILE_HELP_STRING = (
    'length-T numpy array with percentile level used to determine minimum '
    'target/predicted value in reliability curve for each target.  If you '
    'instead want to specify raw values, leave this argument alone and use '
    '{0:s}.'
).format(MIN_RELIA_BIN_EDGES_ARG_NAME)

MAX_RELIA_BIN_EDGES_PRCTILE_HELP_STRING = (
    'Same as {0:s} but for maximum'
).format(MIN_RELIA_BIN_EDGES_PRCTILE_ARG_NAME)

PER_GRID_CELL_HELP_STRING = (
    'Boolean flag.  If 1, will compute a separate set of scores at each grid '
    'cell.  If 0, will compute one set of scores for the whole domain.'
)
KEEP_IT_SIMPLE_HELP_STRING = (
    'Boolean flag.  If 1, will avoid Kolmogorov-Smirnov test and attributes '
    'diagram.'
)
COMPUTE_SSRAT_HELP_STRING = (
    'Boolean flag.  If 1, will compute spread-skill ratio (SSRAT) and spread-'
    'skill difference (SSDIFF).'
)
LATITUDE_LIMITS_HELP_STRING = (
    'Length-2 list with meridional limits (deg north) of bounding box for '
    'evaluation domain.  Will evaluate only at these grid points.  If you do '
    'not want to subset the domain for evaluation, leave this argument alone.'
)
LONGITUDE_LIMITS_HELP_STRING = (
    'Same as {0:s} but for longitudes (deg east).'
).format(
    LATITUDE_LIMITS_ARG_NAME
)
ISOTONIC_MODEL_FILE_HELP_STRING = (
    'Path to file with isotonic-regression model, which will be used to bias-'
    'correct predictions before evaluation.  If you do not want IR, leave this '
    'argument alone.'
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
    '--' + TARGET_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=TARGET_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_RELIA_BINS_ARG_NAME, type=int, nargs='+', required=True,
    help=NUM_RELIA_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_RELIA_BIN_EDGES_ARG_NAME, type=float, nargs='+', required=False,
    default=[SENTINEL_VALUE - 1], help=MIN_RELIA_BIN_EDGES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_RELIA_BIN_EDGES_ARG_NAME, type=float, nargs='+', required=False,
    default=[SENTINEL_VALUE - 1], help=MAX_RELIA_BIN_EDGES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_RELIA_BIN_EDGES_PRCTILE_ARG_NAME, type=float, nargs='+',
    required=False, default=[SENTINEL_VALUE - 1],
    help=MIN_RELIA_BIN_EDGES_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_RELIA_BIN_EDGES_PRCTILE_ARG_NAME, type=float, nargs='+',
    required=False, default=[SENTINEL_VALUE - 1],
    help=MAX_RELIA_BIN_EDGES_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PER_GRID_CELL_ARG_NAME, type=int, required=True,
    help=PER_GRID_CELL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + KEEP_IT_SIMPLE_ARG_NAME, type=int, required=True,
    help=KEEP_IT_SIMPLE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPUTE_SSRAT_ARG_NAME, type=int, required=False, default=0,
    help=COMPUTE_SSRAT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LATITUDE_LIMITS_ARG_NAME, type=float, nargs=2,
    required=False, default=[91, 91], help=LATITUDE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LONGITUDE_LIMITS_ARG_NAME, type=float, nargs=2,
    required=False, default=[361, 361], help=LONGITUDE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ISOTONIC_MODEL_FILE_ARG_NAME, type=str, required=False, default='',
    help=ISOTONIC_MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_dir_name, init_date_limit_strings, num_bootstrap_reps,
         target_field_names, num_relia_bins_by_target,
         min_relia_bin_edge_by_target, max_relia_bin_edge_by_target,
         min_relia_bin_edge_prctile_by_target,
         max_relia_bin_edge_prctile_by_target,
         per_grid_cell, keep_it_simple, compute_ssrat,
         latitude_limits_deg_n, longitude_limits_deg_e,
         isotonic_model_file_name, output_file_name):
    """Evaluates model.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of file.
    :param init_date_limit_strings: Same.
    :param num_bootstrap_reps: Same.
    :param target_field_names: Same.
    :param num_relia_bins_by_target: Same.
    :param min_relia_bin_edge_by_target: Same.
    :param max_relia_bin_edge_by_target: Same.
    :param min_relia_bin_edge_prctile_by_target: Same.
    :param max_relia_bin_edge_prctile_by_target: Same.
    :param per_grid_cell: Same.
    :param keep_it_simple: Same.
    :param compute_ssrat: Same.
    :param latitude_limits_deg_n: Same.
    :param longitude_limits_deg_e: Same.
    :param isotonic_model_file_name: Same.
    :param output_file_name: Same.
    """

    if numpy.all(latitude_limits_deg_n > 90):
        latitude_limits_deg_n = None
    if numpy.all(longitude_limits_deg_e > 360):
        longitude_limits_deg_e = None

    if (
            (len(min_relia_bin_edge_by_target) == 1 and
             min_relia_bin_edge_by_target[0] <= SENTINEL_VALUE) or
            (len(max_relia_bin_edge_by_target) == 1 and
             max_relia_bin_edge_by_target[0] <= SENTINEL_VALUE)
    ):
        min_relia_bin_edge_by_target = None
        max_relia_bin_edge_by_target = None

    if (
            (len(min_relia_bin_edge_prctile_by_target) == 1 and
             min_relia_bin_edge_prctile_by_target[0] <= SENTINEL_VALUE) or
            (len(max_relia_bin_edge_prctile_by_target) == 1 and
             max_relia_bin_edge_prctile_by_target[0] <= SENTINEL_VALUE)
    ):
        min_relia_bin_edge_prctile_by_target = None
        max_relia_bin_edge_prctile_by_target = None

    if isotonic_model_file_name == '':
        isotonic_model_file_name = None

    prediction_file_names = prediction_io.find_files_for_period(
        directory_name=prediction_dir_name,
        first_init_date_string=init_date_limit_strings[0],
        last_init_date_string=init_date_limit_strings[1],
        raise_error_if_any_missing=False, raise_error_if_all_missing=True
    )

    result_table_xarray = regression_eval.get_scores_with_bootstrapping(
        prediction_file_names=prediction_file_names,
        isotonic_model_file_name=isotonic_model_file_name,
        num_bootstrap_reps=num_bootstrap_reps,
        target_field_names=target_field_names,
        num_relia_bins_by_target=num_relia_bins_by_target,
        min_relia_bin_edge_by_target=min_relia_bin_edge_by_target,
        max_relia_bin_edge_by_target=max_relia_bin_edge_by_target,
        min_relia_bin_edge_prctile_by_target=
        min_relia_bin_edge_prctile_by_target,
        max_relia_bin_edge_prctile_by_target=
        max_relia_bin_edge_prctile_by_target,
        per_grid_cell=per_grid_cell,
        keep_it_simple=keep_it_simple,
        compute_ssrat=compute_ssrat,
        latitude_limits_deg_n=latitude_limits_deg_n,
        longitude_limits_deg_e=longitude_limits_deg_e
    )
    print(SEPARATOR_STRING)

    t = result_table_xarray
    target_field_names = t.coords[regression_eval.FIELD_DIM].values

    for k in range(len(target_field_names)):
        print((
            'Stdev of target and predicted {0:s} = {1:f}, {2:f} ... '
            'MSE and skill score = {3:f}, {4:f} ... '
            'DWMSE and skill score = {5:f}, {6:f} ... '
            'MAE and skill score = {7:f}, {8:f} ... '
            'bias = {9:f} ... correlation = {10:f} ... KGE = {11:f}'
        ).format(
            target_field_names[k],
            numpy.nanmean(
                t[regression_eval.TARGET_STDEV_KEY].values[..., k, :]
            ),
            numpy.nanmean(
                t[regression_eval.PREDICTION_STDEV_KEY].values[..., k, :]
            ),
            numpy.nanmean(t[regression_eval.MSE_KEY].values[..., k, :]),
            numpy.nanmean(
                t[regression_eval.MSE_SKILL_SCORE_KEY].values[..., k, :]
            ),
            numpy.nanmean(t[regression_eval.DWMSE_KEY].values[..., k, :]),
            numpy.nanmean(
                t[regression_eval.DWMSE_SKILL_SCORE_KEY].values[..., k, :]
            ),
            numpy.nanmean(t[regression_eval.MAE_KEY].values[..., k, :]),
            numpy.nanmean(
                t[regression_eval.MAE_SKILL_SCORE_KEY].values[..., k, :]
            ),
            numpy.nanmean(t[regression_eval.BIAS_KEY].values[..., k, :]),
            numpy.nanmean(t[regression_eval.CORRELATION_KEY].values[..., k, :]),
            numpy.nanmean(t[regression_eval.KGE_KEY].values[..., k, :])
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
        target_field_names=getattr(
            INPUT_ARG_OBJECT, TARGET_FIELDS_ARG_NAME
        ),
        num_relia_bins_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, NUM_RELIA_BINS_ARG_NAME), dtype=int
        ),
        min_relia_bin_edge_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_RELIA_BIN_EDGES_ARG_NAME), dtype=float
        ),
        max_relia_bin_edge_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_RELIA_BIN_EDGES_ARG_NAME), dtype=float
        ),
        min_relia_bin_edge_prctile_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_RELIA_BIN_EDGES_PRCTILE_ARG_NAME),
            dtype=float
        ),
        max_relia_bin_edge_prctile_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_RELIA_BIN_EDGES_PRCTILE_ARG_NAME),
            dtype=float
        ),
        per_grid_cell=bool(getattr(INPUT_ARG_OBJECT, PER_GRID_CELL_ARG_NAME)),
        keep_it_simple=bool(getattr(INPUT_ARG_OBJECT, KEEP_IT_SIMPLE_ARG_NAME)),
        compute_ssrat=bool(getattr(INPUT_ARG_OBJECT, COMPUTE_SSRAT_ARG_NAME)),
        latitude_limits_deg_n=numpy.array(
            getattr(INPUT_ARG_OBJECT, LATITUDE_LIMITS_ARG_NAME), dtype=float
        ),
        longitude_limits_deg_e=numpy.array(
            getattr(INPUT_ARG_OBJECT, LONGITUDE_LIMITS_ARG_NAME), dtype=float
        ),
        isotonic_model_file_name=getattr(
            INPUT_ARG_OBJECT, ISOTONIC_MODEL_FILE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
