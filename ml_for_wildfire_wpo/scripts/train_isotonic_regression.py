"""Trains isotonic-regression model(s) to bias-correct ensemble mean."""

import argparse
import numpy
import xarray
from ml_for_wildfire_wpo.io import prediction_io
from ml_for_wildfire_wpo.utils import bias_clustering
from ml_for_wildfire_wpo.machine_learning import bias_correction

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIRS_ARG_NAME = 'input_prediction_dir_names'
INIT_DATE_LIMITS_ARG_NAME = 'init_date_limit_strings'
BIAS_CLUSTERING_FILE_ARG_NAME = 'input_bias_clustering_file_name'
PIXEL_RADIUS_ARG_NAME = 'pixel_radius_metres'
WEIGHT_BY_INV_DIST_ARG_NAME = 'weight_pixels_by_inverse_dist'
WEIGHT_BY_INV_SQ_DIST_ARG_NAME = 'weight_pixels_by_inverse_sq_dist'
TARGET_FIELDS_ARG_NAME = 'target_field_names'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIRS_HELP_STRING = (
    'List of paths to input directories, each containing non-bias-corrected '
    'predictions.  Files therein will be found by `prediction_io.find_file` '
    'and read by `prediction_io.read_file`.'
)
INIT_DATE_LIMITS_HELP_STRING = (
    'List of two initialization dates, specifying the beginning and end of the '
    'training period.  Date format is "yyyymmdd".'
)
BIAS_CLUSTERING_FILE_HELP_STRING = (
    'Path to file with bias-clustering (readable by '
    '`bias_clustering.read_file`).  If you one want one model per target field '
    'per cluster, you must specify this argument.  Otherwise, leave this '
    'argument alone.'
)
PIXEL_RADIUS_HELP_STRING = (
    'Radius of influence for per-pixel models (when training the model for '
    'pixel P, will use all pixels within this radius).  If you one want one '
    'model per target field per pixel, you must specify this argument.  '
    'Otherwise, leave this argument alone.'
)
WEIGHT_BY_INV_DIST_HELP_STRING = (
    '[used only if `{0:s}` is specified] Boolean flag.  If 1, when training '
    'the model for pixel P, will weight every other pixel by the inverse of '
    'its distance to P.'
).format(
    PIXEL_RADIUS_ARG_NAME
)
WEIGHT_BY_INV_SQ_DIST_HELP_STRING = (
    '[used only if `{0:s}` is specified] Boolean flag.  If 1, when training '
    'the model for pixel P, will weight every other pixel by the inverse of '
    'its squared distance to P.'
).format(
    PIXEL_RADIUS_ARG_NAME
)
TARGET_FIELDS_HELP_STRING = (
    '1-D list of target fields for which to train models (each name must be '
    'accepted by `canadian_fwi_utils.check_field_name`).  Depending on other '
    'input args, this script will train one model per target field, one per '
    'field per pixel, or one per field per cluster.  If you want to use all '
    'target fields, leave this argument alone.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  The suite of trained isotonic-regression models '
    'will be written here, in Dill format, by `bias_correction.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIRS_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_DIRS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_DATE_LIMITS_ARG_NAME, type=str, nargs=2, required=True,
    help=INIT_DATE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BIAS_CLUSTERING_FILE_ARG_NAME, type=str, required=False, default='',
    help=BIAS_CLUSTERING_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PIXEL_RADIUS_ARG_NAME, type=float, required=False, default=-1.,
    help=PIXEL_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WEIGHT_BY_INV_DIST_ARG_NAME, type=int, required=False, default=1,
    help=WEIGHT_BY_INV_DIST_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WEIGHT_BY_INV_SQ_DIST_ARG_NAME, type=int, required=False, default=0,
    help=WEIGHT_BY_INV_SQ_DIST_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELDS_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=TARGET_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_dir_names, init_date_limit_strings,
         bias_clustering_file_name, pixel_radius_metres,
         weight_pixels_by_inverse_dist, weight_pixels_by_inverse_sq_dist,
         target_field_names, output_file_name):
    """Trains isotonic-regression model(s) to bias-correct ensemble mean.

    This is effectively the main method.

    :param prediction_dir_names: See documentation at top of this script.
    :param init_date_limit_strings: Same.
    :param bias_clustering_file_name: Same.
    :param pixel_radius_metres: Same.
    :param weight_pixels_by_inverse_dist: Same.
    :param weight_pixels_by_inverse_sq_dist: Same.
    :param target_field_names: Same.
    :param output_file_name: Same.
    """

    if pixel_radius_metres <= 0:
        pixel_radius_metres = None
    if len(target_field_names) == 1 and target_field_names[0] == '':
        target_field_names = None

    if bias_clustering_file_name == '':
        cluster_table_xarray = None
    else:
        print('Reading bias clusters from file: "{0:s}"...'.format(
            bias_clustering_file_name
        ))
        cluster_table_xarray = bias_clustering.read_file(
            bias_clustering_file_name
        )

    prediction_file_names = []
    for this_dir_name in prediction_dir_names:
        prediction_file_names += prediction_io.find_files_for_period(
            directory_name=this_dir_name,
            first_init_date_string=init_date_limit_strings[0],
            last_init_date_string=init_date_limit_strings[1],
            raise_error_if_any_missing=False,
            raise_error_if_all_missing=True
        )

    num_files = len(prediction_file_names)
    prediction_tables_xarray = [xarray.Dataset()] * num_files

    for k in range(num_files):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[k]))
        prediction_tables_xarray[k] = prediction_io.read_file(
            prediction_file_names[k]
        )
        prediction_tables_xarray[k] = prediction_io.take_ensemble_mean(
            prediction_tables_xarray[k]
        )

        if target_field_names is None:
            continue

        good_indices = numpy.array([
            numpy.where(
                prediction_tables_xarray[k][prediction_io.FIELD_NAME_KEY] == f
            )[0][0]
            for f in target_field_names
        ], dtype=int)

        prediction_tables_xarray[k] = prediction_tables_xarray[k].isel(
            {prediction_io.FIELD_DIM: good_indices}
        )

    print(SEPARATOR_STRING)

    if pixel_radius_metres is None:
        model_dict = bias_correction.train_model_suite_not_per_pixel(
            prediction_tables_xarray=prediction_tables_xarray,
            do_uncertainty_calibration=False,
            cluster_table_xarray=cluster_table_xarray,
            do_multiprocessing=True
        )
    else:
        model_dict = bias_correction.train_model_suite_per_pixel(
            prediction_tables_xarray=prediction_tables_xarray,
            do_uncertainty_calibration=False,
            pixel_radius_metres=pixel_radius_metres,
            weight_pixels_by_inverse_dist=weight_pixels_by_inverse_dist,
            weight_pixels_by_inverse_sq_dist=weight_pixels_by_inverse_sq_dist,
            do_multiprocessing=True
        )

    print(SEPARATOR_STRING)

    print('Writing model suite to: "{0:s}"...'.format(output_file_name))
    bias_correction.write_file(
        dill_file_name=output_file_name, model_dict=model_dict
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_names=getattr(INPUT_ARG_OBJECT, INPUT_DIRS_ARG_NAME),
        init_date_limit_strings=getattr(
            INPUT_ARG_OBJECT, INIT_DATE_LIMITS_ARG_NAME
        ),
        bias_clustering_file_name=getattr(
            INPUT_ARG_OBJECT, BIAS_CLUSTERING_FILE_ARG_NAME
        ),
        pixel_radius_metres=getattr(INPUT_ARG_OBJECT, PIXEL_RADIUS_ARG_NAME),
        weight_pixels_by_inverse_dist=bool(
            getattr(INPUT_ARG_OBJECT, WEIGHT_BY_INV_DIST_ARG_NAME)
        ),
        weight_pixels_by_inverse_sq_dist=bool(
            getattr(INPUT_ARG_OBJECT, WEIGHT_BY_INV_SQ_DIST_ARG_NAME)
        ),
        target_field_names=getattr(INPUT_ARG_OBJECT, TARGET_FIELDS_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
