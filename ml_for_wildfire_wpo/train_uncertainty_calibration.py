"""Trains uncertainty-calibration model to bias-correct ensemble spread."""

import os
import sys
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io
import bias_correction

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SENTINEL_VALUE = -1e6

INPUT_DIRS_ARG_NAME = 'input_prediction_dir_names'
INIT_DATE_LIMITS_ARG_NAME = 'init_date_limit_strings'
ONE_MODEL_PER_PIXEL_ARG_NAME = 'one_model_per_pixel'
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
ONE_MODEL_PER_PIXEL_HELP_STRING = (
    'Boolean flag.  If 1 (0), will train one UC model per pixel (one UC model '
    'for the whole domain).'
)
PIXEL_RADIUS_HELP_STRING = (
    '[used only if {0:s} == 1] When training the model for pixel P, will use '
    'all pixels within this radius.'
).format(
    ONE_MODEL_PER_PIXEL_ARG_NAME
)
WEIGHT_BY_INV_DIST_HELP_STRING = (
    '[used only if {0:s} == 1] Boolean flag.  If 1, when training the model '
    'for pixel P, will weight every other pixel by the inverse of its distance '
    'to P.'
).format(
    ONE_MODEL_PER_PIXEL_ARG_NAME
)
WEIGHT_BY_INV_SQ_DIST_HELP_STRING = (
    '[used only if {0:s} == 1] Boolean flag.  If 1, when training the model '
    'for pixel P, will weight every other pixel by the inverse of its '
    '*squared* distance to P.'
).format(
    ONE_MODEL_PER_PIXEL_ARG_NAME
)
TARGET_FIELDS_HELP_STRING = (
    '1-D list of target fields for which to train UC models.  Each field name '
    'must be accepted by `canadian_fwi_utils.check_field_name`.  If you want '
    'to train for all target fields, leave this argument alone.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  The suite of trained UC models will be saved here, '
    'in Dill format.'
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
    '--' + ONE_MODEL_PER_PIXEL_ARG_NAME, type=int, required=True,
    help=ONE_MODEL_PER_PIXEL_HELP_STRING
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
         one_model_per_pixel, pixel_radius_metres,
         weight_pixels_by_inverse_dist, weight_pixels_by_inverse_sq_dist,
         target_field_names, output_file_name):
    """Trains uncertainty-calibration model to bias-correct ensemble spread.

    This is effectively the main method.

    :param prediction_dir_names: See documentation at top of this script.
    :param init_date_limit_strings: Same.
    :param one_model_per_pixel: Same.
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
    do_iso_reg_before_uncertainty_calib = None

    for k in range(num_files):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[k]))
        prediction_tables_xarray[k] = prediction_io.read_file(
            prediction_file_names[k]
        )
        ptx_k = prediction_tables_xarray[k]

        if do_iso_reg_before_uncertainty_calib is None:
            do_iso_reg_before_uncertainty_calib = (
                ptx_k.attrs[prediction_io.ISOTONIC_MODEL_FILE_KEY] is not None
            )

        assert (
            do_iso_reg_before_uncertainty_calib ==
            (ptx_k.attrs[prediction_io.ISOTONIC_MODEL_FILE_KEY] is not None)
        )
        assert (
            ptx_k.attrs[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY] is None
        )

        prediction_tables_xarray[k] = (
            prediction_io.prep_for_uncertainty_calib_training(
                prediction_tables_xarray[k]
            )
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

    model_dict = bias_correction.train_model_suite(
        prediction_tables_xarray=prediction_tables_xarray,
        one_model_per_pixel=one_model_per_pixel,
        pixel_radius_metres=pixel_radius_metres,
        weight_pixels_by_inverse_dist=weight_pixels_by_inverse_dist,
        weight_pixels_by_inverse_sq_dist=weight_pixels_by_inverse_sq_dist,
        do_uncertainty_calibration=True,
        do_iso_reg_before_uncertainty_calib=do_iso_reg_before_uncertainty_calib
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
        one_model_per_pixel=bool(
            getattr(INPUT_ARG_OBJECT, ONE_MODEL_PER_PIXEL_ARG_NAME)
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
