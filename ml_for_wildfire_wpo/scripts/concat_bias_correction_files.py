"""Concatenates files with bias-correction models for different target fields.

Besides having different target fields (each file must have one), all files must
have the same metadata.
"""

import argparse
import numpy
from ml_for_wildfire_wpo.machine_learning import bias_correction

TOLERANCE = 1e-6

INPUT_FILES_ARG_NAME = 'input_file_names'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files, with each file containing bias-correction '
    'models (IR or UC) for one target field.  Each file will be read by '
    '`bias_correction.read_file`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  All models (i.e., for all target fields) will be '
    'written here by `bias_correction.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_names, output_file_name):
    """Concatenates isotonic-regression files for different target fields.

    This is effectively the main method.

    :param input_file_names: See documentation at top of this script.
    :param output_file_name: Same.
    """

    num_input_files = len(input_file_names)

    model_latitude_matrix_deg_n = None
    model_longitude_matrix_deg_e = None
    field_names = [None] * num_input_files
    pixel_radius_metres = None
    weight_pixels_by_inverse_dist = None
    weight_pixels_by_inverse_sq_dist = None
    do_uncertainty_calibration = None
    do_iso_reg_before_uncertainty_calib = None
    model_object_matrix = None

    for k in range(num_input_files):
        print('Reading models from: "{0:s}"...'.format(input_file_names[k]))
        this_model_dict = bias_correction.read_file(input_file_names[k])

        if k == 0:
            model_latitude_matrix_deg_n = this_model_dict[
                bias_correction.LATITUDES_KEY
            ]
            model_longitude_matrix_deg_e = this_model_dict[
                bias_correction.LONGITUDES_KEY
            ]
            pixel_radius_metres = this_model_dict[
                bias_correction.PIXEL_RADIUS_KEY
            ]
            weight_pixels_by_inverse_dist = this_model_dict[
                bias_correction.WEIGHT_BY_INV_DIST_KEY
            ]
            weight_pixels_by_inverse_sq_dist = this_model_dict[
                bias_correction.WEIGHT_BY_INV_SQ_DIST_KEY
            ]
            do_uncertainty_calibration = this_model_dict[
                bias_correction.DO_UNCERTAINTY_CALIB_KEY
            ]
            do_iso_reg_before_uncertainty_calib = this_model_dict[
                bias_correction.DO_IR_BEFORE_UC_KEY
            ]

            num_rows = model_latitude_matrix_deg_n.shape[0]
            num_columns = model_latitude_matrix_deg_n.shape[1]
            model_object_matrix = numpy.full(
                (num_rows, num_columns, num_input_files),
                '', dtype=object
            )

        assert numpy.allclose(
            model_latitude_matrix_deg_n,
            this_model_dict[bias_correction.LATITUDES_KEY],
            atol=TOLERANCE, equal_nan=True
        )
        assert numpy.allclose(
            model_longitude_matrix_deg_e,
            this_model_dict[bias_correction.LONGITUDES_KEY],
            atol=TOLERANCE, equal_nan=True
        )

        these_field_names = this_model_dict[bias_correction.FIELD_NAMES_KEY]
        assert len(these_field_names) == 1
        field_names[k] = these_field_names[0]

        if pixel_radius_metres is None:
            assert this_model_dict[bias_correction.PIXEL_RADIUS_KEY] is None
        else:
            assert numpy.isclose(
                pixel_radius_metres,
                this_model_dict[bias_correction.PIXEL_RADIUS_KEY],
                atol=TOLERANCE
            )

        if weight_pixels_by_inverse_dist is None:
            assert (
                this_model_dict[bias_correction.WEIGHT_BY_INV_DIST_KEY] is None
            )
        else:
            assert (
                weight_pixels_by_inverse_dist ==
                this_model_dict[bias_correction.WEIGHT_BY_INV_DIST_KEY]
            )

        if weight_pixels_by_inverse_sq_dist is None:
            assert (
                this_model_dict[bias_correction.WEIGHT_BY_INV_SQ_DIST_KEY]
                is None
            )
        else:
            assert (
                weight_pixels_by_inverse_sq_dist ==
                this_model_dict[bias_correction.WEIGHT_BY_INV_SQ_DIST_KEY]
            )

        assert (
            do_uncertainty_calibration ==
            this_model_dict[bias_correction.DO_UNCERTAINTY_CALIB_KEY]
        )

        if do_iso_reg_before_uncertainty_calib is None:
            assert (
                this_model_dict[bias_correction.DO_IR_BEFORE_UC_KEY] is None
            )
        else:
            assert (
                do_iso_reg_before_uncertainty_calib ==
                this_model_dict[bias_correction.DO_IR_BEFORE_UC_KEY]
            )

        model_object_matrix[..., k] = (
            this_model_dict[bias_correction.MODELS_KEY][..., 0]
        )

    assert len(set(field_names)) == len(field_names)

    model_dict = {
        bias_correction.MODELS_KEY: model_object_matrix,
        bias_correction.LATITUDES_KEY: model_latitude_matrix_deg_n,
        bias_correction.LONGITUDES_KEY: model_longitude_matrix_deg_e,
        bias_correction.FIELD_NAMES_KEY: field_names,
        bias_correction.PIXEL_RADIUS_KEY: pixel_radius_metres,
        bias_correction.WEIGHT_BY_INV_DIST_KEY:
            weight_pixels_by_inverse_dist,
        bias_correction.WEIGHT_BY_INV_SQ_DIST_KEY:
            weight_pixels_by_inverse_sq_dist,
        bias_correction.DO_UNCERTAINTY_CALIB_KEY: do_uncertainty_calibration,
        bias_correction.DO_IR_BEFORE_UC_KEY: do_iso_reg_before_uncertainty_calib
    }

    print('Writing concatenated model suite to: "{0:s}"...'.format(
        output_file_name
    ))
    bias_correction.write_file(
        dill_file_name=output_file_name, model_dict=model_dict
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
