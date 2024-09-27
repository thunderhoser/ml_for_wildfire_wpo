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

    grid_latitudes_deg_n = None
    grid_longitudes_deg_e = None
    cluster_id_matrices = [numpy.array([], dtype=int)] * num_input_files
    field_names = [None] * num_input_files
    model_dicts = [dict()] * num_input_files
    pixel_radius_metres = None
    weight_pixels_by_inverse_dist = None
    weight_pixels_by_inverse_sq_dist = None
    do_uncertainty_calibration = None
    do_iso_reg_before_uncertainty_calib = None

    for k in range(num_input_files):
        print('Reading models from: "{0:s}"...'.format(input_file_names[k]))
        this_model_and_metadata_dict = bias_correction.read_file(
            input_file_names[k]
        )
        tmmd = this_model_and_metadata_dict

        if k == 0:
            grid_latitudes_deg_n = tmmd[
                bias_correction.GRID_LATITUDES_KEY
            ]
            grid_longitudes_deg_e = tmmd[
                bias_correction.GRID_LONGITUDES_KEY
            ]
            pixel_radius_metres = tmmd[
                bias_correction.PIXEL_RADIUS_KEY
            ]
            weight_pixels_by_inverse_dist = tmmd[
                bias_correction.WEIGHT_BY_INV_DIST_KEY
            ]
            weight_pixels_by_inverse_sq_dist = tmmd[
                bias_correction.WEIGHT_BY_INV_SQ_DIST_KEY
            ]
            do_uncertainty_calibration = tmmd[
                bias_correction.DO_UNCERTAINTY_CALIB_KEY
            ]
            do_iso_reg_before_uncertainty_calib = tmmd[
                bias_correction.DO_IR_BEFORE_UC_KEY
            ]

        assert numpy.allclose(
            grid_latitudes_deg_n,
            tmmd[bias_correction.GRID_LATITUDES_KEY],
            atol=TOLERANCE, equal_nan=True
        )
        assert numpy.allclose(
            grid_longitudes_deg_e,
            tmmd[bias_correction.GRID_LONGITUDES_KEY],
            atol=TOLERANCE, equal_nan=True
        )

        these_field_names = tmmd[bias_correction.FIELD_NAMES_KEY]
        assert len(these_field_names) == 1
        field_names[k] = these_field_names[0]

        if pixel_radius_metres is None:
            assert tmmd[bias_correction.PIXEL_RADIUS_KEY] is None
        else:
            assert numpy.isclose(
                pixel_radius_metres,
                tmmd[bias_correction.PIXEL_RADIUS_KEY],
                atol=TOLERANCE
            )

        if weight_pixels_by_inverse_dist is None:
            assert (
                tmmd[bias_correction.WEIGHT_BY_INV_DIST_KEY] is None
            )
        else:
            assert (
                weight_pixels_by_inverse_dist ==
                tmmd[bias_correction.WEIGHT_BY_INV_DIST_KEY]
            )

        if weight_pixels_by_inverse_sq_dist is None:
            assert (
                tmmd[bias_correction.WEIGHT_BY_INV_SQ_DIST_KEY]
                is None
            )
        else:
            assert (
                weight_pixels_by_inverse_sq_dist ==
                tmmd[bias_correction.WEIGHT_BY_INV_SQ_DIST_KEY]
            )

        assert (
            do_uncertainty_calibration ==
            tmmd[bias_correction.DO_UNCERTAINTY_CALIB_KEY]
        )

        if do_iso_reg_before_uncertainty_calib is None:
            assert (
                tmmd[bias_correction.DO_IR_BEFORE_UC_KEY] is None
            )
        else:
            assert (
                do_iso_reg_before_uncertainty_calib ==
                tmmd[bias_correction.DO_IR_BEFORE_UC_KEY]
            )

        cluster_id_matrices[k] = (
            tmmd[bias_correction.CLUSTER_ID_MATRIX_KEY][..., 0]
        )
        model_dicts[k] = tmmd[bias_correction.MODEL_DICT_KEY]

    assert len(set(field_names)) == len(field_names)

    overall_model_dict = model_dicts[0]
    for k in range(1, num_input_files):
        overall_model_dict.update(model_dicts[k])

    cluster_id_matrix = numpy.stack(cluster_id_matrices, axis=-1)

    model_and_metadata_dict = {
        bias_correction.GRID_LATITUDES_KEY: grid_latitudes_deg_n,
        bias_correction.GRID_LONGITUDES_KEY: grid_longitudes_deg_e,
        bias_correction.FIELD_NAMES_KEY: field_names,
        bias_correction.CLUSTER_ID_MATRIX_KEY: cluster_id_matrix,
        bias_correction.MODEL_DICT_KEY: overall_model_dict,
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
        dill_file_name=output_file_name, model_dict=model_and_metadata_dict
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
