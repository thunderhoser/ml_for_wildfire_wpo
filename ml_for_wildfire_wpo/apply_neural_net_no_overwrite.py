"""Applies trained neural net -- inference time!"""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import prediction_io
import canadian_fwi_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_EXAMPLES_PER_BATCH = 16

MODEL_FILE_ARG_NAME = 'input_model_file_name'
GFS_DIRECTORY_ARG_NAME = 'input_gfs_directory_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
GFS_FCST_TARGET_DIR_ARG_NAME = 'input_gfs_fcst_target_dir_name'
DATE_LIMITS_ARG_NAME = 'init_date_limit_strings'
MODEL_LEAD_TIME_ARG_NAME = 'model_lead_time_days'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model (will be read by `neural_net.read_model`).'
)
GFS_DIRECTORY_HELP_STRING = (
    'Name of directory with GFS data (predictors).  Files therein will be '
    'found by `gfs_io.find_file` and read by `gfs_io.read_file`.'
)
TARGET_DIR_HELP_STRING = (
    'Name of directory with target fields.  Files therein will be found by '
    '`canadian_fwo_io.find_file` and read by `canadian_fwo_io.read_file`.'
)
GFS_FCST_TARGET_DIR_HELP_STRING = (
    'Name of directory with raw-GFS-forecast target fields.  Files therein '
    'will be found by `gfs_daily_io.find_file` and read by '
    '`gfs_daily_io.read_file`.'
)
DATE_LIMITS_HELP_STRING = (
    'Length-2 list with first and last GFS model runs (init times in format '
    '"yyyymmdd") to be used.'
)
MODEL_LEAD_TIME_HELP_STRING = 'Model lead time.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written here by '
    '`prediction_io.write_file`, to exact locations determined by '
    '`prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GFS_DIRECTORY_ARG_NAME, type=str, required=True,
    help=GFS_DIRECTORY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GFS_FCST_TARGET_DIR_ARG_NAME, type=str, required=True,
    help=GFS_FCST_TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DATE_LIMITS_ARG_NAME, type=str, nargs=2, required=True,
    help=DATE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_LEAD_TIME_ARG_NAME, type=int, required=True,
    help=MODEL_LEAD_TIME_HELP_STRING
)


def _run(model_file_name, gfs_directory_name, target_dir_name,
         gfs_forecast_target_dir_name, init_date_limit_strings,
         model_lead_time_days, output_dir_name):
    """Applies trained neural net -- inference time!

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param gfs_directory_name: Same.
    :param target_dir_name: Same.
    :param gfs_forecast_target_dir_name: Same.
    :param init_date_limit_strings: Same.
    :param model_lead_time_days: Same.
    :param output_dir_name: Same.
    """

    init_date_strings = time_conversion.get_spc_dates_in_range(
        init_date_limit_strings[0], init_date_limit_strings[1]
    )

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    validation_option_dict[neural_net.GFS_DIRECTORY_KEY] = gfs_directory_name
    validation_option_dict[neural_net.TARGET_DIRECTORY_KEY] = target_dir_name
    validation_option_dict[neural_net.GFS_FORECAST_TARGET_DIR_KEY] = (
        gfs_forecast_target_dir_name
    )
    print(SEPARATOR_STRING)

    target_field_names = copy.deepcopy(
        validation_option_dict[neural_net.TARGET_FIELDS_KEY]
    )
    constrain_dsr = (
        canadian_fwi_utils.FWI_NAME in target_field_names and
        canadian_fwi_utils.DSR_NAME not in target_field_names
    )

    if constrain_dsr:
        target_field_names.append(canadian_fwi_utils.DSR_NAME)
        fwi_index = target_field_names.index(canadian_fwi_utils.FWI_NAME)
    else:
        fwi_index = None

    print('constrain_dsr = {0:d}'.format(
        int(constrain_dsr)
    ))

    for this_init_date_string in init_date_strings:
        output_file_name = prediction_io.find_file(
            directory_name=output_dir_name,
            init_date_string=this_init_date_string,
            raise_error_if_missing=False
        )

        if os.path.isfile(output_file_name):
            try:
                prediction_table_xarray = prediction_io.read_file(output_file_name)
                field_names = prediction_table_xarray[prediction_io.FIELD_NAME_KEY].values
                continue
            except:
                pass

        try:
            data_dict = neural_net.create_data(
                option_dict=validation_option_dict,
                init_date_string=this_init_date_string,
                model_lead_time_days=model_lead_time_days
            )
            print(SEPARATOR_STRING)
        except:
            print(SEPARATOR_STRING)
            continue

        predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]
        target_matrix_with_weights = data_dict[
            neural_net.TARGETS_AND_WEIGHTS_KEY
        ]
        grid_latitudes_deg_n = data_dict[neural_net.GRID_LATITUDES_KEY]
        grid_longitudes_deg_e = data_dict[neural_net.GRID_LONGITUDES_KEY]

        prediction_matrix = neural_net.apply_model(
            model_object=model_object,
            predictor_matrices=predictor_matrices,
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            verbose=True
        )

        if constrain_dsr:
            predicted_dsr_matrix = 0.0272 * numpy.power(
                prediction_matrix[..., fwi_index, :], 1.77
            )
            prediction_matrix = numpy.concatenate([
                prediction_matrix,
                numpy.expand_dims(predicted_dsr_matrix, axis=-2)
            ], axis=-2)

            target_dsr_matrix = 0.0272 * numpy.power(
                target_matrix_with_weights[..., fwi_index], 1.77
            )
            target_matrix_with_weights = numpy.concatenate([
                target_matrix_with_weights[..., :-1],
                numpy.expand_dims(target_dsr_matrix, axis=-1),
                target_matrix_with_weights[..., [-1]]
            ], axis=-1)

        print('Writing results to: "{0:s}"...'.format(output_file_name))
        prediction_io.write_file(
            netcdf_file_name=output_file_name,
            target_matrix_with_weights=target_matrix_with_weights[0, ...],
            prediction_matrix=prediction_matrix[0, ...],
            grid_latitudes_deg_n=grid_latitudes_deg_n,
            grid_longitudes_deg_e=grid_longitudes_deg_e,
            field_names=target_field_names,
            init_date_string=this_init_date_string,
            model_file_name=model_file_name
        )
        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        gfs_directory_name=getattr(INPUT_ARG_OBJECT, GFS_DIRECTORY_ARG_NAME),
        target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        gfs_forecast_target_dir_name=getattr(
            INPUT_ARG_OBJECT, GFS_FCST_TARGET_DIR_ARG_NAME
        ),
        init_date_limit_strings=getattr(INPUT_ARG_OBJECT, DATE_LIMITS_ARG_NAME),
        model_lead_time_days=getattr(
            INPUT_ARG_OBJECT, MODEL_LEAD_TIME_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )