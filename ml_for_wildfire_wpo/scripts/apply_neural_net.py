"""Applies trained neural net -- inference time!"""

import argparse
from gewittergefahr.gg_utils import time_conversion
from ml_for_wildfire_wpo.io import prediction_io
from ml_for_wildfire_wpo.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_EXAMPLES_PER_BATCH = 16

MODEL_FILE_ARG_NAME = 'input_model_file_name'
GFS_DIRECTORY_ARG_NAME = 'input_gfs_directory_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
GFS_FCST_TARGET_DIR_ARG_NAME = 'input_gfs_fcst_target_dir_name'
DATE_LIMITS_ARG_NAME = 'init_date_limit_strings'
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


def _run(model_file_name, gfs_directory_name, target_dir_name,
         gfs_forecast_target_dir_name, init_date_limit_strings,
         output_dir_name):
    """Applies trained neural net -- inference time!

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param gfs_directory_name: Same.
    :param target_dir_name: Same.
    :param gfs_forecast_target_dir_name: Same.
    :param init_date_limit_strings: Same.
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

    for this_init_date_string in init_date_strings:
        try:
            data_dict = neural_net.create_data(
                option_dict=validation_option_dict,
                init_date_string=this_init_date_string
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

        output_file_name = prediction_io.find_file(
            directory_name=output_dir_name,
            init_date_string=this_init_date_string,
            raise_error_if_missing=False
        )

        print('Writing results to: "{0:s}"...'.format(output_file_name))
        prediction_io.write_file(
            netcdf_file_name=output_file_name,
            target_matrix_with_weights=target_matrix_with_weights[0, ...],
            prediction_matrix=prediction_matrix[0, ...],
            grid_latitudes_deg_n=grid_latitudes_deg_n,
            grid_longitudes_deg_e=grid_longitudes_deg_e,
            field_names=validation_option_dict[neural_net.TARGET_FIELDS_KEY],
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
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
