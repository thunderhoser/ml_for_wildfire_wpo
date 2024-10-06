"""Uses DeepSHAP algorithm to create Shapley maps."""

import os
import sys
import argparse
import numpy
import tensorflow
import shap
import shap.explainers
import keras.layers
import keras.models
from keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import region_mask_io
import shapley_io
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

# tensorflow.compat.v1.disable_v2_behavior()
# tensorflow.compat.v1.disable_eager_execution()
# tensorflow.config.threading.set_inter_op_parallelism_threads(1)
# tensorflow.config.threading.set_intra_op_parallelism_threads(1)

MODEL_FILE_ARG_NAME = 'input_model_file_name'
GFS_DIRECTORY_ARG_NAME = 'input_gfs_directory_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
GFS_FCST_TARGET_DIR_ARG_NAME = 'input_gfs_fcst_target_dir_name'
MODEL_LEAD_TIME_ARG_NAME = 'model_lead_time_days'
BASELINE_INIT_DATES_ARG_NAME = 'baseline_init_date_strings'
NEW_INIT_DATES_ARG_NAME = 'new_init_date_strings'
REGION_MASK_FILE_ARG_NAME = 'input_region_mask_file_name'
TARGET_FIELD_ARG_NAME = 'target_field_name'
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
MODEL_LEAD_TIME_HELP_STRING = 'Model lead time.'
BASELINE_INIT_DATES_HELP_STRING = (
    'List of forecast-init days (format "yyyymmdd") used to create baseline '
    'for DeepSHAP.'
)
NEW_INIT_DATES_HELP_STRING = (
    'List of forecast-init days (format "yyyymmdd") for which to compute '
    'Shapley values.'
)
REGION_MASK_FILE_HELP_STRING = (
    'Path to file with region mask (will be read by '
    '`region_mask_io.read_file`).  Shapley values will be computed for the '
    'field `{0:s}` averaged over the spatial region found in this file.'
).format(
    TARGET_FIELD_ARG_NAME
)
TARGET_FIELD_HELP_STRING = (
    'Name of target field (must be accepted by '
    '`canadian_fwi_utils.check_field_name`).  Shapley values will be computed '
    'for this field averaged over the spatial region found in the file `{0:s}`.'
).format(
    REGION_MASK_FILE_ARG_NAME
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Shapley values will be written here by '
    '`shapley_io.write_file`, to exact locations determined by '
    '`shapley_io.find_file`.'
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
    '--' + MODEL_LEAD_TIME_ARG_NAME, type=int, required=True,
    help=MODEL_LEAD_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BASELINE_INIT_DATES_ARG_NAME, type=str, nargs='+', required=True,
    help=BASELINE_INIT_DATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEW_INIT_DATES_ARG_NAME, type=str, nargs='+', required=True,
    help=NEW_INIT_DATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + REGION_MASK_FILE_ARG_NAME, type=str, required=True,
    help=REGION_MASK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELD_ARG_NAME, type=str, required=True,
    help=TARGET_FIELD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _modify_model_output(model_object, region_mask_matrix, target_field_index):
    """Modifies model output.

    The original model output should have dimensions E x M x N x F or
    E x M x N x F x S, where E = number of data examples; M = number of grid
    rows; N = number of grid columns; F = number of target fields; and
    S = number of ensemble members.  We want to collapse this whole array to
    just a length-E vector, by averaging over ensemble members, finding the
    desired target field, and averaging over relevant spatial locations (in the
    desired region).

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param region_mask_matrix: M-by-N numpy array of Boolean flags, where True
        means that the pixel is within the desired region.
    :param target_field_index: Index of desired target field, along the target-
        field axis of the model's output tensor.
    :return: model_predict_function: Function, returning the aggregated model
        predictions as a length-E vector.
    """

    def model_predict(predictor_matrices):
        prediction_matrix = model_object.predict(predictor_matrices)
        if len(prediction_matrix.shape) == 5:
            prediction_matrix_4d = numpy.mean(prediction_matrix, axis=-1)
        else:
            prediction_matrix_4d = prediction_matrix

        prediction_matrix_3d = prediction_matrix_4d[..., target_field_index]

        region_mask_matrix_3d = numpy.expand_dims(
            region_mask_matrix.astype(int), axis=0
        )
        return numpy.mean(
            prediction_matrix_3d * region_mask_matrix_3d,
            axis=(1, 2)
        )

    return model_predict


def _apply_deepshap_1day(
        explainer_object, init_date_string, baseline_init_date_strings,
        target_field_name, region_mask_matrix, validation_option_dict,
        model_input_layer_names, model_lead_time_days, model_file_name,
        output_file_name):
    """Applies DeepSHAP for one forecast-init day.

    M = number of rows in grid
    N = number of columns in grid

    :param explainer_object: Instance of `shap.DeepExplainer`.
    :param init_date_string: Forecast-init day (format "yyyymmdd").
    :param baseline_init_date_strings: 1-D list of forecast-init days used for
        DeepSHAP baseline (format "yyyymmdd").
    :param target_field_name: Name of target field for which Shapley values will
        be computed.
    :param region_mask_matrix: M-by-N numpy array of Boolean flags, indicating
        the spatial region of interest.  Shapley values will be computed for the
        given target field, averaged over this area of interest.
    :param model_input_layer_names: 1-D list with names of input layers for
        trained model.
    :param validation_option_dict: Dictionary with metadata for trained model.
    :param model_lead_time_days: Model lead time.
    :param model_file_name: Path to trained model.
    :param output_file_name: Path to output file.  Shapley values will be
        written here.
    """

    try:
        data_dict = neural_net.create_data(
            option_dict=validation_option_dict,
            init_date_string=init_date_string,
            model_lead_time_days=model_lead_time_days
        )
        print(SEPARATOR_STRING)
    except:
        print(SEPARATOR_STRING)
        return

    predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    grid_latitudes_deg_n = data_dict[neural_net.GRID_LATITUDES_KEY]
    grid_longitudes_deg_e = data_dict[neural_net.GRID_LONGITUDES_KEY]
    del data_dict

    shapley_matrices = explainer_object.shap_values(
        X=predictor_matrices, check_additivity=False
    )

    # TODO(thunderhoser): I'm not sure what this does -- carried over from
    # rapid-intensification code.
    if isinstance(shapley_matrices[0], list):
        shapley_matrices = shapley_matrices[0]

    shapley_matrices = [sm[0, ...] for sm in shapley_matrices]

    # TODO(thunderhoser): I might want to include predictor matrices in the
    # output file -- but I don't know.  These would be normalized predictors,
    # but plots will need to include unnormalized predictors.  So it might be
    # best to just generate all this on the fly when plotting.
    print('Writing results to: "{0:s}"...'.format(output_file_name))
    shapley_io.write_file(
        netcdf_file_name=output_file_name,
        shapley_matrices=shapley_matrices,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        init_date_string=init_date_string,
        baseline_init_date_strings=baseline_init_date_strings,
        region_mask_file_name=region_mask_matrix,
        target_field_name=target_field_name,
        model_input_layer_names=model_input_layer_names,
        model_lead_time_days=model_lead_time_days,
        model_file_name=model_file_name
    )


def _run(model_file_name, gfs_directory_name, target_dir_name,
         gfs_forecast_target_dir_name, model_lead_time_days,
         baseline_init_date_strings, new_init_date_strings,
         region_mask_file_name, target_field_name, output_dir_name):
    """Uses DeepSHAP algorithm to create Shapley maps.

    This is effectively the main method.

    :param model_file_name: See documentation at top of this script.
    :param gfs_directory_name: Same.
    :param target_dir_name: Same.
    :param gfs_forecast_target_dir_name: Same.
    :param model_lead_time_days: Same.
    :param baseline_init_date_strings: Same.
    :param new_init_date_strings: Same.
    :param region_mask_file_name: Same.
    :param target_field_name: Same.
    :param output_dir_name: Same.
    """

    # Read model.
    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    # Read model metadata.
    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    vod = model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    vod[neural_net.GFS_DIRECTORY_KEY] = gfs_directory_name
    vod[neural_net.TARGET_DIRECTORY_KEY] = target_dir_name
    vod[neural_net.GFS_FORECAST_TARGET_DIR_KEY] = gfs_forecast_target_dir_name
    print(SEPARATOR_STRING)

    # Check some input args.
    all_target_field_names = vod[neural_net.TARGET_FIELDS_KEY]
    if not isinstance(all_target_field_names, list):
        all_target_field_names = all_target_field_names.tolist()

    target_field_index = all_target_field_names.index(target_field_name)
    validation_option_dict = vod

    # Change model's output layer to include only the given region and target
    # field.
    mask_table_xarray = region_mask_io.read_file(region_mask_file_name)
    region_mask_matrix = (
        mask_table_xarray[region_mask_io.REGION_MASK_KEY].values
    )
    # model_predict_function = _modify_model_output(
    #     model_object=model_object,
    #     region_mask_matrix=region_mask_matrix,
    #     target_field_index=target_field_index
    # )

    output_layer_object = model_object.output
    output_layer_object = keras.layers.GlobalAveragePooling3D(
        data_format='channels_last'
    )(output_layer_object)
    output_layer_object = keras.layers.Lambda(
        lambda x: K.mean(x, axis=1), output_shape=(None,)
    )(output_layer_object)

    model_predict_function = keras.models.Model(
        inputs=model_object.input, outputs=output_layer_object
    )

    # Read baseline examples.
    num_baseline_examples = len(baseline_init_date_strings)
    baseline_predictor_matrices = []

    for i in range(num_baseline_examples):
        try:
            data_dict = neural_net.create_data(
                option_dict=validation_option_dict,
                init_date_string=baseline_init_date_strings[i],
                model_lead_time_days=model_lead_time_days
            )
            print(SEPARATOR_STRING)
        except:
            print(SEPARATOR_STRING)
            continue

        these_predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]

        if len(baseline_predictor_matrices) == 0:
            for this_predictor_matrix in these_predictor_matrices:
                these_dim = (
                    (num_baseline_examples,) + this_predictor_matrix.shape[1:]
                )
                baseline_predictor_matrices.append(
                    numpy.full(these_dim, numpy.nan)
                )

        for j in range(len(these_predictor_matrices)):
            baseline_predictor_matrices[j][i, ...] = (
                these_predictor_matrices[j][0, ...]
            )

    del data_dict

    # Do actual stuff.
    explainer_object = shap.DeepExplainer(
        model=model_predict_function, data=baseline_predictor_matrices
    )
    del baseline_predictor_matrices

    num_new_examples = len(new_init_date_strings)

    for i in range(num_new_examples):
        this_output_file_name = shapley_io.find_file(
            directory_name=output_dir_name,
            init_date_string=new_init_date_strings[i],
            raise_error_if_missing=False
        )

        _apply_deepshap_1day(
            explainer_object=explainer_object,
            init_date_string=new_init_date_strings[i],
            baseline_init_date_strings=baseline_init_date_strings,
            target_field_name=target_field_name,
            region_mask_matrix=region_mask_matrix,
            validation_option_dict=validation_option_dict,
            model_input_layer_names=[l.name for l in model_object.input],
            model_lead_time_days=model_lead_time_days,
            model_file_name=model_file_name,
            output_file_name=this_output_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        gfs_directory_name=getattr(INPUT_ARG_OBJECT, GFS_DIRECTORY_ARG_NAME),
        target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        gfs_forecast_target_dir_name=getattr(
            INPUT_ARG_OBJECT, GFS_FCST_TARGET_DIR_ARG_NAME
        ),
        model_lead_time_days=getattr(
            INPUT_ARG_OBJECT, MODEL_LEAD_TIME_ARG_NAME
        ),
        baseline_init_date_strings=getattr(
            INPUT_ARG_OBJECT, BASELINE_INIT_DATES_ARG_NAME
        ),
        new_init_date_strings=getattr(
            INPUT_ARG_OBJECT, NEW_INIT_DATES_ARG_NAME
        ),
        region_mask_file_name=getattr(
            INPUT_ARG_OBJECT, REGION_MASK_FILE_ARG_NAME
        ),
        target_field_name=getattr(
            INPUT_ARG_OBJECT, TARGET_FIELD_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
