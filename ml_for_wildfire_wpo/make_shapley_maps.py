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
# from keras import backend as K
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import region_mask_io
import shapley_io
import misc_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

# TODO(thunderhoser): This code does not work.  The line itself throws an error.
# tensorflow.compat.v1.keras.backend.set_learning_phase(0)

# TODO(thunderhoser): I have no idea what these lines are doing.  I copied them
# over from the ml4tc library.
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
USE_INOUT_TENSORS_ARG_NAME = 'use_inout_tensors_only'
DISABLE_TENSORFLOW2_ARG_NAME = 'disable_tensorflow2'
DISABLE_EAGER_EXEC_ARG_NAME = 'disable_eager_execution'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model (will be read by '
    '`neural_net.read_model_for_shapley`).'
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
USE_INOUT_TENSORS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will send full model (only input and output '
    'tensors) to the shap.DeepExplainer or shap.GradientExplainer object.'
)
DISABLE_TENSORFLOW2_HELP_STRING = (
    'Boolean flag.  If 1, will disable TensorFlow 2 operations.  This is '
    'PROBABLY NECESSARY to run this script.'
)
DISABLE_EAGER_EXEC_HELP_STRING = (
    'Boolean flag.  If 1, will disable eager execution in TensorFlow.  Not '
    'sure if this is necessary to run this script.'
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
    '--' + USE_INOUT_TENSORS_ARG_NAME, type=int, required=True,
    help=USE_INOUT_TENSORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DISABLE_TENSORFLOW2_ARG_NAME, type=int, required=True,
    help=DISABLE_TENSORFLOW2_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DISABLE_EAGER_EXEC_ARG_NAME, type=int, required=True,
    help=DISABLE_EAGER_EXEC_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def __dimension_to_int(dimension_object):
    """Converts `tensorflow.Dimension` object to integer.

    :param dimension_object: `tensorflow.Dimension` object.
    :return: dimension_int: Integer.
    """

    try:
        return dimension_object.value
    except:
        return dimension_object


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
    :return: model_object: Equivalent model to input, but returning a length-E
        vector.
    """

    output_layer_object = model_object.output
    has_ensemble = len(output_layer_object.shape) == 5

    # Extract the relevant target field.
    if has_ensemble:
        new_dims = (
            __dimension_to_int(output_layer_object.shape[k])
            for k in [1, 2, 4]
        )
        output_layer_object = keras.layers.Lambda(
            lambda x: x[..., target_field_index, :],
            output_shape=new_dims
        )(output_layer_object)

        output_layer_object = keras.layers.Lambda(
            lambda x: K.mean(x, axis=-1, keepdims=True),
            output_shape=new_dims
        )(output_layer_object)
    else:
        new_dims = [
            __dimension_to_int(output_layer_object.shape[k])
            for k in [1, 2]
        ]
        output_layer_object = keras.layers.Lambda(
            lambda x: x[..., target_field_index],
            output_shape=tuple(new_dims)
        )(output_layer_object)

        output_layer_object = keras.layers.Lambda(
            lambda x: K.expand_dims(x, axis=-1),
            output_shape=tuple(new_dims) + (1,)
        )(output_layer_object)

    # Multiply by region mask.
    region_mask_matrix = numpy.expand_dims(region_mask_matrix, axis=0)
    region_mask_matrix = numpy.expand_dims(region_mask_matrix, axis=-1)
    region_mask_tensor = tensorflow.convert_to_tensor(
        region_mask_matrix, dtype=output_layer_object.dtype
    )
    output_layer_object = keras.layers.Multiply()(
        [output_layer_object, region_mask_tensor]
    )
    output_layer_object = keras.layers.GlobalAveragePooling2D(
        data_format='channels_last'
    )(output_layer_object)

    # Get rid of final (ensemble-member) axis.
    output_layer_object = keras.layers.Lambda(
        lambda x: x[..., 0], output_shape=()
    )(output_layer_object)

    return keras.models.Model(
        inputs=model_object.input, outputs=output_layer_object
    )


def _apply_deepshap_1day(
        explainer_object, init_date_string, baseline_init_date_strings,
        target_field_name, validation_option_dict,
        model_input_layer_names, model_lead_time_days, model_file_name,
        region_mask_file_name, output_file_name):
    """Applies DeepSHAP for one forecast-init day.

    :param explainer_object: Instance of `shap.DeepExplainer`.
    :param init_date_string: Forecast-init day (format "yyyymmdd").
    :param baseline_init_date_strings: 1-D list of forecast-init days used for
        DeepSHAP baseline (format "yyyymmdd").
    :param target_field_name: Name of target field for which Shapley values will
        be computed.
    :param model_input_layer_names: 1-D list with names of input layers for
        trained model.
    :param validation_option_dict: Dictionary with metadata for trained model.
    :param model_lead_time_days: Model lead time.
    :param model_file_name: Path to trained model.
    :param region_mask_file_name: Path to file with region mask.
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

    if 'DeepExplainer' in str(type(explainer_object)):
        shapley_matrices = explainer_object.shap_values(
            X=predictor_matrices, check_additivity=False
        )
    else:
        shapley_matrices = explainer_object.shap_values(
            X=predictor_matrices
        )

    if isinstance(shapley_matrices[0], list):
        shapley_matrices = shapley_matrices[0]

    shapley_matrices = [sm[0, ...] for sm in shapley_matrices]

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    shapley_io.write_file(
        netcdf_file_name=output_file_name,
        shapley_matrices=shapley_matrices,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        init_date_string=init_date_string,
        baseline_init_date_strings=baseline_init_date_strings,
        region_mask_file_name=region_mask_file_name,
        target_field_name=target_field_name,
        model_input_layer_names=model_input_layer_names,
        model_lead_time_days=model_lead_time_days,
        model_file_name=model_file_name
    )


def _run(model_file_name, gfs_directory_name, target_dir_name,
         gfs_forecast_target_dir_name, model_lead_time_days,
         baseline_init_date_strings, new_init_date_strings,
         region_mask_file_name, target_field_name,
         use_inout_tensors_only, disable_tensorflow2, disable_eager_execution,
         output_dir_name):
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
    :param use_inout_tensors_only: Same.
    :param disable_tensorflow2: Same.
    :param disable_eager_execution: Same.
    :param output_dir_name: Same.
    """

    assert disable_tensorflow2 or disable_eager_execution

    if disable_tensorflow2:
        tensorflow.compat.v1.disable_v2_behavior()
    if disable_eager_execution:
        tensorflow.compat.v1.disable_eager_execution()

    # Read model.
    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model_for_shapley(model_file_name)
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

    row_indices = misc_utils.desired_latitudes_to_rows(
        grid_latitudes_deg_n=
        mask_table_xarray[region_mask_io.LATITUDE_KEY].values,
        start_latitude_deg_n=(
            vod[neural_net.INNER_LATITUDE_LIMITS_KEY][0] -
            vod[neural_net.OUTER_LATITUDE_BUFFER_KEY]
        ),
        end_latitude_deg_n=(
            vod[neural_net.INNER_LATITUDE_LIMITS_KEY][1] +
            vod[neural_net.OUTER_LATITUDE_BUFFER_KEY]
        )
    )
    mask_table_xarray = mask_table_xarray.isel({
        region_mask_io.ROW_DIM: row_indices
    })

    column_indices = misc_utils.desired_longitudes_to_columns(
        grid_longitudes_deg_e=
        mask_table_xarray[region_mask_io.LONGITUDE_KEY].values,
        start_longitude_deg_e=(
            vod[neural_net.INNER_LONGITUDE_LIMITS_KEY][0] -
            vod[neural_net.OUTER_LONGITUDE_BUFFER_KEY]
        ),
        end_longitude_deg_e=(
            vod[neural_net.INNER_LONGITUDE_LIMITS_KEY][1] +
            vod[neural_net.OUTER_LONGITUDE_BUFFER_KEY]
        )
    )
    mask_table_xarray = mask_table_xarray.isel({
        region_mask_io.COLUMN_DIM: column_indices
    })

    region_mask_matrix = (
        mask_table_xarray[region_mask_io.REGION_MASK_KEY].values
    )
    model_object = _modify_model_output(
        model_object=model_object,
        region_mask_matrix=region_mask_matrix,
        target_field_index=target_field_index
    )

    try:
        model_object.summary()
    except:
        pass

    # Read baseline examples.
    num_baseline_examples = len(baseline_init_date_strings)
    baseline_predictor_matrices = []
    good_date_indices = []

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

        good_date_indices.append(i)
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
    good_date_indices = numpy.array(good_date_indices, dtype=int)
    baseline_predictor_matrices = [
        m[good_date_indices, ...] for m in baseline_predictor_matrices
    ]
    baseline_init_date_strings = [
        baseline_init_date_strings[k] for k in good_date_indices
    ]

    # Do actual stuff.
    print(model_object.inputs)
    print(model_object.output)

    if use_inout_tensors_only:
        explainer_object = shap.DeepExplainer(
            model=(model_object.inputs, model_object.output),
            data=baseline_predictor_matrices
        )
    else:
        explainer_object = shap.DeepExplainer(
            model=model_object, data=baseline_predictor_matrices
        )

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
            region_mask_file_name=region_mask_file_name,
            validation_option_dict=validation_option_dict,
            model_input_layer_names=
            [l.name.split(':')[0] for l in model_object.input],
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
        use_inout_tensors_only=bool(getattr(
            INPUT_ARG_OBJECT, USE_INOUT_TENSORS_ARG_NAME
        )),
        disable_tensorflow2=bool(getattr(
            INPUT_ARG_OBJECT, DISABLE_TENSORFLOW2_ARG_NAME
        )),
        disable_eager_execution=bool(getattr(
            INPUT_ARG_OBJECT, DISABLE_EAGER_EXEC_ARG_NAME
        )),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
