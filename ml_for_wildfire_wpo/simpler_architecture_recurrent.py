"""USE ONCE AND DESTROY."""

import os
import sys
import numpy
import keras
import tensorflow
from tensorflow.keras.layers import Layer

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import architecture_utils
import chiu_net_architecture
import chiu_net_pp_architecture as basic_arch

GFS_3D_DIMENSIONS_KEY = basic_arch.GFS_3D_DIMENSIONS_KEY
GFS_2D_DIMENSIONS_KEY = basic_arch.GFS_2D_DIMENSIONS_KEY
ERA5_CONST_DIMENSIONS_KEY = basic_arch.ERA5_CONST_DIMENSIONS_KEY
LAGTGT_DIMENSIONS_KEY = basic_arch.LAGTGT_DIMENSIONS_KEY
PREDN_BASELINE_DIMENSIONS_KEY = basic_arch.PREDN_BASELINE_DIMENSIONS_KEY
USE_RESIDUAL_BLOCKS_KEY = basic_arch.USE_RESIDUAL_BLOCKS_KEY

GFS_FC_MODULE_NUM_CONV_LAYERS_KEY = basic_arch.GFS_FC_MODULE_NUM_CONV_LAYERS_KEY
GFS_FC_MODULE_DROPOUT_RATES_KEY = basic_arch.GFS_FC_MODULE_DROPOUT_RATES_KEY
GFS_FC_MODULE_USE_3D_CONV = basic_arch.GFS_FC_MODULE_USE_3D_CONV
LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY = (
    basic_arch.LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY
)
LAGTGT_FC_MODULE_DROPOUT_RATES_KEY = (
    basic_arch.LAGTGT_FC_MODULE_DROPOUT_RATES_KEY
)
LAGTGT_FC_MODULE_USE_3D_CONV = basic_arch.LAGTGT_FC_MODULE_USE_3D_CONV

NUM_LEVELS_KEY = basic_arch.NUM_LEVELS_KEY

GFS_ENCODER_NUM_CONV_LAYERS_KEY = basic_arch.GFS_ENCODER_NUM_CONV_LAYERS_KEY
GFS_ENCODER_NUM_CHANNELS_KEY = basic_arch.GFS_ENCODER_NUM_CHANNELS_KEY
GFS_ENCODER_DROPOUT_RATES_KEY = basic_arch.GFS_ENCODER_DROPOUT_RATES_KEY
LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY = (
    basic_arch.LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY
)
LAGTGT_ENCODER_NUM_CHANNELS_KEY = basic_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY
LAGTGT_ENCODER_DROPOUT_RATES_KEY = (
    basic_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY
)
DECODER_NUM_CONV_LAYERS_KEY = basic_arch.DECODER_NUM_CONV_LAYERS_KEY
DECODER_NUM_CHANNELS_KEY = basic_arch.DECODER_NUM_CHANNELS_KEY
UPSAMPLING_DROPOUT_RATES_KEY = basic_arch.UPSAMPLING_DROPOUT_RATES_KEY
SKIP_DROPOUT_RATES_KEY = basic_arch.SKIP_DROPOUT_RATES_KEY

INCLUDE_PENULTIMATE_KEY = basic_arch.INCLUDE_PENULTIMATE_KEY
PENULTIMATE_DROPOUT_RATE_KEY = basic_arch.PENULTIMATE_DROPOUT_RATE_KEY
INNER_ACTIV_FUNCTION_KEY = basic_arch.INNER_ACTIV_FUNCTION_KEY
INNER_ACTIV_FUNCTION_ALPHA_KEY = basic_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY
OUTPUT_ACTIV_FUNCTION_KEY = basic_arch.OUTPUT_ACTIV_FUNCTION_KEY
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = basic_arch.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY
L1_WEIGHT_KEY = basic_arch.L1_WEIGHT_KEY
L2_WEIGHT_KEY = basic_arch.L2_WEIGHT_KEY
USE_BATCH_NORM_KEY = basic_arch.USE_BATCH_NORM_KEY
ENSEMBLE_SIZE_KEY = basic_arch.ENSEMBLE_SIZE_KEY
USE_EVIDENTIAL_KEY = basic_arch.USE_EVIDENTIAL_KEY

OPTIMIZER_FUNCTION_KEY = basic_arch.OPTIMIZER_FUNCTION_KEY


class RemoveTimeDimLayer(Layer):
    def __init__(self, **kwargs):
        super(RemoveTimeDimLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tensorflow.squeeze(inputs, axis=-2)

    def get_config(self):
        base_config = super(RemoveTimeDimLayer, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_model(option_dict, loss_function, metric_list):
    """Creates Chiu-net++.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    :param option_dict: See doc for `chiu_net_architecture.check_args`.
    :param loss_function: Loss function.
    :param metric_list: 1-D list of metrics.
    :return: model_object: Instance of `keras.models.Model`, with the
        Chiu-net++ architecture.
    """

    option_dict = chiu_net_architecture.check_args(option_dict)
    # error_checking.assert_is_list(metric_list)

    input_dimensions_gfs_3d = option_dict[GFS_3D_DIMENSIONS_KEY]
    input_dimensions_gfs_2d = option_dict[GFS_2D_DIMENSIONS_KEY]
    input_dimensions_era5 = option_dict[ERA5_CONST_DIMENSIONS_KEY]
    input_dimensions_lagged_target = option_dict[LAGTGT_DIMENSIONS_KEY]
    input_dimensions_predn_baseline = option_dict[PREDN_BASELINE_DIMENSIONS_KEY]
    use_residual_blocks = option_dict[USE_RESIDUAL_BLOCKS_KEY]

    if input_dimensions_predn_baseline is not None:
        these_indices = numpy.array([0, 1, 3], dtype=int)
        assert numpy.array_equal(
            input_dimensions_predn_baseline,
            input_dimensions_lagged_target[these_indices]
        )

    gfs_fcst_num_conv_layers = option_dict[GFS_FC_MODULE_NUM_CONV_LAYERS_KEY]
    gfs_fcst_dropout_rates = option_dict[GFS_FC_MODULE_DROPOUT_RATES_KEY]
    gfs_fcst_use_3d_conv = option_dict[GFS_FC_MODULE_USE_3D_CONV]
    lagtgt_fcst_num_conv_layers = option_dict[
        LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY
    ]
    lagtgt_fcst_dropout_rates = option_dict[LAGTGT_FC_MODULE_DROPOUT_RATES_KEY]
    lagtgt_fcst_use_3d_conv = option_dict[LAGTGT_FC_MODULE_USE_3D_CONV]

    num_levels = option_dict[NUM_LEVELS_KEY]

    gfs_encoder_num_conv_layers_by_level = option_dict[
        GFS_ENCODER_NUM_CONV_LAYERS_KEY
    ]
    gfs_encoder_num_channels_by_level = option_dict[
        GFS_ENCODER_NUM_CHANNELS_KEY
    ]
    gfs_encoder_dropout_rate_by_level = option_dict[
        GFS_ENCODER_DROPOUT_RATES_KEY
    ]
    lagtgt_encoder_num_conv_layers_by_level = option_dict[
        LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY
    ]
    lagtgt_encoder_num_channels_by_level = option_dict[
        LAGTGT_ENCODER_NUM_CHANNELS_KEY
    ]
    lagtgt_encoder_dropout_rate_by_level = option_dict[
        LAGTGT_ENCODER_DROPOUT_RATES_KEY
    ]
    decoder_num_conv_layers_by_level = option_dict[DECODER_NUM_CONV_LAYERS_KEY]
    decoder_num_channels_by_level = option_dict[DECODER_NUM_CHANNELS_KEY]
    upsampling_dropout_rate_by_level = option_dict[UPSAMPLING_DROPOUT_RATES_KEY]
    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]

    include_penultimate_conv = option_dict[INCLUDE_PENULTIMATE_KEY]
    penultimate_conv_dropout_rate = option_dict[PENULTIMATE_DROPOUT_RATE_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = option_dict[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = option_dict[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]
    ensemble_size = option_dict[ENSEMBLE_SIZE_KEY]
    use_evidential_nn = option_dict[USE_EVIDENTIAL_KEY]

    if use_evidential_nn:
        ensemble_size = 4

    assert not use_evidential_nn
    assert ensemble_size == 1

    optimizer_function = option_dict[OPTIMIZER_FUNCTION_KEY]
    num_gfs_lead_times = input_dimensions_gfs_2d[-2]

    this_name = 'gfs_2d_inputs'
    input_layer_object_gfs_2d = keras.layers.Input(
        shape=tuple(input_dimensions_gfs_2d.tolist()),
        name=this_name
    )
    put_time_first_layer_object = keras.layers.Permute(dims=(3, 1, 2, 4))

    this_name = 'lagged_target_inputs'
    input_layer_object_lagged_target = keras.layers.Input(
        shape=tuple(input_dimensions_lagged_target.tolist()),
        name=this_name
    )

    this_name = 'predn_baseline_inputs'
    input_layer_object_predn_baseline = keras.layers.Input(
        shape=tuple(input_dimensions_predn_baseline.tolist()),
        name=this_name
    )

    num_target_lag_times = input_dimensions_lagged_target[-2]
    num_target_fields = input_dimensions_lagged_target[-1]
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    # TODO: Make sure to wrap in TimeDistributed.
    gfs_conv2d_layer_object = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=3,
        num_kernel_columns=3,
        num_rows_per_stride=1,
        num_columns_per_stride=1,
        num_filters=gfs_encoder_num_channels_by_level[0],
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )
    put_time_last_layer_object = keras.layers.Permute(
        dims=(2, 3, 1, 4), name=this_name
    )

    gfs_conv3d_layer_object = architecture_utils.get_3d_conv_layer(
        num_kernel_rows=1,
        num_kernel_columns=1,
        num_kernel_heights=num_gfs_lead_times,
        num_rows_per_stride=1,
        num_columns_per_stride=1,
        num_heights_per_stride=1,
        num_filters=gfs_encoder_num_channels_by_level[0],
        padding_type_string=architecture_utils.NO_PADDING_STRING,
        weight_regularizer=regularizer_object
    )
    remove_time_layer_object = RemoveTimeDimLayer()

    # TODO: Make sure to wrap in TimeDistributed.
    lagtgt_conv2d_layer_object = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=3,
        num_kernel_columns=3,
        num_rows_per_stride=1,
        num_columns_per_stride=1,
        num_filters=lagtgt_encoder_num_channels_by_level[0],
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )

    lagtgt_conv3d_layer_object = architecture_utils.get_3d_conv_layer(
        num_kernel_rows=1,
        num_kernel_columns=1,
        num_kernel_heights=num_target_lag_times,
        num_rows_per_stride=1,
        num_columns_per_stride=1,
        num_heights_per_stride=1,
        num_filters=lagtgt_encoder_num_channels_by_level[0],
        padding_type_string=architecture_utils.NO_PADDING_STRING,
        weight_regularizer=regularizer_object
    )

    concat_channels_layer_object = keras.layers.Concatenate(axis=-1)

    output_conv_layer_object = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=3,
        num_kernel_columns=3,
        num_rows_per_stride=1,
        num_columns_per_stride=1,
        num_filters=num_target_fields,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object,
        layer_name=this_name
    )

    add_baseline_layer_object = keras.layers.Add()

    def _construct_basic_model(x_gfs_2d, x_lagged_target, x_predn_baseline):
        print(x_predn_baseline)
        x0 = put_time_first_layer_object(x_gfs_2d)
        x1 = put_time_first_layer_object(x_lagged_target)
        x0 = keras.layers.TimeDistributed(gfs_conv2d_layer_object)(x0)
        x1 = keras.layers.TimeDistributed(lagtgt_conv2d_layer_object)(x1)
        x0 = put_time_last_layer_object(x0)
        x1 = put_time_last_layer_object(x1)
        x0 = gfs_conv3d_layer_object(x0)
        x1 = lagtgt_conv3d_layer_object(x1)
        x0 = remove_time_layer_object(x0)
        x1 = remove_time_layer_object(x1)
        x = concat_channels_layer_object([x0, x1])
        x = output_conv_layer_object(x)
        x = add_baseline_layer_object([x_predn_baseline, x])
        return x
    
    def _construct_recurrent_model(num_integration_steps):
        output_layer_objects = [None] * num_integration_steps
        output_layer_objects[0] = _construct_basic_model(
            x_gfs_2d=input_layer_object_gfs_2d,
            x_lagged_target=input_layer_object_lagged_target,
            x_predn_baseline=input_layer_object_predn_baseline
        )
    
        for i in range(1, num_integration_steps):
            output_layer_objects[i] = _construct_basic_model(
                x_gfs_2d=input_layer_object_gfs_2d,
                x_lagged_target=input_layer_object_lagged_target,
                x_predn_baseline=output_layer_objects[i - 1]
            )

        return output_layer_objects

    input_layer_objects = [
        input_layer_object_gfs_2d,
        input_layer_object_lagged_target,
        input_layer_object_predn_baseline
    ]

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=_construct_recurrent_model(2)
    )

    # TODO(thunderhoser): Hard-coded loss dictionary is a HACK.
    print('LOSS FUNCTION:')
    print(loss_function)

    model_object.compile(
        # loss={
        #     'final_output_step0': loss_function,
        #     'final_output_step1': loss_function
        # },
        loss=loss_function,
        optimizer=optimizer_function,
        metrics=metric_list
    )

    model_object.summary()
    return model_object
