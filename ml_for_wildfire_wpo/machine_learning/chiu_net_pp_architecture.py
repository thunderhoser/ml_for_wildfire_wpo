"""Methods for building a Chiu-net++.

The Chiu-net++ is a hybrid between the Chiu net
(https://doi.org/10.1109/LRA.2020.2992184) and U-net++.
"""

import numpy
import keras
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml_for_wildfire_wpo.machine_learning import \
    chiu_net_architecture as chiu_net_arch

GFS_3D_DIMENSIONS_KEY = chiu_net_arch.GFS_3D_DIMENSIONS_KEY
GFS_2D_DIMENSIONS_KEY = chiu_net_arch.GFS_2D_DIMENSIONS_KEY
ERA5_CONST_DIMENSIONS_KEY = chiu_net_arch.ERA5_CONST_DIMENSIONS_KEY
LAGTGT_DIMENSIONS_KEY = chiu_net_arch.LAGTGT_DIMENSIONS_KEY

GFS_FC_MODULE_NUM_CONV_LAYERS_KEY = (
    chiu_net_arch.GFS_FC_MODULE_NUM_CONV_LAYERS_KEY
)
GFS_FC_MODULE_DROPOUT_RATES_KEY = chiu_net_arch.GFS_FC_MODULE_DROPOUT_RATES_KEY
GFS_FC_MODULE_USE_3D_CONV = chiu_net_arch.GFS_FC_MODULE_USE_3D_CONV
LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY = (
    chiu_net_arch.LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY
)
LAGTGT_FC_MODULE_DROPOUT_RATES_KEY = (
    chiu_net_arch.LAGTGT_FC_MODULE_DROPOUT_RATES_KEY
)
LAGTGT_FC_MODULE_USE_3D_CONV = chiu_net_arch.LAGTGT_FC_MODULE_USE_3D_CONV

NUM_LEVELS_KEY = chiu_net_arch.NUM_LEVELS_KEY

GFS_ENCODER_NUM_CONV_LAYERS_KEY = chiu_net_arch.GFS_ENCODER_NUM_CONV_LAYERS_KEY
GFS_ENCODER_NUM_CHANNELS_KEY = chiu_net_arch.GFS_ENCODER_NUM_CHANNELS_KEY
GFS_ENCODER_DROPOUT_RATES_KEY = chiu_net_arch.GFS_ENCODER_DROPOUT_RATES_KEY
LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY = (
    chiu_net_arch.LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY
)
LAGTGT_ENCODER_NUM_CHANNELS_KEY = chiu_net_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY
LAGTGT_ENCODER_DROPOUT_RATES_KEY = (
    chiu_net_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY
)
DECODER_NUM_CONV_LAYERS_KEY = chiu_net_arch.DECODER_NUM_CONV_LAYERS_KEY
DECODER_NUM_CHANNELS_KEY = chiu_net_arch.DECODER_NUM_CHANNELS_KEY
UPSAMPLING_DROPOUT_RATES_KEY = chiu_net_arch.UPSAMPLING_DROPOUT_RATES_KEY
SKIP_DROPOUT_RATES_KEY = chiu_net_arch.SKIP_DROPOUT_RATES_KEY

INCLUDE_PENULTIMATE_KEY = chiu_net_arch.INCLUDE_PENULTIMATE_KEY
PENULTIMATE_DROPOUT_RATE_KEY = chiu_net_arch.PENULTIMATE_DROPOUT_RATE_KEY
INNER_ACTIV_FUNCTION_KEY = chiu_net_arch.INNER_ACTIV_FUNCTION_KEY
INNER_ACTIV_FUNCTION_ALPHA_KEY = chiu_net_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY
OUTPUT_ACTIV_FUNCTION_KEY = chiu_net_arch.OUTPUT_ACTIV_FUNCTION_KEY
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = chiu_net_arch.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY
L1_WEIGHT_KEY = chiu_net_arch.L1_WEIGHT_KEY
L2_WEIGHT_KEY = chiu_net_arch.L2_WEIGHT_KEY
USE_BATCH_NORM_KEY = chiu_net_arch.USE_BATCH_NORM_KEY
ENSEMBLE_SIZE_KEY = chiu_net_arch.ENSEMBLE_SIZE_KEY

OPTIMIZER_FUNCTION_KEY = chiu_net_arch.OPTIMIZER_FUNCTION_KEY


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

    # TODO(thunderhoser): Might want to combine info from lag/lead targets and
    # GFS earlier in the network -- instead of waiting until the forecast
    # module, which is the bottleneck.

    # TODO(thunderhoser): Might want more efficient way to incorporate ERA5 data
    # in network -- one that doesn't involve repeating ERA5 data over the
    # GFS-lead-time axis and target-lag/lead-time axis.

    option_dict = chiu_net_arch.check_args(option_dict)
    error_checking.assert_is_list(metric_list)

    input_dimensions_gfs_3d = option_dict[GFS_3D_DIMENSIONS_KEY]
    input_dimensions_gfs_2d = option_dict[GFS_2D_DIMENSIONS_KEY]
    input_dimensions_era5 = option_dict[ERA5_CONST_DIMENSIONS_KEY]
    input_dimensions_lagged_target = option_dict[LAGTGT_DIMENSIONS_KEY]

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

    optimizer_function = option_dict[OPTIMIZER_FUNCTION_KEY]
    num_gfs_lead_times = None

    if input_dimensions_gfs_3d is None:
        input_layer_object_gfs_3d = None
        layer_object_gfs_3d = None
    else:
        input_layer_object_gfs_3d = keras.layers.Input(
            shape=tuple(input_dimensions_gfs_3d.tolist()),
            name='gfs_3d_inputs'
        )
        layer_object_gfs_3d = keras.layers.Permute(
            dims=(4, 1, 2, 3, 5),
            name='gfs_3d_put-time-first'
        )(input_layer_object_gfs_3d)

        new_dims = (
            input_dimensions_gfs_3d[3],
            input_dimensions_gfs_3d[0],
            input_dimensions_gfs_3d[1],
            input_dimensions_gfs_3d[2] * input_dimensions_gfs_3d[4]
        )
        layer_object_gfs_3d = keras.layers.Reshape(
            target_shape=new_dims,
            name='gfs_3d_flatten-pressure-levels'
        )(layer_object_gfs_3d)

        num_gfs_lead_times = input_dimensions_gfs_3d[-2]

    if input_dimensions_gfs_2d is None:
        input_layer_object_gfs_2d = None
        layer_object_gfs_2d = None
    else:
        input_layer_object_gfs_2d = keras.layers.Input(
            shape=tuple(input_dimensions_gfs_2d.tolist()),
            name='gfs_2d_inputs'
        )
        layer_object_gfs_2d = keras.layers.Permute(
            dims=(3, 1, 2, 4),
            name='gfs_2d_put-time-first'
        )(input_layer_object_gfs_2d)

        num_gfs_lead_times = input_dimensions_gfs_2d[-2]

    if input_dimensions_gfs_3d is None:
        layer_object_gfs = layer_object_gfs_2d
    elif input_dimensions_gfs_2d is None:
        layer_object_gfs = layer_object_gfs_3d
    else:
        layer_object_gfs = keras.layers.Concatenate(
            axis=-1, name='gfs_concat-2d-and-3d'
        )(
            [layer_object_gfs_3d, layer_object_gfs_2d]
        )

    input_layer_object_lagged_target = keras.layers.Input(
        shape=tuple(input_dimensions_lagged_target.tolist()),
        name='lagged_target_inputs'
    )
    layer_object_lagged_target = keras.layers.Permute(
        dims=(3, 1, 2, 4), name='lagged_targets_put-time-first'
    )(input_layer_object_lagged_target)

    num_target_lag_times = input_dimensions_lagged_target[-2]
    num_target_fields = input_dimensions_lagged_target[-1]

    if input_dimensions_era5 is None:
        input_layer_object_era5 = None
    else:
        input_layer_object_era5 = keras.layers.Input(
            shape=tuple(input_dimensions_era5.tolist()), name='era5_inputs'
        )

        new_dims = (1,) + tuple(input_dimensions_era5.tolist())
        layer_object_era5 = keras.layers.Reshape(
            target_shape=new_dims, name='era5_add-time-dim'
        )(input_layer_object_era5)

        this_layer_object = keras.layers.Concatenate(
            axis=-4, name='era5_add-gfs-times'
        )(
            num_gfs_lead_times * [layer_object_era5]
        )

        layer_object_gfs = keras.layers.Concatenate(
            axis=-1, name='gfs_concat-era5'
        )(
            [layer_object_gfs, this_layer_object]
        )

        this_layer_object = keras.layers.Concatenate(
            axis=-4, name='era5_add-lag-times'
        )(
            num_target_lag_times * [layer_object_era5]
        )

        layer_object_lagged_target = keras.layers.Concatenate(
            axis=-1, name='lagged_targets_concat-era5'
        )(
            [layer_object_lagged_target, this_layer_object]
        )

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    gfs_encoder_conv_layer_objects = [None] * (num_levels + 1)
    gfs_fcst_module_layer_objects = [None] * (num_levels + 1)
    gfs_encoder_pooling_layer_objects = [None] * num_levels

    for i in range(num_levels + 1):
        for j in range(gfs_encoder_num_conv_layers_by_level[i]):
            if j == 0:
                if i == 0:
                    this_input_layer_object = layer_object_gfs
                else:
                    this_input_layer_object = (
                        gfs_encoder_pooling_layer_objects[i - 1]
                    )
            else:
                this_input_layer_object = gfs_encoder_conv_layer_objects[i]

            this_name = 'gfs_encoder_level{0:d}_conv{1:d}'.format(i, j)
            this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=gfs_encoder_num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )
            gfs_encoder_conv_layer_objects[i] = keras.layers.TimeDistributed(
                this_conv_layer_object, name=this_name
            )(this_input_layer_object)

            this_name = 'gfs_encoder_level{0:d}_conv{1:d}_activation'.format(
                i, j
            )
            gfs_encoder_conv_layer_objects[i] = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha,
                    layer_name=this_name
                )(gfs_encoder_conv_layer_objects[i])
            )

            if gfs_encoder_dropout_rate_by_level[i] > 0:
                this_name = 'gfs_encoder_level{0:d}_conv{1:d}_dropout'.format(
                    i, j
                )
                gfs_encoder_conv_layer_objects[i] = (
                    architecture_utils.get_dropout_layer(
                        dropout_fraction=gfs_encoder_dropout_rate_by_level[i],
                        layer_name=this_name
                    )(gfs_encoder_conv_layer_objects[i])
                )

            if use_batch_normalization:
                this_name = 'gfs_encoder_level{0:d}_conv{1:d}_bn'.format(i, j)
                gfs_encoder_conv_layer_objects[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(gfs_encoder_conv_layer_objects[i])
                )

        this_name = 'gfs_fcst_level{0:d}_put-time-last'.format(i)
        gfs_fcst_module_layer_objects[i] = keras.layers.Permute(
            dims=(2, 3, 1, 4), name=this_name
        )(gfs_encoder_conv_layer_objects[i])

        if not gfs_fcst_use_3d_conv:
            orig_dims = gfs_fcst_module_layer_objects[i].get_shape()
            new_dims = orig_dims[1:-2] + [orig_dims[-2] * orig_dims[-1]]

            this_name = 'gfs_fcst_level{0:d}_remove-time-dim'.format(i)
            gfs_fcst_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(gfs_fcst_module_layer_objects[i])

        for j in range(gfs_fcst_num_conv_layers):
            this_name = 'gfs_fcst_level{0:d}_conv{1:d}'.format(i, j)

            if gfs_fcst_use_3d_conv:
                if j == 0:
                    gfs_fcst_module_layer_objects[i] = (
                        architecture_utils.get_3d_conv_layer(
                            num_kernel_rows=1, num_kernel_columns=1,
                            num_kernel_heights=num_gfs_lead_times,
                            num_rows_per_stride=1, num_columns_per_stride=1,
                            num_heights_per_stride=1,
                            num_filters=gfs_encoder_num_channels_by_level[i],
                            padding_type_string=
                            architecture_utils.NO_PADDING_STRING,
                            weight_regularizer=regularizer_object,
                            layer_name=this_name
                        )(gfs_fcst_module_layer_objects[i])
                    )

                    new_dims = (
                        gfs_fcst_module_layer_objects[i].shape[1:3] +
                        [gfs_fcst_module_layer_objects[i].shape[-1]]
                    )

                    this_name = 'gfs_fcst_level{0:d}_remove-time-dim'.format(i)
                    gfs_fcst_module_layer_objects[i] = keras.layers.Reshape(
                        target_shape=new_dims, name=this_name
                    )(gfs_fcst_module_layer_objects[i])
                else:
                    gfs_fcst_module_layer_objects[i] = (
                        architecture_utils.get_2d_conv_layer(
                            num_kernel_rows=3, num_kernel_columns=3,
                            num_rows_per_stride=1, num_columns_per_stride=1,
                            num_filters=gfs_encoder_num_channels_by_level[i],
                            padding_type_string=
                            architecture_utils.YES_PADDING_STRING,
                            weight_regularizer=regularizer_object,
                            layer_name=this_name
                        )(gfs_fcst_module_layer_objects[i])
                    )
            else:
                gfs_fcst_module_layer_objects[i] = architecture_utils.get_2d_conv_layer(
                    num_kernel_rows=3, num_kernel_columns=3,
                    num_rows_per_stride=1, num_columns_per_stride=1,
                    num_filters=gfs_encoder_num_channels_by_level[i],
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )(gfs_fcst_module_layer_objects[i])

            this_name = 'gfs_fcst_level{0:d}_conv{1:d}_activation'.format(i, j)
            gfs_fcst_module_layer_objects[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(gfs_fcst_module_layer_objects[i])

            if gfs_fcst_dropout_rates[j] > 0:
                this_name = 'gfs_fcst_level{0:d}_conv{1:d}_dropout'.format(i, j)
                gfs_fcst_module_layer_objects[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=gfs_fcst_dropout_rates[j],
                    layer_name=this_name
                )(gfs_fcst_module_layer_objects[i])

            if use_batch_normalization:
                this_name = 'gfs_fcst_level{0:d}_conv{1:d}_bn'.format(i, j)
                gfs_fcst_module_layer_objects[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(gfs_fcst_module_layer_objects[i])
                )

        if i == num_levels:
            break

        this_name = 'gfs_encoder_level{0:d}_pooling'.format(i)
        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_rows_per_stride=2, num_columns_per_stride=2,
            pooling_type_string=architecture_utils.MAX_POOLING_STRING
        )
        gfs_encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(gfs_encoder_conv_layer_objects[i])

    lagtgt_encoder_conv_layer_objects = [None] * (num_levels + 1)
    lagtgt_fcst_module_layer_objects = [None] * (num_levels + 1)
    lagtgt_encoder_pooling_layer_objects = [None] * num_levels

    for i in range(num_levels + 1):
        for j in range(lagtgt_encoder_num_conv_layers_by_level[i]):
            if j == 0:
                if i == 0:
                    this_input_layer_object = layer_object_lagged_target
                else:
                    this_input_layer_object = (
                        lagtgt_encoder_pooling_layer_objects[i - 1]
                    )
            else:
                this_input_layer_object = lagtgt_encoder_conv_layer_objects[i]

            this_name = 'lagtgt_encoder_level{0:d}_conv{1:d}'.format(i, j)
            this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=lagtgt_encoder_num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )
            lagtgt_encoder_conv_layer_objects[i] = keras.layers.TimeDistributed(
                this_conv_layer_object, name=this_name
            )(this_input_layer_object)

            this_name = 'lagtgt_encoder_level{0:d}_conv{1:d}_activation'.format(
                i, j
            )
            lagtgt_encoder_conv_layer_objects[i] = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha,
                    layer_name=this_name
                )(lagtgt_encoder_conv_layer_objects[i])
            )

            if lagtgt_encoder_dropout_rate_by_level[i] > 0:
                this_name = (
                    'lagtgt_encoder_level{0:d}_conv{1:d}_dropout'
                ).format(i, j)

                lagtgt_encoder_conv_layer_objects[i] = (
                    architecture_utils.get_dropout_layer(
                        dropout_fraction=
                        lagtgt_encoder_dropout_rate_by_level[i],
                        layer_name=this_name
                    )(lagtgt_encoder_conv_layer_objects[i])
                )

            if use_batch_normalization:
                this_name = 'lagtgt_encoder_level{0:d}_conv{1:d}_bn'.format(
                    i, j
                )
                lagtgt_encoder_conv_layer_objects[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(lagtgt_encoder_conv_layer_objects[i])
                )

        this_name = 'lagtgt_fcst_level{0:d}_put-time-last'.format(i)
        lagtgt_fcst_module_layer_objects[i] = keras.layers.Permute(
            dims=(2, 3, 1, 4), name=this_name
        )(lagtgt_encoder_conv_layer_objects[i])

        if not lagtgt_fcst_use_3d_conv:
            orig_dims = lagtgt_fcst_module_layer_objects[i].get_shape()
            new_dims = orig_dims[1:-2] + [orig_dims[-2] * orig_dims[-1]]

            this_name = 'lagtgt_fcst_level{0:d}_remove-time-dim'.format(i)
            lagtgt_fcst_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(lagtgt_fcst_module_layer_objects[i])

        for j in range(lagtgt_fcst_num_conv_layers):
            this_name = 'lagtgt_fcst_level{0:d}_conv{1:d}'.format(i, j)

            if lagtgt_fcst_use_3d_conv:
                if j == 0:
                    lagtgt_fcst_module_layer_objects[i] = (
                        architecture_utils.get_3d_conv_layer(
                            num_kernel_rows=1, num_kernel_columns=1,
                            num_kernel_heights=num_target_lag_times,
                            num_rows_per_stride=1, num_columns_per_stride=1,
                            num_heights_per_stride=1,
                            num_filters=lagtgt_encoder_num_channels_by_level[i],
                            padding_type_string=
                            architecture_utils.NO_PADDING_STRING,
                            weight_regularizer=regularizer_object,
                            layer_name=this_name
                        )(lagtgt_fcst_module_layer_objects[i])
                    )

                    new_dims = (
                        lagtgt_fcst_module_layer_objects[i].shape[1:3] +
                        [lagtgt_fcst_module_layer_objects[i].shape[-1]]
                    )

                    this_name = 'lagtgt_fcst_level{0:d}_remove-time-dim'.format(i)
                    lagtgt_fcst_module_layer_objects[i] = keras.layers.Reshape(
                        target_shape=new_dims, name=this_name
                    )(lagtgt_fcst_module_layer_objects[i])
                else:
                    lagtgt_fcst_module_layer_objects[i] = (
                        architecture_utils.get_2d_conv_layer(
                            num_kernel_rows=3, num_kernel_columns=3,
                            num_rows_per_stride=1, num_columns_per_stride=1,
                            num_filters=lagtgt_encoder_num_channels_by_level[i],
                            padding_type_string=
                            architecture_utils.YES_PADDING_STRING,
                            weight_regularizer=regularizer_object,
                            layer_name=this_name
                        )(lagtgt_fcst_module_layer_objects[i])
                    )
            else:
                lagtgt_fcst_module_layer_objects[i] = (
                    architecture_utils.get_2d_conv_layer(
                        num_kernel_rows=3, num_kernel_columns=3,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_filters=lagtgt_encoder_num_channels_by_level[i],
                        padding_type_string=architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=regularizer_object,
                        layer_name=this_name
                    )(lagtgt_fcst_module_layer_objects[i])
                )

            this_name = 'lagtgt_fcst_level{0:d}_conv{1:d}_activation'.format(i, j)
            lagtgt_fcst_module_layer_objects[i] = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha,
                    layer_name=this_name
                )(lagtgt_fcst_module_layer_objects[i])
            )

            if lagtgt_fcst_dropout_rates[j] > 0:
                this_name = 'lagtgt_fcst_level{0:d}_conv{1:d}_dropout'.format(i, j)
                lagtgt_fcst_module_layer_objects[i] = (
                    architecture_utils.get_dropout_layer(
                        dropout_fraction=lagtgt_fcst_dropout_rates[j],
                        layer_name=this_name
                    )(lagtgt_fcst_module_layer_objects[i])
                )

            if use_batch_normalization:
                this_name = 'lagtgt_fcst_level{0:d}_conv{1:d}_bn'.format(i, j)
                lagtgt_fcst_module_layer_objects[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(lagtgt_fcst_module_layer_objects[i])
                )

        if i == num_levels:
            break

        this_name = 'lagtgt_encoder_level{0:d}_pooling'.format(i)
        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_rows_per_stride=2, num_columns_per_stride=2,
            pooling_type_string=architecture_utils.MAX_POOLING_STRING
        )
        lagtgt_encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(lagtgt_encoder_conv_layer_objects[i])

    last_conv_layer_matrix = numpy.full(
        (num_levels + 1, num_levels + 1), '', dtype=object
    )

    for i in range(num_levels + 1):
        this_name = 'fcst_level{0:d}_concat'.format(i)

        last_conv_layer_matrix[i, 0] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )([
            gfs_fcst_module_layer_objects[i],
            lagtgt_fcst_module_layer_objects[i]
        ])

        i_new = i + 0
        j = 0

        while i_new > 0:
            i_new -= 1
            j += 1

            this_name = 'block{0:d}-{1:d}_upsampling'.format(i_new, j)
            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2), name=this_name
            )(last_conv_layer_matrix[i_new + 1, j - 1])

            this_name = 'block{0:d}-{1:d}_upconv'.format(i_new, j)
            this_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=decoder_num_channels_by_level[i_new],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_layer_object)

            this_name = 'block{0:d}-{1:d}_upconv_activation'.format(i_new, j)
            this_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(this_layer_object)

            if upsampling_dropout_rate_by_level[i_new] > 0:
                this_name = 'block{0:d}-{1:d}_upconv_dropout'.format(i_new, j)
                this_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=upsampling_dropout_rate_by_level[i_new],
                    layer_name=this_name
                )(this_layer_object)

            num_upconv_rows = this_layer_object.get_shape()[1]
            num_desired_rows = last_conv_layer_matrix[i_new, 0].get_shape()[1]
            num_padding_rows = num_desired_rows - num_upconv_rows

            num_upconv_columns = this_layer_object.get_shape()[2]
            num_desired_columns = (
                last_conv_layer_matrix[i_new, 0].get_shape()[2]
            )
            num_padding_columns = num_desired_columns - num_upconv_columns

            if num_padding_rows + num_padding_columns > 0:
                padding_arg = ((0, num_padding_rows), (0, num_padding_columns))

                this_layer_object = keras.layers.ZeroPadding2D(
                    padding=padding_arg
                )(this_layer_object)

            last_conv_layer_matrix[i_new, j] = this_layer_object

            this_name = 'block{0:d}-{1:d}_skip'.format(i_new, j)
            last_conv_layer_matrix[i_new, j] = keras.layers.Concatenate(
                axis=-1, name=this_name
            )(last_conv_layer_matrix[i_new, :(j + 1)].tolist())

            for k in range(decoder_num_conv_layers_by_level[i_new]):
                this_name = 'block{0:d}-{1:d}_skipconv{2:d}'.format(i_new, j, k)
                last_conv_layer_matrix[i_new, j] = (
                    architecture_utils.get_2d_conv_layer(
                        num_kernel_rows=3, num_kernel_columns=3,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_filters=decoder_num_channels_by_level[i_new],
                        padding_type_string=
                        architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=regularizer_object,
                        layer_name=this_name
                    )(last_conv_layer_matrix[i_new, j])
                )

                this_name = 'block{0:d}-{1:d}_skipconv{2:d}_activation'.format(
                    i_new, j, k
                )

                last_conv_layer_matrix[i_new, j] = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=inner_activ_function_name,
                        alpha_for_relu=inner_activ_function_alpha,
                        alpha_for_elu=inner_activ_function_alpha,
                        layer_name=this_name
                    )(last_conv_layer_matrix[i_new, j])
                )

                if skip_dropout_rate_by_level[i_new] > 0:
                    this_name = 'block{0:d}-{1:d}_skipconv{2:d}_dropout'.format(
                        i_new, j, k
                    )

                    last_conv_layer_matrix[i_new, j] = (
                        architecture_utils.get_dropout_layer(
                            dropout_fraction=skip_dropout_rate_by_level[i_new],
                            layer_name=this_name
                        )(last_conv_layer_matrix[i_new, j])
                    )

                if use_batch_normalization:
                    this_name = 'block{0:d}-{1:d}_skipconv{2:d}_bn'.format(
                        i_new, j, k
                    )

                    last_conv_layer_matrix[i_new, j] = (
                        architecture_utils.get_batch_norm_layer(
                            layer_name=this_name
                        )(last_conv_layer_matrix[i_new, j])
                    )

    if include_penultimate_conv:
        last_conv_layer_matrix[0, -1] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=3, num_kernel_columns=3,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=2 * num_target_fields * ensemble_size,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object, layer_name='penultimate_conv'
        )(last_conv_layer_matrix[0, -1])

        last_conv_layer_matrix[0, -1] = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name='penultimate_conv_activation'
        )(last_conv_layer_matrix[0, -1])

        if penultimate_conv_dropout_rate > 0:
            last_conv_layer_matrix[0, -1] = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=penultimate_conv_dropout_rate,
                    layer_name='penultimate_conv_dropout'
                )(last_conv_layer_matrix[0, -1])
            )

        if use_batch_normalization:
            last_conv_layer_matrix[0, -1] = (
                architecture_utils.get_batch_norm_layer(
                    layer_name='penultimate_conv_bn'
                )(last_conv_layer_matrix[0, -1])
            )

    output_layer_object = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=1, num_kernel_columns=1,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_filters=num_target_fields * ensemble_size,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object, layer_name='last_conv'
    )(last_conv_layer_matrix[0, -1])

    output_layer_object = architecture_utils.get_activation_layer(
        activation_function_string=output_activ_function_name,
        alpha_for_relu=output_activ_function_alpha,
        alpha_for_elu=output_activ_function_alpha,
        layer_name='last_conv_activation'
    )(output_layer_object)

    if ensemble_size > 1:
        new_dims = (
            input_dimensions_lagged_target[0],
            input_dimensions_lagged_target[1],
            num_target_fields,
            ensemble_size
        )
        output_layer_object = keras.layers.Reshape(
            target_shape=new_dims, name='reshape_predictions'
        )(output_layer_object)

    input_layer_objects = [
        l for l in [
            input_layer_object_gfs_3d, input_layer_object_gfs_2d,
            input_layer_object_era5, input_layer_object_lagged_target
        ] if l is not None
    ]
    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=output_layer_object
    )

    model_object.compile(
        loss=loss_function, optimizer=optimizer_function,
        metrics=metric_list
    )

    model_object.summary()
    return model_object
