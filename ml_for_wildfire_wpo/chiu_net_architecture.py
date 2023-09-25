"""Methods for building Chiu nets.

https://doi.org/10.1109/LRA.2020.2992184
"""

import os
import sys
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import architecture_utils

GFS_3D_DIMENSIONS_KEY = 'input_dimensions_gfs_3d'
GFS_2D_DIMENSIONS_KEY = 'input_dimensions_gfs_2d'
ERA5_CONST_DIMENSIONS_KEY = 'input_dimensions_era5_constants'
LAGGED_TARGET_DIMENSIONS_KEY = 'input_dimensions_lagged_target'
NUM_FC_CONV_LAYERS_KEY = 'num_conv_layers_in_fc_module'
FC_MODULE_DROPOUT_RATES_KEY = 'fc_module_dropout_rates'
USE_3D_CONV_IN_FC_KEY = 'use_3d_conv_in_fc_module'
NUM_LEVELS_KEY = 'num_levels'
CONV_LAYER_COUNTS_KEY = 'num_conv_layers_by_level'
CHANNEL_COUNTS_KEY = 'num_channels_by_level'
ENCODER_DROPOUT_RATES_KEY = 'encoder_dropout_rate_by_level'
DECODER_DROPOUT_RATES_KEY = 'decoder_dropout_rate_by_level'
SKIP_DROPOUT_RATES_KEY = 'skip_dropout_rate_by_level'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
OUTPUT_ACTIV_FUNCTION_KEY = 'output_activ_function_name'
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = 'output_activ_function_alpha'
L1_WEIGHT_KEY = 'l1_weight'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'

DEFAULT_ARCHITECTURE_OPTION_DICT = {
    NUM_FC_CONV_LAYERS_KEY: 1,
    FC_MODULE_DROPOUT_RATES_KEY: numpy.full(1, 0.),
    USE_3D_CONV_IN_FC_KEY: True,
    NUM_LEVELS_KEY: 4,
    CONV_LAYER_COUNTS_KEY: numpy.array([2, 2, 2, 4, 4], dtype=int),
    CHANNEL_COUNTS_KEY: numpy.array([16, 24, 32, 64, 128], dtype=int),
    ENCODER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    DECODER_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    SKIP_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: architecture_utils.SIGMOID_FUNCTION_STRING,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    L1_WEIGHT_KEY: 0.,
    L2_WEIGHT_KEY: 0.001,
    USE_BATCH_NORM_KEY: True
}


def _check_args(option_dict):
    """Error-checks input arguments.

    L = number of levels in encoder = number of levels in decoder

    :param option_dict: Dictionary with the following keys.
    option_dict['input_dimensions_gfs_3d']: numpy array with input dimensions
        for 3-D GFS data.  Array elements should be [num_rows, num_columns,
        num_pressure_levels, num_lead_times, num_fields].  If predictors do not
        include 3-D GFS data, make this None.
    option_dict['input_dimensions_gfs_2d']: numpy array with input dimensions
        for 2-D GFS data.  Array elements should be [num_rows, num_columns,
        num_lead_times, num_fields].  If predictors do not include 2-D GFS data,
        make this None.
    option_dict['input_dimensions_era5_constants']: numpy array with input
        dimensions for ERA5 constants.  Array elements should be [num_rows,
        num_columns, num_fields].  If predictors do not include ERA5 constants,
        make this None.
    option_dict['input_dimensions_lagged_targets']: numpy array with input
        dimensions for lagged targets.  Array elements should be [num_rows,
        num_columns, num_lag_times, 1].
    option_dict['num_conv_layers_in_fc_module']: Number of conv layers in
        forecasting module.
    option_dict['fc_module_dropout_rates']: length-N numpy array of dropout
        rates in forecasting module, where N = 'num_conv_layers_in_fc_module'.
    option_dict['num_levels']: L in the above discussion.
    option_dict['num_conv_layers_by_level']: length-(L + 1) numpy array with
        number of conv layers at each level.
    option_dict['num_channels_by_level']: length-(L + 1) numpy array with number
        of channels at each level.
    option_dict['encoder_dropout_rate_by_level']: length-(L + 1) numpy array
        with dropout rate for conv layers in encoder at each level.
    option_dict['decoder_dropout_rate_by_level']: length-L numpy array
        with dropout rate for conv layers in decoder at each level.
    option_dict['skip_dropout_rate_by_level']: length-L numpy array with dropout
        rate for conv layer after skip connection at each level.
    option_dict['inner_activ_function_name']: Name of activation function for
        all inner (non-output) layers.  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict['inner_activ_function_alpha']: Alpha (slope parameter) for
        activation function for all inner layers.  Applies only to ReLU and eLU.
    option_dict['output_activ_function_name']: Same as
        `inner_activ_function_name` but for output layer.
    option_dict['output_activ_function_alpha']: Same as
        `inner_activ_function_alpha` but for output layer.
    option_dict['l1_weight']: Weight for L_1 regularization.
    option_dict['l2_weight']: Weight for L_2 regularization.
    option_dict['use_batch_normalization']: Boolean flag.  If True, will use
        batch normalization after each inner (non-output) conv layer.

    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_ARCHITECTURE_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    input_dimensions_gfs_3d = option_dict[GFS_3D_DIMENSIONS_KEY]
    input_dimensions_gfs_2d = option_dict[GFS_2D_DIMENSIONS_KEY]
    input_dimensions_era5_constants = option_dict[ERA5_CONST_DIMENSIONS_KEY]
    input_dimensions_lagged_target = option_dict[LAGGED_TARGET_DIMENSIONS_KEY]

    assert not (
        input_dimensions_gfs_3d is None and input_dimensions_gfs_2d is None
    )

    num_grid_rows = -1
    num_grid_columns = -1
    num_gfs_lead_times = -1

    if input_dimensions_gfs_3d is not None:
        error_checking.assert_is_numpy_array(
            input_dimensions_gfs_3d,
            exact_dimensions=numpy.array([5], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(input_dimensions_gfs_3d)
        error_checking.assert_is_greater_numpy_array(input_dimensions_gfs_3d, 0)

        num_grid_rows = input_dimensions_gfs_3d[0]
        num_grid_columns = input_dimensions_gfs_3d[1]
        num_gfs_lead_times = input_dimensions_gfs_3d[3]

    if input_dimensions_gfs_2d is not None:
        error_checking.assert_is_numpy_array(
            input_dimensions_gfs_2d,
            exact_dimensions=numpy.array([4], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(input_dimensions_gfs_2d)
        error_checking.assert_is_greater_numpy_array(input_dimensions_gfs_2d, 0)

        if input_dimensions_gfs_3d is not None:
            these_dim = numpy.array([
                num_grid_rows, num_grid_columns, num_gfs_lead_times,
                input_dimensions_gfs_2d[3]
            ], dtype=int)

            assert numpy.array_equal(input_dimensions_gfs_2d, these_dim)

    if input_dimensions_era5_constants is not None:
        error_checking.assert_is_numpy_array(
            input_dimensions_era5_constants,
            exact_dimensions=numpy.array([3], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            input_dimensions_era5_constants
        )
        error_checking.assert_is_greater_numpy_array(
            input_dimensions_era5_constants, 0
        )

        these_dim = numpy.array([
            num_grid_rows, num_grid_columns, input_dimensions_era5_constants[2]
        ], dtype=int)

        assert numpy.array_equal(input_dimensions_era5_constants, these_dim)

    error_checking.assert_is_numpy_array(
        input_dimensions_lagged_target,
        exact_dimensions=numpy.array([4], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(input_dimensions_lagged_target)
    error_checking.assert_is_greater_numpy_array(
        input_dimensions_lagged_target, 0
    )

    these_dim = numpy.array([
        num_grid_rows, num_grid_columns, input_dimensions_lagged_target[2], 1
    ], dtype=int)

    assert numpy.array_equal(input_dimensions_lagged_target, these_dim)

    num_conv_layers_in_fc_module = option_dict[NUM_FC_CONV_LAYERS_KEY]
    error_checking.assert_is_integer(num_conv_layers_in_fc_module)
    error_checking.assert_is_greater(num_conv_layers_in_fc_module, 0)

    expected_dim = numpy.array([num_conv_layers_in_fc_module], dtype=int)

    fc_module_dropout_rates = option_dict[FC_MODULE_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        fc_module_dropout_rates, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        fc_module_dropout_rates, 1., allow_nan=True
    )

    error_checking.assert_is_boolean(option_dict[USE_3D_CONV_IN_FC_KEY])

    num_levels = option_dict[NUM_LEVELS_KEY]
    error_checking.assert_is_integer(num_levels)
    error_checking.assert_is_geq(num_levels, 2)

    expected_dim = numpy.array([num_levels + 1], dtype=int)

    num_conv_layers_by_level = option_dict[CONV_LAYER_COUNTS_KEY]
    error_checking.assert_is_numpy_array(
        num_conv_layers_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(num_conv_layers_by_level)
    error_checking.assert_is_greater_numpy_array(num_conv_layers_by_level, 0)

    num_channels_by_level = option_dict[CHANNEL_COUNTS_KEY]
    error_checking.assert_is_numpy_array(
        num_channels_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(num_channels_by_level)
    error_checking.assert_is_greater_numpy_array(num_channels_by_level, 0)

    encoder_dropout_rate_by_level = option_dict[ENCODER_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        encoder_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        encoder_dropout_rate_by_level, 1., allow_nan=True
    )

    expected_dim = numpy.array([num_levels], dtype=int)

    decoder_dropout_rate_by_level = option_dict[DECODER_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        decoder_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        decoder_dropout_rate_by_level, 1., allow_nan=True
    )

    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        skip_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        skip_dropout_rate_by_level, 1., allow_nan=True
    )

    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])

    return option_dict


def _get_time_slicing_function(time_index):
    """Returns function that takes one time step from input tensor.

    :param time_index: Will take the [k]th time step, where k = `time_index`.
    :return: time_slicing_function: Function handle (see below).
    """

    def time_slicing_function(input_tensor_3d):
        """Takes one time step from the input tensor.

        :param input_tensor_3d: Input tensor with 3 spatiotemporal dimensions.
        :return: input_tensor_2d: Input tensor with 2 spatial dimensions.
        """

        return input_tensor_3d[..., time_index, :]

    return time_slicing_function


def create_model(option_dict, loss_function):
    """Creates Chiu net.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    Architecture based on: https://doi.org/10.1109/LRA.2020.2992184

    M = number of rows in grid
    N = number of columns in grid

    :param option_dict: See doc for `_check_args`.
    :param loss_function: Loss function.
    :return: model_object: Instance of `keras.models.Model`, with the
        aforementioned architecture.
    """

    option_dict = _check_args(option_dict)

    input_dimensions_gfs_3d = option_dict[GFS_3D_DIMENSIONS_KEY]
    input_dimensions_gfs_2d = option_dict[GFS_2D_DIMENSIONS_KEY]
    input_dimensions_era5 = option_dict[ERA5_CONST_DIMENSIONS_KEY]
    input_dimensions_lagged_target = option_dict[LAGGED_TARGET_DIMENSIONS_KEY]
    num_conv_layers_in_fc_module = option_dict[NUM_FC_CONV_LAYERS_KEY]
    fc_module_dropout_rates = option_dict[FC_MODULE_DROPOUT_RATES_KEY]
    use_3d_conv_in_fc_module = option_dict[USE_3D_CONV_IN_FC_KEY]
    num_levels = option_dict[NUM_LEVELS_KEY]
    num_conv_layers_by_level = option_dict[CONV_LAYER_COUNTS_KEY]
    num_channels_by_level = option_dict[CHANNEL_COUNTS_KEY]
    encoder_dropout_rate_by_level = option_dict[ENCODER_DROPOUT_RATES_KEY]
    decoder_dropout_rate_by_level = option_dict[DECODER_DROPOUT_RATES_KEY]
    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = option_dict[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = option_dict[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]

    if input_dimensions_gfs_3d is None:
        input_layer_object_gfs_3d = None
        layer_object_gfs_3d = None
    else:
        input_layer_object_gfs_3d = keras.layers.Input(
            shape=tuple(input_dimensions_gfs_3d.tolist())
        )
        layer_object_gfs_3d = keras.layers.Permute(
            dims=(4, 1, 2, 3, 5)
        )(input_layer_object_gfs_3d)

        new_dims = (
            input_dimensions_gfs_3d[3],
            input_dimensions_gfs_3d[0],
            input_dimensions_gfs_3d[1],
            input_dimensions_gfs_3d[2] * input_dimensions_gfs_3d[4]
        )
        layer_object_gfs_3d = keras.layers.Reshape(
            target_shape=new_dims
        )(layer_object_gfs_3d)

        num_gfs_lead_times = input_dimensions_gfs_3d[-2]

    if input_dimensions_gfs_2d is None:
        input_layer_object_gfs_2d = None
        layer_object_gfs_2d = None
    else:
        input_layer_object_gfs_2d = keras.layers.Input(
            shape=tuple(input_dimensions_gfs_2d.tolist())
        )
        layer_object_gfs_2d = keras.layers.Permute(
            dims=(3, 1, 2, 4)
        )(input_layer_object_gfs_2d)

        num_gfs_lead_times = input_dimensions_gfs_2d[-2]

    if input_dimensions_gfs_3d is None:
        layer_object_gfs = layer_object_gfs_2d
    elif input_dimensions_gfs_2d is None:
        layer_object_gfs = layer_object_gfs_3d
    else:
        layer_object_gfs = keras.layers.Concatenate(axis=-1)(
            [layer_object_gfs_3d, layer_object_gfs_2d]
        )

    if input_dimensions_era5 is None:
        input_layer_object_era5 = None
    else:
        input_layer_object_era5 = keras.layers.Input(
            shape=tuple(input_dimensions_era5.tolist())
        )

    input_layer_object_lagged_target = keras.layers.Input(
        shape=tuple(input_dimensions_lagged_target.tolist())
    )
    layer_object_lagged_target = keras.layers.Permute(
        dims=(3, 1, 2, 4)
    )(input_layer_object_lagged_target)

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    encoder_conv_layer_objects = [None] * (num_levels + 1)
    encoder_pooling_layer_objects = [None] * num_levels

    for i in range(num_levels + 1):
        for j in range(num_conv_layers_by_level[i]):
            if j == 0:
                if i == 0:
                    this_input_layer_object = layer_object_gfs
                else:
                    this_input_layer_object = (
                        encoder_pooling_layer_objects[i - 1]
                    )
            else:
                this_input_layer_object = encoder_conv_layer_objects[i]

            this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )
            encoder_conv_layer_objects[i] = keras.layers.TimeDistributed(
                this_conv_layer_object
            )(this_input_layer_object)

            encoder_conv_layer_objects[i] = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha
                )(encoder_conv_layer_objects[i])
            )

            if encoder_dropout_rate_by_level[i] > 0:
                encoder_conv_layer_objects[i] = (
                    architecture_utils.get_dropout_layer(
                        dropout_fraction=encoder_dropout_rate_by_level[i]
                    )(encoder_conv_layer_objects[i])
                )

            if use_batch_normalization:
                encoder_conv_layer_objects[i] = (
                    architecture_utils.get_batch_norm_layer()(
                        encoder_conv_layer_objects[i]
                    )
                )

        if i == num_levels:
            break

        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_rows_per_stride=2, num_columns_per_stride=2,
            pooling_type_string=architecture_utils.MAX_POOLING_STRING
        )
        encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
            this_pooling_layer_object
        )(encoder_conv_layer_objects[i])

    fc_module_layer_object = keras.layers.Permute(
        dims=(2, 3, 1, 4)
    )(encoder_conv_layer_objects[-1])

    if not use_3d_conv_in_fc_module:
        orig_dims = encoder_conv_layer_objects[-1].get_shape()
        new_dims = orig_dims[1:-2] + [orig_dims[-2] * orig_dims[-1]]

        fc_module_layer_object = keras.layers.Reshape(
            target_shape=new_dims, name='fc_module_remove-time-dim'
        )(encoder_conv_layer_objects[-1])

    for j in range(num_conv_layers_in_fc_module):
        this_name = 'fc_module_conv{0:d}'.format(j)

        if use_3d_conv_in_fc_module:
            if j == 0:
                fc_module_layer_object = architecture_utils.get_3d_conv_layer(
                    num_kernel_rows=1, num_kernel_columns=1,
                    num_kernel_heights=num_gfs_lead_times,
                    num_rows_per_stride=1, num_columns_per_stride=1,
                    num_heights_per_stride=1,
                    num_filters=num_channels_by_level[-1],
                    padding_type_string=architecture_utils.NO_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )(fc_module_layer_object)

                new_dims = (
                    fc_module_layer_object.shape[1:3] +
                    [fc_module_layer_object.shape[-1]]
                )
                fc_module_layer_object = keras.layers.Reshape(
                    target_shape=new_dims,
                    name='fc_module_remove-time-dim'
                )(fc_module_layer_object)
            else:
                fc_module_layer_object = architecture_utils.get_2d_conv_layer(
                    num_kernel_rows=3, num_kernel_columns=3,
                    num_rows_per_stride=1, num_columns_per_stride=1,
                    num_filters=num_channels_by_level[-1],
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )(fc_module_layer_object)
        else:
            fc_module_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[-1],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )(fc_module_layer_object)

        this_name = 'fc_module_conv{0:d}_activation'.format(j)

        fc_module_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name=this_name
        )(fc_module_layer_object)

        if fc_module_dropout_rates[j] > 0:
            this_name = 'fc_module_conv{0:d}_dropout'.format(j)

            fc_module_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=fc_module_dropout_rates[j],
                layer_name=this_name
            )(fc_module_layer_object)

        if use_batch_normalization:
            this_name = 'fc_module_conv{0:d}_bn'.format(j)

            fc_module_layer_object = architecture_utils.get_batch_norm_layer(
                layer_name=this_name
            )(fc_module_layer_object)

    upconv_layer_by_level = [None] * num_levels
    skip_layer_by_level = [None] * num_levels
    merged_layer_by_level = [None] * num_levels

    this_name = 'upsampling_level{0:d}'.format(num_levels - 1)

    try:
        this_layer_object = keras.layers.UpSampling2D(
            size=(2, 2), interpolation='bilinear', name=this_name
        )(fc_module_layer_object)
    except:
        this_layer_object = keras.layers.UpSampling2D(
            size=(2, 2), name=this_name
        )(fc_module_layer_object)

    this_name = 'upsampling_level{0:d}_conv'.format(num_levels - 1)
    i = num_levels - 1

    upconv_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=2, num_kernel_columns=2,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_filters=num_channels_by_level[i],
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object,
        layer_name=this_name
    )(this_layer_object)

    this_name = 'upsampling_level{0:d}_activation'.format(num_levels - 1)

    upconv_layer_by_level[i] = architecture_utils.get_activation_layer(
        activation_function_string=inner_activ_function_name,
        alpha_for_relu=inner_activ_function_alpha,
        alpha_for_elu=inner_activ_function_alpha,
        layer_name=this_name
    )(upconv_layer_by_level[i])

    if decoder_dropout_rate_by_level[i] > 0:
        this_name = 'upsampling_level{0:d}_dropout'.format(i)

        upconv_layer_by_level[i] = architecture_utils.get_dropout_layer(
            dropout_fraction=decoder_dropout_rate_by_level[i],
            layer_name=this_name
        )(upconv_layer_by_level[i])

    num_upconv_rows = upconv_layer_by_level[i].get_shape()[1]
    num_desired_rows = encoder_conv_layer_objects[i].get_shape()[1]
    num_padding_rows = num_desired_rows - num_upconv_rows

    num_upconv_columns = upconv_layer_by_level[i].get_shape()[2]
    num_desired_columns = encoder_conv_layer_objects[i].get_shape()[2]
    num_padding_columns = num_desired_columns - num_upconv_columns

    if num_padding_rows + num_padding_columns > 0:
        padding_arg = ((0, num_padding_rows), (0, num_padding_columns))
        this_name = 'padding_level{0:d}'.format(i)

        upconv_layer_by_level[i] = keras.layers.ZeroPadding2D(
            padding=padding_arg, name=this_name
        )(upconv_layer_by_level[i])

    this_function = _get_time_slicing_function(num_gfs_lead_times - 1)
    this_layer_object = keras.layers.Lambda(
        this_function
    )(encoder_conv_layer_objects[i])

    this_name = 'skip_level{0:d}'.format(i)
    merged_layer_by_level[i] = keras.layers.Concatenate(
        axis=-1, name=this_name
    )([this_layer_object, upconv_layer_by_level[i]])

    level_indices = numpy.linspace(
        0, num_levels - 1, num=num_levels, dtype=int
    )[::-1]

    for i in level_indices:
        for j in range(num_conv_layers_by_level[i]):
            if j == 0:
                this_input_layer_object = merged_layer_by_level[i]
            else:
                this_input_layer_object = skip_layer_by_level[i]

            this_name = 'skip_level{0:d}_conv{1:d}'.format(i, j)
            skip_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )(this_input_layer_object)

            this_name = 'skip_level{0:d}_conv{1:d}_activation'.format(i, j)
            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(skip_layer_by_level[i])

            if skip_dropout_rate_by_level[i] > 0:
                this_name = 'skip_level{0:d}_conv{1:d}_dropout'.format(i, j)
                skip_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=skip_dropout_rate_by_level[i],
                    layer_name=this_name
                )(skip_layer_by_level[i])

            if use_batch_normalization:
                this_name = 'skip_level{0:d}_conv{1:d}_bn'.format(i, j)
                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(skip_layer_by_level[i])
                )

        # TODO(thunderhoser): Allow penultimate conv to be skipped.
        if i == 0:
            skip_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1, num_filters=2,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name='penultimate_conv'
            )(skip_layer_by_level[i])

            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name='penultimate_conv_activation'
            )(skip_layer_by_level[i])

            if use_batch_normalization:
                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name='penultimate_conv_bn'
                    )(skip_layer_by_level[i])
                )

            break

        this_name = 'upsampling_level{0:d}'.format(i - 1)
        try:
            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2), interpolation='bilinear', name=this_name
            )(skip_layer_by_level[i])
        except:
            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2), name=this_name
            )(skip_layer_by_level[i])

        this_name = 'upsampling_level{0:d}_conv'.format(i - 1)
        upconv_layer_by_level[i - 1] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=2, num_kernel_columns=2,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=num_channels_by_level[i - 1],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )(this_layer_object)

        this_name = 'upsampling_level{0:d}_activation'.format(i - 1)
        upconv_layer_by_level[i - 1] = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name=this_name
        )(upconv_layer_by_level[i - 1])

        if decoder_dropout_rate_by_level[i - 1] > 0:
            this_name = 'upsampling_level{0:d}_dropout'.format(i - 1)
            upconv_layer_by_level[i - 1] = architecture_utils.get_dropout_layer(
                dropout_fraction=decoder_dropout_rate_by_level[i - 1],
                layer_name=this_name
            )(upconv_layer_by_level[i - 1])

        # TODO(thunderhoser): Modularize this fuckery.
        num_upconv_rows = upconv_layer_by_level[i - 1].get_shape()[1]
        num_desired_rows = encoder_conv_layer_objects[i - 1].get_shape()[1]
        num_padding_rows = num_desired_rows - num_upconv_rows

        num_upconv_columns = upconv_layer_by_level[i - 1].get_shape()[2]
        num_desired_columns = encoder_conv_layer_objects[i - 1].get_shape()[2]
        num_padding_columns = num_desired_columns - num_upconv_columns

        if num_padding_rows + num_padding_columns > 0:
            padding_arg = ((0, num_padding_rows), (0, num_padding_columns))
            this_name = 'padding_level{0:d}'.format(i - 1)

            upconv_layer_by_level[i - 1] = keras.layers.ZeroPadding2D(
                padding=padding_arg, name=this_name
            )(upconv_layer_by_level[i - 1])

        this_function = _get_time_slicing_function(num_gfs_lead_times - 1)
        this_layer_object = keras.layers.Lambda(
            this_function
        )(encoder_conv_layer_objects[i - 1])

        this_name = 'skip_level{0:d}'.format(i - 1)
        merged_layer_by_level[i - 1] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )([this_layer_object, upconv_layer_by_level[i - 1]])

    skip_layer_by_level[0] = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=1, num_kernel_columns=1,
        num_rows_per_stride=1, num_columns_per_stride=1, num_filters=1,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object,
        layer_name='last_conv'
    )(skip_layer_by_level[0])

    skip_layer_by_level[0] = architecture_utils.get_activation_layer(
        activation_function_string=output_activ_function_name,
        alpha_for_relu=output_activ_function_alpha,
        alpha_for_elu=output_activ_function_alpha,
        layer_name='last_conv_activation'
    )(skip_layer_by_level[0])

    input_layer_objects = [
        l for l in [
            input_layer_object_gfs_3d, input_layer_object_gfs_2d,
            input_layer_object_era5, input_layer_object_lagged_target
        ] if l is not None
    ]
    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=skip_layer_by_level[0]
    )

    # TODO(thunderhoser): Will probably want metrics.
    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(), metrics=[]
    )

    model_object.summary()
    return model_object
