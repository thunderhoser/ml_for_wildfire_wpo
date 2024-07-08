"""Methods for building a Chiu-net++ with flexible lead times.

'Flexible lead times' means that the same model can be predict at different lead
times -- but each prediction is for a single lead time.  This does not change
the architecture of the output layer, but it does change the architecture of the
input and encoder layers, for two reasons:

[1] To have a model that can predict at different lead times -- but with each
    prediction being at a single lead time -- the target lead time should be
    encoded somehow.  I have decided to make it an input variable (predictor).

[2] Changing the target lead time -- e.g., from 12 to 120 hours -- changes the
    predictors needed.  In particular, it may change the *number* of lead times
    needed in the NWP-based predictors, which changes the *dimensionality* of
    the NWP-based predictors.  Therefore, a model with flexible lead times needs
    flexible input dimensions -- specifically, a flexible time dimension for the
    NWP-based predictors.
"""

import numpy
import keras
import tensorflow
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
import chiu_net_pp_architecture as chiu_net_pp_arch

GFS_3D_DIMENSIONS_KEY = 'input_dimensions_gfs_3d'
GFS_2D_DIMENSIONS_KEY = 'input_dimensions_gfs_2d'
ERA5_CONST_DIMENSIONS_KEY = 'input_dimensions_era5_constants'
LAGTGT_DIMENSIONS_KEY = 'input_dimensions_lagged_target'
PREDN_BASELINE_DIMENSIONS_KEY = 'input_dimensions_predn_baseline'
USE_RESIDUAL_BLOCKS_KEY = 'use_residual_blocks'

GFS_FC_MODULE_NUM_LSTM_LAYERS_KEY = 'gfs_forecast_module_num_lstm_layers'
GFS_FC_MODULE_DROPOUT_RATES_KEY = 'gfs_forecast_module_dropout_rates'
LAGTGT_FC_MODULE_NUM_LSTM_LAYERS_KEY = (
    'lagged_target_forecast_module_num_lstm_layers'
)
LAGTGT_FC_MODULE_DROPOUT_RATES_KEY = (
    'lagged_target_forecast_module_dropout_rates'
)

NUM_LEVELS_KEY = 'num_levels'

GFS_ENCODER_NUM_CONV_LAYERS_KEY = 'gfs_encoder_num_conv_layers_by_level'
GFS_ENCODER_NUM_CHANNELS_KEY = 'gfs_encoder_num_channels_by_level'
GFS_ENCODER_DROPOUT_RATES_KEY = 'gfs_encoder_dropout_rate_by_level'
LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY = (
    'lagged_target_encoder_num_conv_layers_by_level'
)
LAGTGT_ENCODER_NUM_CHANNELS_KEY = 'lagged_target_encoder_num_channels_by_level'
LAGTGT_ENCODER_DROPOUT_RATES_KEY = 'lagged_target_encoder_dropout_rate_by_level'
DECODER_NUM_CONV_LAYERS_KEY = 'decoder_num_conv_layers_by_level'
DECODER_NUM_CHANNELS_KEY = 'decoder_num_channels_by_level'
UPSAMPLING_DROPOUT_RATES_KEY = 'upsampling_dropout_rate_by_level'
SKIP_DROPOUT_RATES_KEY = 'skip_dropout_rate_by_level'

INCLUDE_PENULTIMATE_KEY = 'include_penultimate_conv'
PENULTIMATE_DROPOUT_RATE_KEY = 'penultimate_conv_dropout_rate'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
OUTPUT_ACTIV_FUNCTION_KEY = 'output_activ_function_name'
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = 'output_activ_function_alpha'
L1_WEIGHT_KEY = 'l1_weight'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'
ENSEMBLE_SIZE_KEY = 'ensemble_size'

OPTIMIZER_FUNCTION_KEY = 'optimizer_function'

DEFAULT_ARCHITECTURE_OPTION_DICT = {
    GFS_FC_MODULE_NUM_LSTM_LAYERS_KEY: 1,
    GFS_FC_MODULE_DROPOUT_RATES_KEY: numpy.full(1, 0.),
    LAGTGT_FC_MODULE_NUM_LSTM_LAYERS_KEY: 1,
    LAGTGT_FC_MODULE_DROPOUT_RATES_KEY: numpy.full(1, 0.),
    NUM_LEVELS_KEY: 4,
    GFS_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(5, 2, dtype=int),
    GFS_ENCODER_NUM_CHANNELS_KEY: numpy.array([16, 24, 32, 64, 128], dtype=int),
    GFS_ENCODER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(5, 2, dtype=int),
    LAGTGT_ENCODER_NUM_CHANNELS_KEY: numpy.array([4, 6, 8, 12, 16], dtype=int),
    LAGTGT_ENCODER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    DECODER_NUM_CONV_LAYERS_KEY: numpy.full(5, 2, dtype=int),
    DECODER_NUM_CHANNELS_KEY: numpy.array([20, 30, 40, 76, 144], dtype=int),
    UPSAMPLING_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    SKIP_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    INCLUDE_PENULTIMATE_KEY: True,
    PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    L1_WEIGHT_KEY: 0.,
    L2_WEIGHT_KEY: 0.001,
    USE_BATCH_NORM_KEY: True
}


def __repeat_tensor_get_output_shape(input_shapes):
    """Computes output shape for a Lambda layer that calls __repeat_tensor.

    :param input_shapes: length-2 list, where the [k]th item is the shape, in
        tuple form, of the [k]th input to __repeat_tensor.
    :return: output_shape: Output shape in tuple form.
    """

    first_shape, second_shape = input_shapes
    return (
        first_shape[0], second_shape[1],
        first_shape[2], first_shape[3], first_shape[4]
    )


def _get_num_time_steps_time_first(input_tensor):
    """Returns number of time steps in tensor.

    Time must be the first non-batch dimension.

    :param input_tensor: Keras tensor.
    :return: num_time_steps: Keras tensor with shape of (1,).
    """

    return tensorflow.shape(input_tensor)[1]


def _repeat_tensor_along_time_axis(input_tensor_time_first,
                                   comparison_tensor_time_first):
    """Repeats input tensor along time axis.

    Time must be the first non-batch dimension in each tensor.

    :param input_tensor_time_first: Keras tensor to be repeated.
    :param comparison_tensor_time_first: Comparison tensor.  The output tensor
        will have the same number of time steps as the comparison tensor.
    :return: output_tensor_time_first: Same as input tensor but with more time
        steps.
    """

    num_times = _get_num_time_steps_time_first(comparison_tensor_time_first)[0]

    return tensorflow.tile(
        input_tensor_time_first,
        [1, num_times, 1, 1]
    )


def _get_lstm_layer(
        main_activ_function_name, main_activ_function_alpha,
        recurrent_activ_function_name, recurrent_activ_function_alpha,
        num_filters, regularizer_object, layer_name, return_sequences):
    """Returns LSTM layer with the desired hyperparameters.

    :param main_activ_function_name: See documentation for `_get_lstm_block`.
    :param main_activ_function_alpha: Same.
    :param recurrent_activ_function_name: Same.
    :param recurrent_activ_function_alpha: Same.
    :param num_filters: Same.
    :param regularizer_object: Same.
    :param layer_name: Layer name.
    :param return_sequences: Boolean flag.  If True (False), layer will (not)
        include time dimension.
    :return: lstm_layer_object: Instance of `keras.layers.LSTM`.
    """

    main_activ_function = architecture_utils.get_activation_layer(
        activation_function_string=main_activ_function_name,
        alpha_for_elu=main_activ_function_alpha,
        alpha_for_relu=main_activ_function_alpha
    )
    recurrent_activ_function = architecture_utils.get_activation_layer(
        activation_function_string=recurrent_activ_function_name,
        alpha_for_elu=recurrent_activ_function_alpha,
        alpha_for_relu=recurrent_activ_function_alpha
    )

    return keras.layers.ConvLSTM2D(
        filters=num_filters,
        kernel_size=1,
        strides=1,
        dilation_rate=1,
        padding='valid',
        data_format='channels_last',
        activation=main_activ_function,
        recurrent_activation=recurrent_activ_function,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        unit_forget_bias=True,
        kernel_regularizer=regularizer_object,
        recurrent_regularizer=regularizer_object,
        bias_regularizer=regularizer_object,
        activity_regularizer=None,
        return_sequences=return_sequences,
        name=layer_name
    )


def _get_lstm_block(
        input_layer_object, num_lstm_layers, num_filters, regularizer_object,
        main_activ_function_name, main_activ_function_alpha,
        recurrent_activ_function_name, recurrent_activ_function_alpha,
        dropout_rates, use_batch_norm, basic_layer_name):
    """Creates LSTM block.

    :param input_layer_object: Input layer to block (with 2 spatial dims and 1
        time dim).
    :param num_lstm_layers: Number of LSTM layers in block.
    :param num_filters: Number of filters (same for every LSTM layer).
    :param regularizer_object: See documentation for `_get_2d_conv_block`.
    :param main_activ_function_name: Name of main activation function -- same
        for every LSTM layer.  Must be accepted by
        `architecture_utils.check_activation_function`.
    :param main_activ_function_alpha: Alpha (slope parameter) for main
        activation function -- same for every LSTM layer.  Applies only to ReLU
        and eLU.
    :param recurrent_activ_function_name: Name of activation function for
        recurrent step -- same for every LSTM layer.  Must be accepted by
        `architecture_utils.check_activation_function`.
    :param recurrent_activ_function_alpha: Alpha (slope parameter) for
        activation function for recurrent step -- same for every LSTM layer.
        Applies only to ReLU and eLU.
    :param dropout_rates: See documentation for `_get_2d_conv_block`.
    :param use_batch_norm: Same.
    :param basic_layer_name: Same.
    :return: output_layer_object: Output layer from block (with 2 spatial dims
        and 0 time dims).
    """

    # Process input args.
    try:
        _ = len(dropout_rates)
    except:
        dropout_rates = numpy.full(num_lstm_layers, dropout_rates)

    if len(dropout_rates) < num_lstm_layers:
        dropout_rates = numpy.concatenate([
            dropout_rates, dropout_rates[[-1]]
        ])

    assert len(dropout_rates) == num_lstm_layers

    # Do actual stuff.
    current_layer_object = None

    for i in range(num_lstm_layers):
        if current_layer_object is None:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = current_layer_object

        this_name = '{0:s}_lstm{1:d}'.format(basic_layer_name, i)
        current_layer_object = _get_lstm_layer(
            main_activ_function_name=main_activ_function_name,
            main_activ_function_alpha=main_activ_function_alpha,
            recurrent_activ_function_name=recurrent_activ_function_name,
            recurrent_activ_function_alpha=recurrent_activ_function_alpha,
            num_filters=num_filters,
            regularizer_object=regularizer_object,
            layer_name=this_name,
            return_sequences=i < num_lstm_layers - 1
        )(this_input_layer_object)

        if dropout_rates[i] > 0:
            this_name = '{0:s}_dropout{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rates[i], layer_name=this_name
            )(current_layer_object)

        if use_batch_norm:
            this_name = '{0:s}_bn{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_batch_norm_layer(
                layer_name=this_name
            )(current_layer_object)

    return current_layer_object


def _check_args(option_dict):
    """Error-checks input arguments.

    L = number of levels in encoder = number of levels in decoder

    :param option_dict: Dictionary with the following keys.
    option_dict["input_dimensions_gfs_3d"]: numpy array with input dimensions
        for 3-D GFS data.  Array elements should be [num_rows, num_columns,
        num_pressure_levels, num_lead_times, num_fields].  If predictors do not
        include 3-D GFS data, make this None.
    option_dict["input_dimensions_gfs_2d"]: numpy array with input dimensions
        for 2-D GFS data.  Array elements should be [num_rows, num_columns,
        num_lead_times, num_fields].  If predictors do not include 2-D GFS data,
        make this None.
    option_dict["input_dimensions_era5_constants"]: numpy array with input
        dimensions for ERA5 constants.  Array elements should be [num_rows,
        num_columns, num_fields].  If predictors do not include ERA5 constants,
        make this None.
    option_dict["input_dimensions_lagged_targets"]: numpy array with input
        dimensions for lagged targets.  Array elements should be [num_rows,
        num_columns, num_lag_times, num_target_fields].
    option_dict["input_dimensions_predn_baseline"] numpy array with input
        dimensions for residual baseline.  Array elements should be [num_rows,
        num_columns, num_target_fields].
    option_dict["use_residual_blocks"]: Boolean flag.  If True (False), will use
        residual (basic convolutional) blocks throughout the network.
    option_dict["gfs_forecast_module_num_lstm_layers"]: Number of LSTM layers in
        forecasting module applied to GFS data.
    option_dict["lagged_target_forecast_module_num_lstm_layers"]: Same as
        "gfs_forecast_module_num_lstm_layers" but for lagged targets.
    option_dict["gfs_forecast_module_dropout_rates"]: length-N numpy array of
        dropout rates in forecasting module applied to GFS data, where
        N = "gfs_forecast_module_num_lstm_layers".
    option_dict["lagged_target_forecast_module_dropout_rates"]: Same as
        "gfs_forecast_module_dropout_rates" but for lagged targets.
    option_dict["num_levels"]: L in the above definitions.
    option_dict["gfs_encoder_num_conv_layers_by_level"]: length-(L + 1) numpy
        array with number of conv layers for GFS data at each level.
    option_dict["lagged_target_encoder_num_conv_layers_by_level"]: Same as
        "gfs_encoder_num_conv_layers_by_level" but for lagged targets.
    option_dict["gfs_encoder_num_channels_by_level"]: length-(L + 1) numpy array
        with number of channels (feature maps) for GFS data at each level.
    option_dict["lagged_target_encoder_num_channels_by_level"]: Same as
        "gfs_encoder_num_channels_by_level" but for lagged targets.
    option_dict["gfs_encoder_dropout_rate_by_level"]: length-(L + 1) numpy array
        with conv-layer dropout rate for GFS data at each level.
    option_dict["lagged_target_encoder_dropout_rate_by_level"]: Same as
        "gfs_encoder_dropout_rate_by_level" but for lagged targets.
    option_dict["decoder_num_conv_layers_by_level"]: length-L numpy array with
        number of conv layers at each decoder level.
    option_dict["decoder_num_channels_by_level"]: length-L numpy array with
        number of channels (feature maps) at each decoder level.
    option_dict["upsampling_dropout_rate_by_level"]: length-L numpy array with
        dropout rate for post-upsampling conv layer at each decoder level.
    option_dict["skip_dropout_rate_by_level"]: length-L numpy array with dropout
        rate for post-skip-connection conv layer at each decoder level.
    option_dict["include_penultimate_conv"]: Boolean flag.  If True, will put in
        extra conv layer (with 3 x 3 filter) before final pixelwise conv.
    option_dict["penultimate_conv_dropout_rate"]: Dropout rate for penultimate
        conv layer.
    option_dict["inner_activ_function_name"]: Name of activation function for
        all inner (non-output) layers.  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict["inner_activ_function_alpha"]: Alpha (slope parameter) for
        activation function for all inner layers.  Applies only to ReLU and eLU.
    option_dict["output_activ_function_name"]: Same as
        `inner_activ_function_name` but for output layer.
    option_dict["output_activ_function_alpha"]: Same as
        `inner_activ_function_alpha` but for output layer.
    option_dict["l1_weight"]: Weight for L_1 regularization.
    option_dict["l2_weight"]: Weight for L_2 regularization.
    option_dict["use_batch_normalization"]: Boolean flag.  If True, will use
        batch normalization after each inner (non-output) conv layer.
    option_dict["ensemble_size"]: Ensemble size.
    option_dict["optimizer_function"]: Optimizer function.

    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_ARCHITECTURE_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    input_dimensions_gfs_3d = option_dict[GFS_3D_DIMENSIONS_KEY]
    input_dimensions_gfs_2d = option_dict[GFS_2D_DIMENSIONS_KEY]
    input_dimensions_era5_constants = option_dict[ERA5_CONST_DIMENSIONS_KEY]
    input_dimensions_lagged_target = option_dict[LAGTGT_DIMENSIONS_KEY]
    input_dimensions_predn_baseline = option_dict[PREDN_BASELINE_DIMENSIONS_KEY]

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
        # error_checking.assert_is_greater_numpy_array(input_dimensions_gfs_3d, 0)

        num_grid_rows = input_dimensions_gfs_3d[0]
        num_grid_columns = input_dimensions_gfs_3d[1]
        num_gfs_lead_times = input_dimensions_gfs_3d[3]

    if input_dimensions_gfs_2d is not None:
        error_checking.assert_is_numpy_array(
            input_dimensions_gfs_2d,
            exact_dimensions=numpy.array([4], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(input_dimensions_gfs_2d)
        # error_checking.assert_is_greater_numpy_array(input_dimensions_gfs_2d, 0)

        if input_dimensions_gfs_3d is not None:
            these_dim = numpy.array([
                num_grid_rows, num_grid_columns, num_gfs_lead_times,
                input_dimensions_gfs_2d[3]
            ], dtype=int)

            assert numpy.array_equal(input_dimensions_gfs_2d, these_dim)

        num_grid_rows = input_dimensions_gfs_2d[0]
        num_grid_columns = input_dimensions_gfs_2d[1]

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

    these_dim = numpy.array([
        num_grid_rows, num_grid_columns,
        input_dimensions_lagged_target[2], input_dimensions_lagged_target[3]
    ], dtype=int)

    assert numpy.array_equal(input_dimensions_lagged_target, these_dim)
    error_checking.assert_is_integer_numpy_array(input_dimensions_lagged_target)
    # error_checking.assert_is_greater_numpy_array(
    #     input_dimensions_lagged_target, 0
    # )

    if input_dimensions_predn_baseline is not None:
        these_dim = numpy.array([
            num_grid_rows, num_grid_columns, input_dimensions_lagged_target[3]
        ], dtype=int)

        assert numpy.array_equal(input_dimensions_predn_baseline, these_dim)
        error_checking.assert_is_integer_numpy_array(
            input_dimensions_predn_baseline
        )
        error_checking.assert_is_greater_numpy_array(
            input_dimensions_predn_baseline, 0
        )

    error_checking.assert_is_boolean(option_dict[USE_RESIDUAL_BLOCKS_KEY])

    gfs_fcst_num_lstm_layers = option_dict[GFS_FC_MODULE_NUM_LSTM_LAYERS_KEY]
    error_checking.assert_is_integer(gfs_fcst_num_lstm_layers)
    error_checking.assert_is_greater(gfs_fcst_num_lstm_layers, 0)

    expected_dim = numpy.array([gfs_fcst_num_lstm_layers], dtype=int)

    gfs_fcst_dropout_rates = option_dict[GFS_FC_MODULE_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        gfs_fcst_dropout_rates, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        gfs_fcst_dropout_rates, 1., allow_nan=True
    )

    lagtgt_fcst_num_lstm_layers = option_dict[
        LAGTGT_FC_MODULE_NUM_LSTM_LAYERS_KEY
    ]
    error_checking.assert_is_integer(lagtgt_fcst_num_lstm_layers)
    error_checking.assert_is_greater(lagtgt_fcst_num_lstm_layers, 0)

    expected_dim = numpy.array([lagtgt_fcst_num_lstm_layers], dtype=int)

    lagtgt_fcst_dropout_rates = option_dict[LAGTGT_FC_MODULE_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        lagtgt_fcst_dropout_rates, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        lagtgt_fcst_dropout_rates, 1., allow_nan=True
    )

    num_levels = option_dict[NUM_LEVELS_KEY]
    error_checking.assert_is_integer(num_levels)
    error_checking.assert_is_geq(num_levels, 2)
    expected_dim = numpy.array([num_levels + 1], dtype=int)

    gfs_num_conv_by_level = option_dict[GFS_ENCODER_NUM_CONV_LAYERS_KEY]
    error_checking.assert_is_numpy_array(
        gfs_num_conv_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(gfs_num_conv_by_level)
    error_checking.assert_is_greater_numpy_array(gfs_num_conv_by_level, 0)

    gfs_num_channels_by_level = option_dict[GFS_ENCODER_NUM_CHANNELS_KEY]
    error_checking.assert_is_numpy_array(
        gfs_num_channels_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(gfs_num_channels_by_level)
    error_checking.assert_is_greater_numpy_array(gfs_num_channels_by_level, 0)

    gfs_dropout_rate_by_level = option_dict[GFS_ENCODER_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        gfs_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        gfs_dropout_rate_by_level, 1., allow_nan=True
    )

    lagtgt_num_conv_by_level = option_dict[LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY]
    error_checking.assert_is_numpy_array(
        lagtgt_num_conv_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(lagtgt_num_conv_by_level)
    error_checking.assert_is_greater_numpy_array(lagtgt_num_conv_by_level, 0)

    lagtgt_num_channels_by_level = option_dict[LAGTGT_ENCODER_NUM_CHANNELS_KEY]
    error_checking.assert_is_numpy_array(
        lagtgt_num_channels_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(lagtgt_num_channels_by_level)
    error_checking.assert_is_greater_numpy_array(
        lagtgt_num_channels_by_level, 0
    )

    lagtgt_dropout_rate_by_level = option_dict[LAGTGT_ENCODER_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        lagtgt_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        lagtgt_dropout_rate_by_level, 1., allow_nan=True
    )

    expected_dim = numpy.array([num_levels], dtype=int)

    decoder_num_conv_by_level = option_dict[DECODER_NUM_CONV_LAYERS_KEY]
    error_checking.assert_is_numpy_array(
        decoder_num_conv_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(decoder_num_conv_by_level)
    error_checking.assert_is_greater_numpy_array(decoder_num_conv_by_level, 0)

    decoder_num_channels_by_level = option_dict[DECODER_NUM_CHANNELS_KEY]
    error_checking.assert_is_numpy_array(
        decoder_num_channels_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(decoder_num_channels_by_level)
    error_checking.assert_is_greater_numpy_array(
        decoder_num_channels_by_level, 0
    )

    upsampling_dropout_rate_by_level = option_dict[UPSAMPLING_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        upsampling_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        upsampling_dropout_rate_by_level, 1., allow_nan=True
    )

    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        skip_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        skip_dropout_rate_by_level, 1., allow_nan=True
    )

    error_checking.assert_is_boolean(option_dict[INCLUDE_PENULTIMATE_KEY])
    error_checking.assert_is_leq(
        option_dict[PENULTIMATE_DROPOUT_RATE_KEY], 1., allow_nan=True
    )

    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])
    error_checking.assert_is_integer(option_dict[ENSEMBLE_SIZE_KEY])
    error_checking.assert_is_greater(option_dict[ENSEMBLE_SIZE_KEY], 0)

    return option_dict


def create_model(option_dict, loss_function, metric_list):
    """Creates Chiu-net++ with flexible lead times.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    :param option_dict: See doc for `_check_args`.
    :param loss_function: Loss function.
    :param metric_list: 1-D list of metrics.
    :return: model_object: Instance of `keras.models.Model`, with the
        Chiu-net++ architecture.
    """

    option_dict = _check_args(option_dict)
    error_checking.assert_is_list(metric_list)

    input_dimensions_gfs_3d = option_dict[GFS_3D_DIMENSIONS_KEY]
    input_dimensions_gfs_2d = option_dict[GFS_2D_DIMENSIONS_KEY]
    input_dimensions_era5 = option_dict[ERA5_CONST_DIMENSIONS_KEY]
    input_dimensions_lagged_target = option_dict[LAGTGT_DIMENSIONS_KEY]
    input_dimensions_predn_baseline = option_dict[PREDN_BASELINE_DIMENSIONS_KEY]
    use_residual_blocks = option_dict[USE_RESIDUAL_BLOCKS_KEY]

    gfs_fcst_num_lstm_layers = option_dict[GFS_FC_MODULE_NUM_LSTM_LAYERS_KEY]
    gfs_fcst_dropout_rates = option_dict[GFS_FC_MODULE_DROPOUT_RATES_KEY]
    lagtgt_fcst_num_lstm_layers = option_dict[
        LAGTGT_FC_MODULE_NUM_LSTM_LAYERS_KEY
    ]
    lagtgt_fcst_dropout_rates = option_dict[LAGTGT_FC_MODULE_DROPOUT_RATES_KEY]

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

    if input_dimensions_gfs_3d is not None:
        input_dimensions_gfs_3d = tuple([
            d if d > 0 else None for d in input_dimensions_gfs_3d
        ])

    if input_dimensions_gfs_2d is not None:
        input_dimensions_gfs_2d = tuple([
            d if d > 0 else None for d in input_dimensions_gfs_2d
        ])

    if input_dimensions_era5 is not None:
        input_dimensions_era5 = tuple([
            d for d in input_dimensions_era5
        ])

    if input_dimensions_lagged_target is not None:
        input_dimensions_lagged_target = tuple([
            d if d > 0 else None for d in input_dimensions_lagged_target
        ])

    if input_dimensions_predn_baseline is not None:
        input_dimensions_predn_baseline = tuple([
            d for d in input_dimensions_predn_baseline
        ])

    if input_dimensions_gfs_3d is None:
        input_layer_object_gfs_3d = None
        layer_object_gfs_3d = None
    else:
        input_layer_object_gfs_3d = keras.layers.Input(
            shape=input_dimensions_gfs_3d, name='gfs_3d_inputs'
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

    if input_dimensions_gfs_2d is None:
        input_layer_object_gfs_2d = None
        layer_object_gfs_2d = None
    else:
        input_layer_object_gfs_2d = keras.layers.Input(
            shape=input_dimensions_gfs_2d, name='gfs_2d_inputs'
        )
        layer_object_gfs_2d = keras.layers.Permute(
            dims=(3, 1, 2, 4),
            name='gfs_2d_put-time-first'
        )(input_layer_object_gfs_2d)

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
        shape=input_dimensions_lagged_target, name='lagged_target_inputs'
    )
    layer_object_lagged_target = keras.layers.Permute(
        dims=(3, 1, 2, 4), name='lagged_targets_put-time-first'
    )(input_layer_object_lagged_target)

    if input_dimensions_predn_baseline is None:
        input_layer_object_predn_baseline = None
    else:
        input_layer_object_predn_baseline = keras.layers.Input(
            shape=input_dimensions_predn_baseline, name='predn_baseline_inputs'
        )

    input_layer_object_lead_time = keras.layers.Input(
        shape=(1,), name='lead_time'
    )
    num_grid_rows = input_dimensions_lagged_target[0]
    num_grid_columns = input_dimensions_lagged_target[1]

    layer_object_lead_time = keras.layers.Reshape(
        target_shape=(1, 1), name='lead_time_add-column-dim'
    )(input_layer_object_lead_time)

    layer_object_lead_time = keras.layers.Concatenate(
        axis=-2, name='lead_time_add-columns'
    )(
        num_grid_columns * [layer_object_lead_time]
    )

    layer_object_lead_time = keras.layers.Reshape(
        target_shape=(1, num_grid_columns, 1), name='lead_time_add-row-dim'
    )(layer_object_lead_time)

    layer_object_lead_time = keras.layers.Concatenate(
        axis=-3, name='lead_time_add-rows'
    )(
        num_grid_rows * [layer_object_lead_time]
    )

    num_target_fields = input_dimensions_lagged_target[-1]

    if input_dimensions_era5 is None:
        input_layer_object_era5 = None
        layer_object_constants = layer_object_lead_time
        new_dims = (1, num_grid_rows, num_grid_columns, 1)
    else:
        input_layer_object_era5 = keras.layers.Input(
            shape=input_dimensions_era5, name='era5_inputs'
        )

        layer_object_constants = keras.layers.Concatenate(
            axis=-1, name='concat_constants'
        )(
            [input_layer_object_era5, layer_object_lead_time]
        )

        new_dims = (
            1, num_grid_rows, num_grid_columns, 1 + input_dimensions_era5[-1]
        )

    layer_object_constants = keras.layers.Reshape(
        target_shape=new_dims, name='const_add-time-dim'
    )(layer_object_constants)

    this_layer_object = keras.layers.Lambda(
        lambda x: _repeat_tensor_along_time_axis(x[0], x[1]),
        name='const_add-gfs-times',
        output_shape=__repeat_tensor_get_output_shape
    )([layer_object_constants, layer_object_gfs])

    layer_object_gfs = keras.layers.Concatenate(
        axis=-1, name='gfs_concat-const'
    )(
        [layer_object_gfs, this_layer_object]
    )

    this_layer_object = keras.layers.Lambda(
        lambda x: _repeat_tensor_along_time_axis(x[0], x[1]),
        name='const_add-target-times',
        output_shape=__repeat_tensor_get_output_shape
    )([layer_object_constants, layer_object_lagged_target])

    layer_object_lagged_target = keras.layers.Concatenate(
        axis=-1, name='lagged_targets_concat-const'
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
        if i == 0:
            this_input_layer_object = layer_object_gfs
        else:
            this_input_layer_object = gfs_encoder_pooling_layer_objects[i - 1]

        gfs_encoder_conv_layer_objects[i] = chiu_net_pp_arch._get_2d_conv_block(
            input_layer_object=this_input_layer_object,
            do_residual=use_residual_blocks,
            num_conv_layers=gfs_encoder_num_conv_layers_by_level[i],
            filter_size_px=3,
            num_filters=gfs_encoder_num_channels_by_level[i],
            do_time_distributed_conv=True,
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=gfs_encoder_dropout_rate_by_level[i],
            use_batch_norm=use_batch_normalization,
            basic_layer_name='gfs_encoder_level{0:d}'.format(i)
        )

        # TODO(thunderhoser): Maybe I want to allow a different activation
        # function for LSTM?  Probably not, though.  I imagine there would not
        # be much benefit to using leaky ReLU everywhere in the network but
        # then, e.g., tanh for LSTM layers.
        gfs_fcst_module_layer_objects[i] = _get_lstm_block(
            input_layer_object=gfs_encoder_conv_layer_objects[i],
            num_lstm_layers=gfs_fcst_num_lstm_layers,
            num_filters=gfs_encoder_num_channels_by_level[i],
            regularizer_object=regularizer_object,
            main_activ_function_name=inner_activ_function_name,
            main_activ_function_alpha=inner_activ_function_alpha,
            recurrent_activ_function_name=
            architecture_utils.SIGMOID_FUNCTION_STRING,
            recurrent_activ_function_alpha=0.,
            dropout_rates=gfs_fcst_dropout_rates,
            use_batch_norm=use_batch_normalization,
            basic_layer_name='gfs_fcst_level{0:d}'.format(i)
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
        if i == 0:
            this_input_layer_object = layer_object_lagged_target
        else:
            this_input_layer_object = (
                lagtgt_encoder_pooling_layer_objects[i - 1]
            )

        lagtgt_encoder_conv_layer_objects[i] = (
            chiu_net_pp_arch._get_2d_conv_block(
                input_layer_object=this_input_layer_object,
                do_residual=use_residual_blocks,
                num_conv_layers=lagtgt_encoder_num_conv_layers_by_level[i],
                filter_size_px=3,
                num_filters=lagtgt_encoder_num_channels_by_level[i],
                do_time_distributed_conv=True,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=lagtgt_encoder_dropout_rate_by_level[i],
                use_batch_norm=use_batch_normalization,
                basic_layer_name='lagtgt_encoder_level{0:d}'.format(i)
            )
        )

        lagtgt_fcst_module_layer_objects[i] = _get_lstm_block(
            input_layer_object=lagtgt_encoder_conv_layer_objects[i],
            num_lstm_layers=lagtgt_fcst_num_lstm_layers,
            num_filters=lagtgt_encoder_num_channels_by_level[i],
            regularizer_object=regularizer_object,
            main_activ_function_name=inner_activ_function_name,
            main_activ_function_alpha=inner_activ_function_alpha,
            recurrent_activ_function_name=
            architecture_utils.SIGMOID_FUNCTION_STRING,
            recurrent_activ_function_alpha=0.,
            dropout_rates=lagtgt_fcst_dropout_rates,
            use_batch_norm=use_batch_normalization,
            basic_layer_name='lagtgt_fcst_level{0:d}'.format(i)
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

            this_layer_object = chiu_net_pp_arch._pad_2d_layer(
                source_layer_object=this_layer_object,
                target_layer_object=last_conv_layer_matrix[i_new, 0],
                padding_layer_name='block{0:d}-{1:d}_padding'.format(i_new, j)
            )

            this_num_channels = int(numpy.round(
                0.5 * decoder_num_channels_by_level[i_new]
            ))

            last_conv_layer_matrix[i_new, j] = (
                chiu_net_pp_arch._get_2d_conv_block(
                    input_layer_object=this_layer_object,
                    do_residual=use_residual_blocks,
                    num_conv_layers=1,
                    filter_size_px=3,
                    num_filters=this_num_channels,
                    do_time_distributed_conv=False,
                    regularizer_object=regularizer_object,
                    activation_function_name=inner_activ_function_name,
                    activation_function_alpha=inner_activ_function_alpha,
                    dropout_rates=upsampling_dropout_rate_by_level[i_new],
                    use_batch_norm=use_batch_normalization,
                    basic_layer_name='block{0:d}-{1:d}_up'.format(i_new, j)
                )
            )

            last_conv_layer_matrix[i_new, j] = (
                chiu_net_pp_arch._create_skip_connection(
                    input_layer_objects=
                    last_conv_layer_matrix[i_new, :(j + 1)].tolist(),
                    num_output_channels=decoder_num_channels_by_level[i_new],
                    current_level_num=i_new,
                    regularizer_object=regularizer_object
                )
            )

            last_conv_layer_matrix[i_new, j] = (
                chiu_net_pp_arch._get_2d_conv_block(
                    input_layer_object=last_conv_layer_matrix[i_new, j],
                    do_residual=use_residual_blocks,
                    num_conv_layers=decoder_num_conv_layers_by_level[i_new],
                    filter_size_px=3,
                    num_filters=decoder_num_channels_by_level[i_new],
                    do_time_distributed_conv=False,
                    regularizer_object=regularizer_object,
                    activation_function_name=inner_activ_function_name,
                    activation_function_alpha=inner_activ_function_alpha,
                    dropout_rates=skip_dropout_rate_by_level[i_new],
                    use_batch_norm=use_batch_normalization,
                    basic_layer_name='block{0:d}-{1:d}_skip'.format(i_new, j)
                )
            )

    if include_penultimate_conv:
        last_conv_layer_matrix[0, -1] = chiu_net_pp_arch._get_2d_conv_block(
            input_layer_object=last_conv_layer_matrix[0, -1],
            do_residual=use_residual_blocks,
            num_conv_layers=1,
            filter_size_px=3,
            num_filters=2 * num_target_fields * ensemble_size,
            do_time_distributed_conv=False,
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=penultimate_conv_dropout_rate,
            use_batch_norm=use_batch_normalization,
            basic_layer_name='penultimate'
        )

    output_layer_object = chiu_net_pp_arch._get_2d_conv_block(
        input_layer_object=last_conv_layer_matrix[0, -1],
        do_residual=use_residual_blocks,
        num_conv_layers=1,
        filter_size_px=1,
        num_filters=num_target_fields * ensemble_size,
        do_time_distributed_conv=False,
        regularizer_object=regularizer_object,
        activation_function_name=output_activ_function_name,
        activation_function_alpha=output_activ_function_alpha,
        dropout_rates=-1.,
        use_batch_norm=False,
        basic_layer_name='output'
    )

    if ensemble_size > 1:
        new_dims = (
            input_dimensions_lagged_target[0],
            input_dimensions_lagged_target[1],
            num_target_fields,
            ensemble_size
        )
        output_layer_object = keras.layers.Reshape(
            target_shape=new_dims, name='reshape_output'
        )(output_layer_object)

    if input_layer_object_predn_baseline is not None:
        if ensemble_size > 1:
            new_dims = (
                input_dimensions_predn_baseline[0],
                input_dimensions_predn_baseline[1],
                input_dimensions_predn_baseline[2],
                1
            )

            layer_object_predn_baseline = keras.layers.Reshape(
                target_shape=new_dims,
                name='reshape_predn_baseline'
            )(input_layer_object_predn_baseline)
        else:
            layer_object_predn_baseline = input_layer_object_predn_baseline

        output_layer_object = keras.layers.Add(name='output_add_baseline')([
            output_layer_object, layer_object_predn_baseline
        ])

    input_layer_objects = [
        l for l in [
            input_layer_object_gfs_3d, input_layer_object_gfs_2d,
            input_layer_object_lead_time, input_layer_object_era5,
            input_layer_object_lagged_target, input_layer_object_predn_baseline
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
