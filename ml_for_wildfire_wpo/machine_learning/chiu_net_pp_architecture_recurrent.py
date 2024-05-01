"""Methods for building a recurrent Chiu-net++.

The 'recurrent' part means that the network predicts multiple time steps.  For
predicting the (k + 1)th time step, the network's prediction at the (k)th time
step is used as an input, i.e., a predictor.
"""

import time
import numpy
import keras
import tensorflow
from tensorflow.keras.layers import Layer
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml_for_wildfire_wpo.machine_learning import chiu_net_architecture
from ml_for_wildfire_wpo.machine_learning import \
    chiu_net_pp_architecture as basic_arch

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


class EnsembleMeanLayer(Layer):
    def __init__(self, **kwargs):
        super(EnsembleMeanLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # No weights to be defined in this case
        pass

    def call(self, inputs):
        return tensorflow.reduce_mean(inputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        base_config = super(EnsembleMeanLayer, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _get_channel_counts_for_skip_cnxn(
        current_channel_counts, num_output_channels):
    """Determines number of channels for each input layer to skip connection.

    A = number of input layers

    :param current_channel_counts: length-A numpy array of channel counts.
    :param num_output_channels: Number of desired output channels (after
        concatenation).
    :return: desired_channel_counts: length-A numpy array with number of
        desired channels for each input layer.
    """

    num_input_layers = len(current_channel_counts)
    desired_channel_counts = numpy.full(num_input_layers, -1, dtype=int)

    half_num_output_channels = int(numpy.round(0.5 * num_output_channels))
    desired_channel_counts[-1] = half_num_output_channels

    remaining_num_output_channels = (
        num_output_channels - half_num_output_channels
    )
    this_ratio = (
        float(remaining_num_output_channels) /
        numpy.sum(current_channel_counts[:-1])
    )
    desired_channel_counts[:-1] = numpy.round(
        current_channel_counts[:-1] * this_ratio
    ).astype(int)

    while numpy.sum(desired_channel_counts) > num_output_channels:
        desired_channel_counts[numpy.argmax(desired_channel_counts[:-1])] -= 1
    while numpy.sum(desired_channel_counts) < num_output_channels:
        desired_channel_counts[numpy.argmin(desired_channel_counts[:-1])] += 1

    assert numpy.sum(desired_channel_counts) == num_output_channels
    desired_channel_counts = numpy.maximum(desired_channel_counts, 1)

    return desired_channel_counts


def _create_skip_connection(
        input_layer_names, input_channel_counts, num_output_channels,
        current_level_num, regularizer_object):
    """Creates skip connection.

    A = number of input layers

    :param input_layer_names: length-A list with names of input layers.
    :param input_channel_counts: length-A numpy array of channel counts.
    :param num_output_channels: Desired number of output channels.
    :param current_level_num: Current level in Chiu-net++ architecture.  This
        should be a zero-based integer index.
    :param regularizer_object: Regularizer for conv layers (instance of
        `keras.regularizers.l1_l2` or similar).
    :return: layer_names: 1-D list of layer names.
    :return: layer_name_to_input_layer_names: Dictionary, where each key is a
        string from layer_names and the corresponding value is a list of
        strings -- also from layer_names -- indicating the inputs to said layer.
    :return: layer_name_to_object: Dictionary, where each key is a string from
        layer_names and the corresponding value is the layer object.
    """

    desired_input_channel_counts = _get_channel_counts_for_skip_cnxn(
        current_channel_counts=input_channel_counts,
        num_output_channels=num_output_channels
    )
    current_width = len(input_layer_names) - 1

    layer_names = [''] * current_width
    layer_name_to_input_layer_names = dict()
    layer_name_to_object = dict()

    for j in range(current_width):
        layer_names[j] = 'block{0:d}-{1:d}_preskipconv{2:d}'.format(
            current_level_num, current_width, j
        )

        this_layer_object = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=1,
            num_kernel_columns=1,
            num_rows_per_stride=1,
            num_columns_per_stride=1,
            num_filters=desired_input_channel_counts[j],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=layer_names[j]
        )

        layer_name_to_input_layer_names[layer_names[j]] = input_layer_names[j]
        layer_name_to_object[layer_names[j]] = this_layer_object

    this_name = 'block{0:d}-{1:d}_skip'.format(current_level_num, current_width)
    concat_layer_object = keras.layers.Concatenate(axis=-1, name=this_name)

    layer_names.append(this_name)
    layer_name_to_input_layer_names[this_name] = layer_names[:-1] + [input_layer_names[-1]]
    layer_name_to_object[this_name] = concat_layer_object

    return layer_names, layer_name_to_input_layer_names, layer_name_to_object


def _get_2d_conv_block(
        input_layer_name, do_residual,
        num_conv_layers, filter_size_px, num_filters, do_time_distributed_conv,
        regularizer_object,
        activation_function_name, activation_function_alpha,
        dropout_rates, use_batch_norm, basic_layer_name):
    """Creates convolutional block for data with 2 spatial dimensions.

    L = number of conv layers

    :param input_layer_name: Name of input layer to block.
    :param do_residual: Boolean flag.  If True (False), this will be a residual
        (basic convolutional) block.
    :param num_conv_layers: Number of conv layers in block.
    :param filter_size_px: Filter size for conv layers.  The same filter size
        will be used in both dimensions, and the same filter size will be used
        for every conv layer.
    :param num_filters: Number of filters -- same for every conv layer.
    :param do_time_distributed_conv: Boolean flag.  If True (False), will do
        time-distributed (basic) convolution.
    :param regularizer_object: Regularizer for conv layers (instance of
        `keras.regularizers.l1_l2` or similar).
    :param activation_function_name: Name of activation function -- same for
        every conv layer.  Must be accepted by
        `architecture_utils.check_activation_function`.
    :param activation_function_alpha: Alpha (slope parameter) for activation
        function -- same for every conv layer.  Applies only to ReLU and eLU.
    :param dropout_rates: Dropout rates for conv layers.  This can be a scalar
        (applied to every conv layer) or length-L numpy array.
    :param use_batch_norm: Boolean flag.  If True, will use batch normalization.
    :param basic_layer_name: Basic layer name.  Each layer name will be made
        unique by adding a suffix.
    :return: layer_names: See documentation for `_create_skip_connection`.
    :return: layer_name_to_input_layer_names: Same.
    :return: layer_name_to_object: Same.
    """

    # Process input args.
    if do_residual:
        num_conv_layers = max([num_conv_layers, 2])

    try:
        _ = len(dropout_rates)
    except:
        dropout_rates = numpy.full(num_conv_layers, dropout_rates)

    if len(dropout_rates) < num_conv_layers:
        dropout_rates = numpy.concatenate([
            dropout_rates, dropout_rates[[-1]]
        ])

    assert len(dropout_rates) == num_conv_layers

    # Do actual stuff.
    layer_names = []
    layer_name_to_input_layer_names = dict()
    layer_name_to_object = dict()

    for i in range(num_conv_layers):
        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)

        # TODO(thunderhoser): Will assigning a different layer to the same
        # variable nane, every time thru the for-loop, fuck things up?
        conv_layer_object = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=filter_size_px,
            num_kernel_columns=filter_size_px,
            num_rows_per_stride=1,
            num_columns_per_stride=1,
            num_filters=num_filters,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        # TODO(thunderhoser): I don't know if this layer combo will work outside
        # of a model.
        if do_time_distributed_conv:
            conv_layer_object = keras.layers.TimeDistributed(
                conv_layer_object, name=this_name
            )

        if i == 0:
            layer_name_to_input_layer_names[this_name] = input_layer_name
        else:
            layer_name_to_input_layer_names[this_name] = layer_names[-1]

        layer_name_to_object[this_name] = conv_layer_object
        layer_names.append(this_name)

        if i == num_conv_layers - 1 and do_residual:
            this_name = '{0:s}_preresidual_conv'.format(basic_layer_name)
            new_conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=filter_size_px,
                num_kernel_columns=filter_size_px,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )

            if do_time_distributed_conv:
                new_conv_layer_object = keras.layers.TimeDistributed(
                    new_conv_layer_object, name=this_name
                )

            layer_name_to_input_layer_names[this_name] = input_layer_name
            layer_name_to_object[this_name] = new_conv_layer_object
            layer_names.append(this_name)

            this_name = '{0:s}_residual'.format(basic_layer_name)
            add_layer_object = keras.layers.Add(name=this_name)

            layer_name_to_input_layer_names[this_name] = [
                layer_names[-1], layer_names[-2]
            ]
            layer_name_to_object[this_name] = add_layer_object
            layer_names.append(this_name)

        if activation_function_name is not None:
            this_name = '{0:s}_activ{1:d}'.format(basic_layer_name, i)
            activation_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_name,
                alpha_for_relu=activation_function_alpha,
                alpha_for_elu=activation_function_alpha,
                layer_name=this_name
            )

            layer_name_to_input_layer_names[this_name] = layer_names[-1]
            layer_name_to_object[this_name] = activation_layer_object
            layer_names.append(this_name)

        if dropout_rates[i] > 0:
            this_name = '{0:s}_dropout{1:d}'.format(basic_layer_name, i)
            dropout_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rates[i], layer_name=this_name
            )

            layer_name_to_input_layer_names[this_name] = layer_names[-1]
            layer_name_to_object[this_name] = dropout_layer_object
            layer_names.append(this_name)

        if use_batch_norm:
            this_name = '{0:s}_bn{1:d}'.format(basic_layer_name, i)
            batch_norm_layer_object = architecture_utils.get_batch_norm_layer(
                layer_name=this_name
            )

            layer_name_to_input_layer_names[this_name] = layer_names[-1]
            layer_name_to_object[this_name] = batch_norm_layer_object
            layer_names.append(this_name)

    return layer_names, layer_name_to_input_layer_names, layer_name_to_object


def _get_3d_conv_block(
        input_layer_name, num_time_steps, num_filters,
        do_residual, num_conv_layers, filter_size_px,
        regularizer_object, activation_function_name, activation_function_alpha,
        dropout_rates, use_batch_norm, basic_layer_name):
    """Creates convolutional block for data with 3 spatial dimensions.

    :param input_layer_name: Name of input layer to block.
    :param num_time_steps: Number of time steps.
    :param num_filters: Number of filters.
    :param do_residual: See documentation for `_get_2d_conv_block`.
    :param num_conv_layers: Same.
    :param filter_size_px: Same.
    :param regularizer_object: Same.
    :param activation_function_name: Same.
    :param activation_function_alpha: Same.
    :param dropout_rates: Same.
    :param use_batch_norm: Same.
    :param basic_layer_name: Same.
    :return: layer_names: See documentation for `_create_skip_connection`.
    :return: layer_name_to_input_layer_names: Same.
    :return: layer_name_to_object: Same.
    """

    # Process input args.
    if do_residual:
        num_conv_layers = max([num_conv_layers, 2])

    try:
        _ = len(dropout_rates)
    except:
        dropout_rates = numpy.full(num_conv_layers, dropout_rates)

    if len(dropout_rates) < num_conv_layers:
        dropout_rates = numpy.concatenate([
            dropout_rates, dropout_rates[[-1]]
        ])

    assert len(dropout_rates) == num_conv_layers

    # Do actual stuff.
    layer_names = []
    layer_name_to_input_layer_names = dict()
    layer_name_to_object = dict()

    for i in range(num_conv_layers):
        conv_layer_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)

        if i == 0:
            conv_layer_object = architecture_utils.get_3d_conv_layer(
                num_kernel_rows=filter_size_px,
                num_kernel_columns=filter_size_px,
                num_kernel_heights=num_time_steps,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_heights_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.NO_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=conv_layer_name
            )

            layer_name_to_input_layer_names[conv_layer_name] = input_layer_name
            layer_name_to_object[conv_layer_name] = conv_layer_object
            layer_names.append(conv_layer_name)

            this_name = '{0:s}_remove-time-dim'.format(basic_layer_name)
            remove_time_layer_object = RemoveTimeDimLayer(name=this_name)

            layer_name_to_input_layer_names[this_name] = layer_names[-1]
            layer_name_to_object[this_name] = remove_time_layer_object
            layer_names.append(this_name)
        else:
            conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=filter_size_px,
                num_kernel_columns=filter_size_px,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )

            layer_name_to_input_layer_names[conv_layer_name] = layer_names[-1]
            layer_name_to_object[conv_layer_name] = conv_layer_object
            layer_names.append(conv_layer_name)

        if i == num_conv_layers - 1 and do_residual:
            this_name = '{0:s}_preresidual_avg'.format(basic_layer_name)
            pooling_layer_object = architecture_utils.get_3d_pooling_layer(
                num_rows_in_window=1,
                num_columns_in_window=1,
                num_heights_in_window=num_time_steps,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_heights_per_stride=num_time_steps,
                pooling_type_string=architecture_utils.MEAN_POOLING_STRING,
                layer_name=this_name
            )

            layer_name_to_input_layer_names[this_name] = input_layer_name
            layer_name_to_object[this_name] = pooling_layer_object
            layer_names.append(this_name)

            squeeze_layer_name = '{0:s}_preresidual_squeeze'.format(
                basic_layer_name
            )
            squeeze_layer_object = RemoveTimeDimLayer(name=squeeze_layer_name)

            layer_name_to_input_layer_names[squeeze_layer_name] = (
                layer_names[-1]
            )
            layer_name_to_object[squeeze_layer_name] = squeeze_layer_object
            layer_names.append(squeeze_layer_name)

            this_name = '{0:s}_residual'.format(basic_layer_name)
            add_layer_object = keras.layers.Add(name=this_name)

            layer_name_to_input_layer_names[this_name] = [
                conv_layer_name, squeeze_layer_name
            ]
            layer_name_to_object[this_name] = add_layer_object
            layer_names.append(this_name)

        this_name = '{0:s}_activ{1:d}'.format(basic_layer_name, i)
        activation_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=activation_function_name,
            alpha_for_relu=activation_function_alpha,
            alpha_for_elu=activation_function_alpha,
            layer_name=this_name
        )

        layer_name_to_input_layer_names[this_name] = layer_names[-1]
        layer_name_to_object[this_name] = activation_layer_object
        layer_names.append(this_name)

        if dropout_rates[i] > 0:
            this_name = '{0:s}_dropout{1:d}'.format(basic_layer_name, i)
            dropout_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rates[i], layer_name=this_name
            )

            layer_name_to_input_layer_names[this_name] = layer_names[-1]
            layer_name_to_object[this_name] = dropout_layer_object
            layer_names.append(this_name)

        if use_batch_norm:
            this_name = '{0:s}_bn{1:d}'.format(basic_layer_name, i)
            batch_norm_layer_object = architecture_utils.get_batch_norm_layer(
                layer_name=this_name
            )

            layer_name_to_input_layer_names[this_name] = layer_names[-1]
            layer_name_to_object[this_name] = batch_norm_layer_object
            layer_names.append(this_name)

    return layer_names, layer_name_to_input_layer_names, layer_name_to_object


def _pad_2d_layer(source_layer_object, target_layer_object, padding_layer_name):
    """Pads layer with 2 spatial dimensions.

    :param source_layer_object: Source layer.
    :param target_layer_object: Target layer.  The source layer will be padded,
        if necessary, to have the same dimensions as the target layer.
    :param padding_layer_name: Name of padding layer.
    :return: source_layer_object: Same as input, except maybe with different
        spatial dimensions.
    """

    num_source_rows = source_layer_object.shape[1]
    num_target_rows = target_layer_object.shape[1]
    num_padding_rows = num_target_rows - num_source_rows

    num_source_columns = source_layer_object.shape[2]
    num_target_columns = target_layer_object.shape[2]
    num_padding_columns = num_target_columns - num_source_columns

    if num_padding_rows + num_padding_columns > 0:
        padding_arg = ((0, num_padding_rows), (0, num_padding_columns))

        return keras.layers.ZeroPadding2D(
            padding=padding_arg, name=padding_layer_name
        )(source_layer_object)

    return source_layer_object


def _combine_layer_lists(
        layer_name_lists, layer_name_to_input_dicts,
        layer_name_to_object_dicts):
    """Combines layer lists.

    S = number of layer sets

    :param layer_name_lists: length-S list, where the element
        layer_name_lists[i] is a 1-D list of layer names for the [i]th set.
    :param layer_name_to_input_dicts: length-S list, where the element
        layer_name_to_input_dicts[i] is a dictionary for the [i]th set,
        such that each key is a string from layer_name_lists[i] and the
        corresponding value is a list of strings -- also from
        layer_name_lists[i] -- indicating the inputs to said layer.
    :param layer_name_to_object_dicts: length-S list, where the element
        layer_name_to_object_dicts[i] is a dictionary for the [i]th set,
        such that each key is a string from layer_name_lists[i] and the
        corresponding value is the layer object.
    :return: layer_names: A single 1-D list of layer names.
    :return: layer_name_to_input_layer_names: A single dictionary, where each
        key is a string from layer_names and the corresponding value is a list
        of strings -- also from layer_names -- indicating the inputs to said
        layer.
    :return: layer_name_to_object: A single dictionary, where each key is a
        string from layer_names and the corresponding value is the layer object.
    """

    layer_names = layer_name_lists[0]
    layer_name_to_input_layer_names = layer_name_to_input_dicts[0]
    layer_name_to_object = layer_name_to_object_dicts[0]

    for i in range(1, len(layer_name_lists)):
        layer_names += layer_name_lists[i]
        layer_name_to_input_layer_names.update(
            layer_name_to_input_dicts[i]
        )
        layer_name_to_object.update(
            layer_name_to_object_dicts[i]
        )

    return layer_names, layer_name_to_input_layer_names, layer_name_to_object


def _construct_basic_model(layer_names, layer_name_to_input_layer_names,
                           layer_name_to_object_immutable, ensemble_size,
                           integration_step):
    """Constructs basic model, i.e., puts the layers together.

    :param layer_names: See output doc for `_combine_layer_lists`.
    :param layer_name_to_input_layer_names: Same.
    :param layer_name_to_object_immutable: Same.
    :return: output_layer_object: Output layer.
    """

    # TODO: input doc

    layer_name_to_object = dict()
    for this_key in layer_name_to_object_immutable:
        layer_name_to_object[this_key] = (
            layer_name_to_object_immutable[this_key]
        )

    for curr_layer_name in layer_names:
        input_layer_names = layer_name_to_input_layer_names[curr_layer_name]
        if not isinstance(input_layer_names, list):
            input_layer_names = [input_layer_names]

        if len(input_layer_names) == 0:
            continue

        input_objects = [layer_name_to_object[n] for n in input_layer_names]

        try:
            if len(input_objects) == 1:
                layer_name_to_object[curr_layer_name] = (
                    layer_name_to_object[curr_layer_name](input_objects[0])
                )
            else:
                layer_name_to_object[curr_layer_name] = (
                    layer_name_to_object[curr_layer_name](input_objects)
                )

            continue
        except:
            pass

        input_pixel_counts = numpy.array(
            [l.shape[1] * l.shape[2] for l in input_objects],
            dtype=int
        )
        target_layer_index = numpy.argmax(input_pixel_counts)

        for j in range(len(input_objects)):
            if j == target_layer_index:
                continue

            input_objects[j] = _pad_2d_layer(
                source_layer_object=input_objects[j],
                target_layer_object=input_objects[target_layer_index],
                padding_layer_name='padding_{0:.7f}'.format(time.time())
            )

            layer_name_to_object[input_layer_names[j]] = (
                input_objects[j]
            )

        layer_name_to_object[curr_layer_name] = (
            layer_name_to_object[curr_layer_name](input_objects)
        )

    if ensemble_size > 1:
        this_name = 'take_ens_mean_step{0:d}'.format(integration_step + 1)
        extra_output_object = EnsembleMeanLayer(name=this_name)(
            layer_name_to_object[layer_names[-1]]
        )
        print(extra_output_object)
    else:
        extra_output_object = None

    return layer_name_to_object[layer_names[-1]], extra_output_object


def _construct_recurrent_model(
        layer_names, layer_name_to_input_layer_names, layer_name_to_object,
        num_integration_steps, ensemble_size, use_evidential_nn):
    """Constructs recurrent model, i.e., puts the layers together.

    :param layer_names: See output doc for `_combine_layer_lists`.
    :param layer_name_to_input_layer_names: Same.
    :param layer_name_to_object: Same.
    :param num_integration_steps: Number of model-integration steps.
    :param ensemble_size: Ensemble size for network outputs.
    :param use_evidential_nn: Boolean flag, indicating whether or not the
        network is evidential.
    :return: output_layer_objects: 1-D list of output layers.
    """

    # TODO(thunderhoser): Make this work for evidential NNs (low priority,
    # since evidential NNs appear to perform poorly for this problem in
    # general).

    if use_evidential_nn:
        raise ValueError()

    output_layer_objects = [None] * num_integration_steps
    output_layer_objects[0], extra_layer_object = _construct_basic_model(
        layer_names=layer_names,
        layer_name_to_input_layer_names=layer_name_to_input_layer_names,
        layer_name_to_object_immutable=layer_name_to_object,
        ensemble_size=ensemble_size,
        integration_step=0
    )

    for i in range(1, num_integration_steps):
        if extra_layer_object is None:
            layer_name_to_object['predn_baseline_inputs'] = (
                output_layer_objects[i - 1]
            )
        else:
            layer_name_to_object['predn_baseline_inputs'] = extra_layer_object

        output_layer_objects[i], extra_layer_object = _construct_basic_model(
            layer_names=layer_names,
            layer_name_to_input_layer_names=layer_name_to_input_layer_names,
            layer_name_to_object_immutable=layer_name_to_object,
            ensemble_size=ensemble_size,
            integration_step=i
        )

    return output_layer_objects


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
    error_checking.assert_is_list(metric_list)

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

    optimizer_function = option_dict[OPTIMIZER_FUNCTION_KEY]
    num_gfs_lead_times = None

    layer_names = []
    layer_name_to_input_layer_names = dict()
    layer_name_to_object = dict()

    if input_dimensions_gfs_3d is None:
        input_layer_object_gfs_3d = None
        layer_object_gfs_3d = None
        layer_name_gfs_3d = None
    else:
        this_name = 'gfs_3d_inputs'
        input_layer_object_gfs_3d = keras.layers.Input(
            shape=tuple(input_dimensions_gfs_3d.tolist()),
            name=this_name
        )

        layer_name_to_input_layer_names[this_name] = []
        layer_name_to_object[this_name] = input_layer_object_gfs_3d
        layer_names.append(this_name)

        this_name = 'gfs_3d_put-time-first'
        permute_layer_object = keras.layers.Permute(
            dims=(4, 1, 2, 3, 5),
            name=this_name
        )

        layer_name_to_input_layer_names[this_name] = layer_names[-1]
        layer_name_to_object[this_name] = permute_layer_object
        layer_names.append(this_name)

        new_dims = (
            input_dimensions_gfs_3d[3],
            input_dimensions_gfs_3d[0],
            input_dimensions_gfs_3d[1],
            input_dimensions_gfs_3d[2] * input_dimensions_gfs_3d[4]
        )
        this_name = 'gfs_3d_flatten-pressure-levels'
        reshape_layer_object = keras.layers.Reshape(
            target_shape=new_dims,
            name=this_name
        )

        layer_name_to_input_layer_names[this_name] = layer_names[-1]
        layer_name_to_object[this_name] = reshape_layer_object
        layer_names.append(this_name)

        layer_object_gfs_3d = reshape_layer_object
        layer_name_gfs_3d = this_name
        num_gfs_lead_times = input_dimensions_gfs_3d[-2]

    if input_dimensions_gfs_2d is None:
        input_layer_object_gfs_2d = None
        layer_object_gfs_2d = None
        layer_name_gfs_2d = None
    else:
        this_name = 'gfs_2d_inputs'
        input_layer_object_gfs_2d = keras.layers.Input(
            shape=tuple(input_dimensions_gfs_2d.tolist()),
            name=this_name
        )

        layer_name_to_input_layer_names[this_name] = []
        layer_name_to_object[this_name] = input_layer_object_gfs_2d
        layer_names.append(this_name)

        this_name = 'gfs_2d_put-time-first'
        permute_layer_object = keras.layers.Permute(
            dims=(3, 1, 2, 4),
            name=this_name
        )

        layer_name_to_input_layer_names[this_name] = layer_names[-1]
        layer_name_to_object[this_name] = permute_layer_object
        layer_names.append(this_name)

        layer_object_gfs_2d = permute_layer_object
        layer_name_gfs_2d = this_name
        num_gfs_lead_times = input_dimensions_gfs_2d[-2]

    if input_dimensions_gfs_3d is None:
        layer_object_gfs = layer_object_gfs_2d
        layer_name_gfs = layer_name_gfs_2d
    elif input_dimensions_gfs_2d is None:
        layer_object_gfs = layer_object_gfs_3d
        layer_name_gfs = layer_name_gfs_3d
    else:
        this_name = 'gfs_concat-2d-and-3d'
        layer_object_gfs = keras.layers.Concatenate(axis=-1, name=this_name)

        layer_name_to_input_layer_names[this_name] = [
            layer_name_gfs_3d, layer_name_gfs_2d
        ]
        layer_name_to_object[this_name] = layer_object_gfs
        layer_names.append(this_name)

        layer_name_gfs = this_name

    this_name = 'lagged_target_inputs'
    input_layer_object_lagged_target = keras.layers.Input(
        shape=tuple(input_dimensions_lagged_target.tolist()),
        name=this_name
    )

    layer_name_to_input_layer_names[this_name] = []
    layer_name_to_object[this_name] = input_layer_object_lagged_target
    layer_names.append(this_name)

    layer_name_lagged_target = 'lagged_targets_put-time-first'
    layer_object_lagged_target = keras.layers.Permute(
        dims=(3, 1, 2, 4),
        name=layer_name_lagged_target
    )

    layer_name_to_input_layer_names[layer_name_lagged_target] = layer_names[-1]
    layer_name_to_object[layer_name_lagged_target] = layer_object_lagged_target
    layer_names.append(layer_name_lagged_target)

    if input_dimensions_predn_baseline is None:
        input_layer_object_predn_baseline = None
        input_layer_name_predn_baseline = None
    else:
        this_name = 'predn_baseline_inputs'
        input_layer_name_predn_baseline = this_name
        input_layer_object_predn_baseline = keras.layers.Input(
            shape=tuple(input_dimensions_predn_baseline.tolist()),
            name=this_name
        )

        layer_name_to_input_layer_names[this_name] = []
        layer_name_to_object[this_name] = input_layer_object_predn_baseline
        layer_names.append(this_name)

    num_target_lag_times = input_dimensions_lagged_target[-2]
    num_target_fields = input_dimensions_lagged_target[-1]

    if input_dimensions_era5 is None:
        input_layer_object_era5 = None
    else:
        this_name = 'era5_inputs'
        input_layer_object_era5 = keras.layers.Input(
            shape=tuple(input_dimensions_era5.tolist()),
            name=this_name
        )

        layer_name_to_input_layer_names[this_name] = []
        layer_name_to_object[this_name] = input_layer_object_era5
        layer_names.append(this_name)

        new_dims = (1,) + tuple(input_dimensions_era5.tolist())
        reshape_layer_name = 'era5_add-time-dim'
        reshape_layer_object = keras.layers.Reshape(
            target_shape=new_dims, name=reshape_layer_name
        )

        layer_name_to_input_layer_names[reshape_layer_name] = layer_names[-1]
        layer_name_to_object[reshape_layer_name] = reshape_layer_object
        layer_names.append(reshape_layer_name)

        add_gfs_times_layer_name = 'era5_add-gfs-times'
        add_gfs_times_layer_object = keras.layers.Concatenate(
            axis=-4, name=add_gfs_times_layer_name
        )

        layer_name_to_input_layer_names[add_gfs_times_layer_name] = (
            num_gfs_lead_times * [reshape_layer_name]
        )
        layer_name_to_object[add_gfs_times_layer_name] = (
            add_gfs_times_layer_object
        )
        layer_names.append(add_gfs_times_layer_name)

        this_name = 'gfs_concat-era5'
        layer_names.append(this_name)
        layer_name_to_input_layer_names[this_name] = [
            layer_name_gfs, add_gfs_times_layer_name
        ]

        layer_object_gfs = keras.layers.Concatenate(
            axis=-1, name=this_name
        )
        layer_name_gfs = 'gfs_concat-era5'
        layer_name_to_object[layer_name_gfs] = layer_object_gfs

        add_lag_times_layer_name = 'era5_add-lag-times'
        add_lag_times_layer_object = keras.layers.Concatenate(
            axis=-4, name=add_lag_times_layer_name
        )

        layer_name_to_input_layer_names[add_lag_times_layer_name] = (
            num_target_lag_times * [reshape_layer_name]
        )
        layer_name_to_object[add_lag_times_layer_name] = (
            add_lag_times_layer_object
        )
        layer_names.append(add_lag_times_layer_name)

        this_name = 'lagged_targets_concat-era5'
        layer_names.append(this_name)
        layer_name_to_input_layer_names[this_name] = [
            layer_name_lagged_target, add_lag_times_layer_name
        ]

        layer_object_lagged_target = keras.layers.Concatenate(
            axis=-1, name=this_name
        )
        layer_name_lagged_target = 'lagged_targets_concat-era5'
        layer_name_to_object[layer_name_lagged_target] = (
            layer_object_lagged_target
        )

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    gfs_encoder_conv_layer_names = [''] * (num_levels + 1)
    gfs_fcst_module_layer_names = [''] * (num_levels + 1)
    gfs_encoder_pooling_layer_names = [''] * num_levels

    for i in range(num_levels + 1):
        if i == 0:
            this_input_layer_name = layer_name_gfs
        else:
            this_input_layer_name = gfs_encoder_pooling_layer_names[i - 1]

        (
            new_layer_names,
            new_layer_name_to_input_layer_names,
            new_layer_name_to_object
        ) = _get_2d_conv_block(
            input_layer_name=this_input_layer_name,
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

        gfs_encoder_conv_layer_names[i] = new_layer_names[-1]

        (
            layer_names,
            layer_name_to_input_layer_names,
            layer_name_to_object
        ) = _combine_layer_lists(
            layer_name_lists=[layer_names, new_layer_names],
            layer_name_to_input_dicts=[
                layer_name_to_input_layer_names,
                new_layer_name_to_input_layer_names
            ],
            layer_name_to_object_dicts=[
                layer_name_to_object,
                new_layer_name_to_object
            ]
        )

        this_name = 'gfs_fcst_level{0:d}_put-time-last'.format(i)
        gfs_fcst_module_layer_names[i] = this_name
        permute_layer_object = keras.layers.Permute(
            dims=(2, 3, 1, 4), name=this_name
        )

        layer_name_to_input_layer_names[this_name] = layer_names[-1]
        layer_name_to_object[this_name] = permute_layer_object
        layer_names.append(this_name)

        if gfs_fcst_use_3d_conv:
            (
                new_layer_names,
                new_layer_name_to_input_layer_names,
                new_layer_name_to_object
            ) = _get_3d_conv_block(
                input_layer_name=gfs_fcst_module_layer_names[i],
                num_time_steps=num_gfs_lead_times,
                num_filters=gfs_encoder_num_channels_by_level[i],
                do_residual=use_residual_blocks,
                num_conv_layers=gfs_fcst_num_conv_layers,
                filter_size_px=1,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=gfs_fcst_dropout_rates,
                use_batch_norm=use_batch_normalization,
                basic_layer_name='gfs_fcst_level{0:d}'.format(i)
            )

            (
                layer_names,
                layer_name_to_input_layer_names,
                layer_name_to_object
            ) = _combine_layer_lists(
                layer_name_lists=[layer_names, new_layer_names],
                layer_name_to_input_dicts=[
                    layer_name_to_input_layer_names,
                    new_layer_name_to_input_layer_names
                ],
                layer_name_to_object_dicts=[
                    layer_name_to_object,
                    new_layer_name_to_object
                ]
            )
        else:
            this_name = 'gfs_fcst_level{0:d}_remove-time-dim'.format(i)
            gfs_fcst_module_layer_names[i] = this_name
            remove_time_layer_object = RemoveTimeDimLayer(name=this_name)  # TODO(thunderhoser): Fuck, this is wrong.

            layer_name_to_input_layer_names[this_name] = layer_names[-1]
            layer_name_to_object[this_name] = remove_time_layer_object
            layer_names.append(this_name)

            (
                new_layer_names,
                new_layer_name_to_input_layer_names,
                new_layer_name_to_object
            ) = _get_2d_conv_block(
                input_layer_name=gfs_fcst_module_layer_names[i],
                do_residual=use_residual_blocks,
                num_conv_layers=gfs_fcst_num_conv_layers,
                filter_size_px=1,
                num_filters=gfs_encoder_num_channels_by_level[i],
                do_time_distributed_conv=False,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=gfs_fcst_dropout_rates,
                use_batch_norm=use_batch_normalization,
                basic_layer_name='gfs_fcst_level{0:d}'.format(i)
            )

            (
                layer_names,
                layer_name_to_input_layer_names,
                layer_name_to_object
            ) = _combine_layer_lists(
                layer_name_lists=[layer_names, new_layer_names],
                layer_name_to_input_dicts=[
                    layer_name_to_input_layer_names,
                    new_layer_name_to_input_layer_names
                ],
                layer_name_to_object_dicts=[
                    layer_name_to_object,
                    new_layer_name_to_object
                ]
            )

        gfs_fcst_module_layer_names[i] = layer_names[-1]

        if i == num_levels:
            break

        this_name = 'gfs_encoder_level{0:d}_pooling'.format(i)
        gfs_encoder_pooling_layer_names[i] = this_name

        pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_rows_per_stride=2, num_columns_per_stride=2,
            pooling_type_string=architecture_utils.MAX_POOLING_STRING
        )
        pooling_layer_object = keras.layers.TimeDistributed(
            pooling_layer_object, name=this_name
        )

        layer_name_to_input_layer_names[this_name] = (
            gfs_encoder_conv_layer_names[i]
        )
        layer_name_to_object[this_name] = pooling_layer_object
        layer_names.append(this_name)

    lagtgt_encoder_conv_layer_names = [''] * (num_levels + 1)
    lagtgt_fcst_module_layer_names = [''] * (num_levels + 1)
    lagtgt_encoder_pooling_layer_names = [''] * num_levels

    for i in range(num_levels + 1):
        if i == 0:
            this_input_layer_name = layer_name_lagged_target
        else:
            this_input_layer_name = lagtgt_encoder_pooling_layer_names[i - 1]

        (
            new_layer_names,
            new_layer_name_to_input_layer_names,
            new_layer_name_to_object
        ) = _get_2d_conv_block(
            input_layer_name=this_input_layer_name,
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

        lagtgt_encoder_conv_layer_names[i] = new_layer_names[-1]

        (
            layer_names,
            layer_name_to_input_layer_names,
            layer_name_to_object
        ) = _combine_layer_lists(
            layer_name_lists=[layer_names, new_layer_names],
            layer_name_to_input_dicts=[
                layer_name_to_input_layer_names,
                new_layer_name_to_input_layer_names
            ],
            layer_name_to_object_dicts=[
                layer_name_to_object,
                new_layer_name_to_object
            ]
        )

        this_name = 'lagtgt_fcst_level{0:d}_put-time-last'.format(i)
        lagtgt_fcst_module_layer_names[i] = this_name
        permute_layer_object = keras.layers.Permute(
            dims=(2, 3, 1, 4), name=this_name
        )

        layer_name_to_input_layer_names[this_name] = layer_names[-1]
        layer_name_to_object[this_name] = permute_layer_object
        layer_names.append(this_name)

        if lagtgt_fcst_use_3d_conv:
            (
                new_layer_names,
                new_layer_name_to_input_layer_names,
                new_layer_name_to_object
            ) = _get_3d_conv_block(
                input_layer_name=lagtgt_fcst_module_layer_names[i],
                num_time_steps=num_target_lag_times,
                num_filters=lagtgt_encoder_num_channels_by_level[i],
                do_residual=use_residual_blocks,
                num_conv_layers=lagtgt_fcst_num_conv_layers,
                filter_size_px=1,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=lagtgt_fcst_dropout_rates,
                use_batch_norm=use_batch_normalization,
                basic_layer_name='lagtgt_fcst_level{0:d}'.format(i)
            )

            (
                layer_names,
                layer_name_to_input_layer_names,
                layer_name_to_object
            ) = _combine_layer_lists(
                layer_name_lists=[layer_names, new_layer_names],
                layer_name_to_input_dicts=[
                    layer_name_to_input_layer_names,
                    new_layer_name_to_input_layer_names
                ],
                layer_name_to_object_dicts=[
                    layer_name_to_object,
                    new_layer_name_to_object
                ]
            )
        else:
            this_name = 'lagtgt_fcst_level{0:d}_remove-time-dim'.format(i)
            lagtgt_fcst_module_layer_names[i] = this_name
            remove_time_layer_object = RemoveTimeDimLayer(name=this_name)  # TODO(thunderhoser): Fuck, this is wrong.

            layer_name_to_input_layer_names[this_name] = layer_names[-1]
            layer_name_to_object[this_name] = remove_time_layer_object
            layer_names.append(this_name)

            (
                new_layer_names,
                new_layer_name_to_input_layer_names,
                new_layer_name_to_object
            ) = _get_2d_conv_block(
                input_layer_name=lagtgt_fcst_module_layer_names[i],
                do_residual=use_residual_blocks,
                num_conv_layers=lagtgt_fcst_num_conv_layers,
                filter_size_px=1,
                num_filters=lagtgt_encoder_num_channels_by_level[i],
                do_time_distributed_conv=False,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=lagtgt_fcst_dropout_rates,
                use_batch_norm=use_batch_normalization,
                basic_layer_name='lagtgt_fcst_level{0:d}'.format(i)
            )

            (
                layer_names,
                layer_name_to_input_layer_names,
                layer_name_to_object
            ) = _combine_layer_lists(
                layer_name_lists=[layer_names, new_layer_names],
                layer_name_to_input_dicts=[
                    layer_name_to_input_layer_names,
                    new_layer_name_to_input_layer_names
                ],
                layer_name_to_object_dicts=[
                    layer_name_to_object,
                    new_layer_name_to_object
                ]
            )

        lagtgt_fcst_module_layer_names[i] = layer_names[-1]

        if i == num_levels:
            break

        this_name = 'lagtgt_encoder_level{0:d}_pooling'.format(i)
        lagtgt_encoder_pooling_layer_names[i] = this_name
        pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_rows_per_stride=2, num_columns_per_stride=2,
            pooling_type_string=architecture_utils.MAX_POOLING_STRING
        )
        pooling_layer_object = keras.layers.TimeDistributed(
            pooling_layer_object, name=this_name
        )

        layer_name_to_input_layer_names[this_name] = (
            lagtgt_encoder_conv_layer_names[i]
        )
        layer_name_to_object[this_name] = pooling_layer_object
        layer_names.append(this_name)

    last_conv_layer_name_matrix = numpy.full(
        (num_levels + 1, num_levels + 1), '', dtype=object
    )
    last_conv_num_channels_matrix = numpy.full(
        (num_levels + 1, num_levels + 1), -1, dtype=int
    )

    for i in range(num_levels + 1):
        this_name = 'fcst_level{0:d}_concat'.format(i)
        last_conv_layer_name_matrix[i, 0] = this_name
        concat_layer_object = keras.layers.Concatenate(axis=-1, name=this_name)

        layer_name_to_input_layer_names[this_name] = [
            gfs_fcst_module_layer_names[i],
            lagtgt_fcst_module_layer_names[i]
        ]
        layer_name_to_object[this_name] = concat_layer_object
        layer_names.append(this_name)

        last_conv_num_channels_matrix[i, 0] = (
            gfs_encoder_num_channels_by_level[i] +
            lagtgt_encoder_num_channels_by_level[i]
        )

        i_new = i + 0
        j = 0

        while i_new > 0:
            i_new -= 1
            j += 1

            upsampling_layer_name = 'block{0:d}-{1:d}_upsampling'.format(
                i_new, j
            )
            upsampling_layer_object = keras.layers.UpSampling2D(
                size=(2, 2), name=upsampling_layer_name
            )

            layer_name_to_input_layer_names[upsampling_layer_name] = (
                last_conv_layer_name_matrix[i_new + 1, j - 1]
            )
            layer_name_to_object[upsampling_layer_name] = (
                upsampling_layer_object
            )
            layer_names.append(upsampling_layer_name)

            this_num_channels = int(numpy.round(
                0.5 * decoder_num_channels_by_level[i_new]
            ))

            (
                new_layer_names,
                new_layer_name_to_input_layer_names,
                new_layer_name_to_object
            ) = _get_2d_conv_block(
                input_layer_name=upsampling_layer_name,
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

            last_conv_layer_name_matrix[i_new, j] = new_layer_names[-1]
            last_conv_num_channels_matrix[i_new, j] = this_num_channels

            (
                layer_names,
                layer_name_to_input_layer_names,
                layer_name_to_object
            ) = _combine_layer_lists(
                layer_name_lists=[layer_names, new_layer_names],
                layer_name_to_input_dicts=[
                    layer_name_to_input_layer_names,
                    new_layer_name_to_input_layer_names
                ],
                layer_name_to_object_dicts=[
                    layer_name_to_object,
                    new_layer_name_to_object
                ]
            )

            (
                new_layer_names,
                new_layer_name_to_input_layer_names,
                new_layer_name_to_object
            ) = _create_skip_connection(
                input_layer_names=
                last_conv_layer_name_matrix[i_new, :(j + 1)].tolist(),
                input_channel_counts=
                last_conv_num_channels_matrix[i_new, :(j + 1)],
                num_output_channels=decoder_num_channels_by_level[i_new],
                current_level_num=i_new,
                regularizer_object=regularizer_object
            )

            last_conv_layer_name_matrix[i_new, j] = new_layer_names[-1]

            (
                layer_names,
                layer_name_to_input_layer_names,
                layer_name_to_object
            ) = _combine_layer_lists(
                layer_name_lists=[layer_names, new_layer_names],
                layer_name_to_input_dicts=[
                    layer_name_to_input_layer_names,
                    new_layer_name_to_input_layer_names
                ],
                layer_name_to_object_dicts=[
                    layer_name_to_object,
                    new_layer_name_to_object
                ]
            )

            (
                new_layer_names,
                new_layer_name_to_input_layer_names,
                new_layer_name_to_object
            ) = _get_2d_conv_block(
                input_layer_name=last_conv_layer_name_matrix[i_new, j],
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

            last_conv_layer_name_matrix[i_new, j] = new_layer_names[-1]

            (
                layer_names,
                layer_name_to_input_layer_names,
                layer_name_to_object
            ) = _combine_layer_lists(
                layer_name_lists=[layer_names, new_layer_names],
                layer_name_to_input_dicts=[
                    layer_name_to_input_layer_names,
                    new_layer_name_to_input_layer_names
                ],
                layer_name_to_object_dicts=[
                    layer_name_to_object,
                    new_layer_name_to_object
                ]
            )

    if include_penultimate_conv:
        (
            new_layer_names,
            new_layer_name_to_input_layer_names,
            new_layer_name_to_object
        ) = _get_2d_conv_block(
            input_layer_name=last_conv_layer_name_matrix[0, -1],
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

        last_conv_layer_name_matrix[0, -1] = new_layer_names[-1]

        (
            layer_names,
            layer_name_to_input_layer_names,
            layer_name_to_object
        ) = _combine_layer_lists(
            layer_name_lists=[layer_names, new_layer_names],
            layer_name_to_input_dicts=[
                layer_name_to_input_layer_names,
                new_layer_name_to_input_layer_names
            ],
            layer_name_to_object_dicts=[
                layer_name_to_object,
                new_layer_name_to_object
            ]
        )

    (
        new_layer_names,
        new_layer_name_to_input_layer_names,
        new_layer_name_to_object
    ) = _get_2d_conv_block(
        input_layer_name=last_conv_layer_name_matrix[0, -1],
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

    (
        layer_names,
        layer_name_to_input_layer_names,
        layer_name_to_object
    ) = _combine_layer_lists(
        layer_name_lists=[layer_names, new_layer_names],
        layer_name_to_input_dicts=[
            layer_name_to_input_layer_names,
            new_layer_name_to_input_layer_names
        ],
        layer_name_to_object_dicts=[
            layer_name_to_object,
            new_layer_name_to_object
        ]
    )

    if ensemble_size > 1:
        new_dims = (
            input_dimensions_lagged_target[0],
            input_dimensions_lagged_target[1],
            num_target_fields,
            ensemble_size
        )
        this_name = 'reshape_output'
        reshape_layer_object = keras.layers.Reshape(
            target_shape=new_dims, name=this_name
        )

        layer_name_to_input_layer_names[this_name] = layer_names[-1]
        layer_name_to_object[this_name] = reshape_layer_object
        layer_names.append(this_name)

    output_layer_name = layer_names[-1]

    if input_layer_object_predn_baseline is not None:
        if ensemble_size > 1:
            new_dims = (
                input_dimensions_predn_baseline[0],
                input_dimensions_predn_baseline[1],
                input_dimensions_predn_baseline[2],
                1
            )
            this_name = 'reshape_predn_baseline'
            reshape_layer_object = keras.layers.Reshape(
                target_shape=new_dims,
                name=this_name
            )

            layer_name_to_input_layer_names[this_name] = (
                input_layer_name_predn_baseline
            )
            layer_name_to_object[this_name] = reshape_layer_object
            layer_names.append(this_name)

            if use_evidential_nn:
                this_name = 'permute_predn_baseline'
                permute_layer_object = keras.layers.Permute(
                    dims=(1, 2, 4, 3),
                    name=this_name
                )

                layer_name_to_input_layer_names[this_name] = layer_names[-1]
                layer_name_to_object[this_name] = permute_layer_object
                layer_names.append(this_name)

                padding_arg = ((0, 0), (0, 0), (0, ensemble_size - 1))
                this_name = 'pad_predn_baseline'
                padding_layer_object = keras.layers.ZeroPadding3D(
                    padding=padding_arg, name=this_name
                )

                layer_name_to_input_layer_names[this_name] = layer_names[-1]
                layer_name_to_object[this_name] = padding_layer_object
                layer_names.append(this_name)

                this_name = 'permute_predn_baseline_back'
                permute_layer_object = keras.layers.Permute(
                    dims=(1, 2, 4, 3),
                    name=this_name
                )

                layer_name_to_input_layer_names[this_name] = layer_names[-1]
                layer_name_to_object[this_name] = permute_layer_object
                layer_names.append(this_name)

            layer_name_predn_baseline = layer_names[-1]
        else:
            layer_name_predn_baseline = input_layer_name_predn_baseline

        this_name = 'output_add_baseline'
        add_layer_object = keras.layers.Add(name=this_name)

        layer_name_to_input_layer_names[this_name] = [
            output_layer_name, layer_name_predn_baseline
        ]
        layer_name_to_object[this_name] = add_layer_object
        layer_names.append(this_name)

    input_layer_objects = [
        l for l in [
            input_layer_object_gfs_3d, input_layer_object_gfs_2d,
            input_layer_object_era5, input_layer_object_lagged_target,
            input_layer_object_predn_baseline
        ] if l is not None
    ]

    output_layer_objects = _construct_recurrent_model(
        layer_names=layer_names,
        layer_name_to_input_layer_names=layer_name_to_input_layer_names,
        layer_name_to_object=layer_name_to_object,
        num_integration_steps=2,  # TODO(thunderhoser): Should be input arg.
        ensemble_size=ensemble_size,
        use_evidential_nn=use_evidential_nn
    )

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=output_layer_objects
    )

    model_object.compile(
        loss=loss_function, optimizer=optimizer_function,
        metrics=metric_list
    )

    model_object.summary()
    return model_object
