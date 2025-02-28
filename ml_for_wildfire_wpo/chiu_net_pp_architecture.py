"""Methods for building a Chiu-net++.

The Chiu-net++ is a hybrid between the Chiu net
(https://doi.org/10.1109/LRA.2020.2992184) and U-net++.
"""

import os
import sys
import numpy
import keras
import tensorflow
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import architecture_utils
import chiu_net_architecture as chiu_net_arch

GFS_3D_DIMENSIONS_KEY = chiu_net_arch.GFS_3D_DIMENSIONS_KEY
GFS_2D_DIMENSIONS_KEY = chiu_net_arch.GFS_2D_DIMENSIONS_KEY
ERA5_CONST_DIMENSIONS_KEY = chiu_net_arch.ERA5_CONST_DIMENSIONS_KEY
LAGTGT_DIMENSIONS_KEY = chiu_net_arch.LAGTGT_DIMENSIONS_KEY
PREDN_BASELINE_DIMENSIONS_KEY = 'input_dimensions_predn_baseline'
USE_RESIDUAL_BLOCKS_KEY = 'use_residual_blocks'
USE_CONVNEXT_BLOCKS_KEY = 'use_convnext_blocks'
USE_LEAD_TIME_AS_PRED_KEY = 'use_lead_time_as_predictor'
CONSTRAINT_INDICES_KEY = 'dmc_dc_isi_indices_for_constraints'
NUM_FREE_TARGETS_KEY = 'num_free_target_fields'

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
LOSS_FUNCTION_KEY = chiu_net_arch.LOSS_FUNCTION_KEY
METRIC_FUNCTIONS_KEY = chiu_net_arch.METRIC_FUNCTIONS_KEY

EPSILON_FOR_LAYER_NORM = 1e-6
EXPANSION_FACTOR_FOR_CONVNEXT = 4
INIT_VALUE_FOR_LAYER_SCALE = 1e-6


# @keras.saving.register_keras_serializable()
class LayerScale(keras.layers.Layer):
    """Layer-scale module.

    Scavenged from: https://github.com/danielabdi-noaa/HRRRemulator/blob/
                    master/tfmodel/convnext.py
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, _):
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
            name="gamma",
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


# @keras.saving.register_keras_serializable()
class StochasticDepth(keras.layers.Layer):
    def __init__(self, survival_prob=0.9, **kwargs):
        super().__init__(**kwargs)
        self.survival_prob = survival_prob

    def call(self, inputs, training=None):
        if not training:
            return inputs[0] + inputs[1]

        batch_size = tensorflow.shape(inputs[0])[0]
        random_tensor = self.survival_prob + tensorflow.random.uniform(
            [batch_size, 1, 1, 1]
        )
        binary_tensor = tensorflow.floor(random_tensor)
        output = inputs[0] + binary_tensor * inputs[1] / self.survival_prob
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "survival_prob": self.survival_prob
            }
        )
        return config


class DeriveBUIPredictions(keras.layers.Layer):
    def __init__(self, dmc_index, dc_index, expect_ensemble=False, **kwargs):
        """Constructor.

        :param dmc_index: Array index for DMC.  This tells the method that DMC
            predictions can be found at
            prediction_tensor[:, :, :, dmc_index, ...].
        :param dc_index: Same as above but for DC.
        :param expect_ensemble: Boolean flag, indicating whether to expect
            ensemble or deterministic predictions.
        :param kwargs: Keyword arguments.
        """

        super(DeriveBUIPredictions, self).__init__(**kwargs)
        self.dmc_index = dmc_index
        self.dc_index = dc_index
        self.expect_ensemble = expect_ensemble

    def _power(self, input_tensor, exponent):
        """Computes power.

        :param input_tensor: Keras tensor.
        :param exponent: Scalar exponent (float).
        :return: power_tensor: Keras tensor with the same shape as
            input_tensor`.
        """

        power_tensor = K.pow(K.maximum(input_tensor, K.epsilon()), exponent)
        return K.maximum(power_tensor, K.epsilon())

    def _compute_bui(self, dmc_tensor, dc_tensor):
        """Computes BUI from DMC and DC.

        :param dmc_tensor: Tensor of DMC values.
        :param dc_tensor: Tensor of DC values.
        :return: bui_tensor: Tensor of BUI values.
        """

        new_dmc_tensor = K.maximum(dmc_tensor, K.epsilon())
        new_dc_tensor = K.maximum(dc_tensor, K.epsilon())

        bui_tensor = (
            (0.8 * new_dmc_tensor * new_dc_tensor) /
            (new_dmc_tensor + 0.4 * new_dc_tensor)
        )
        first_term = (new_dmc_tensor - bui_tensor) / new_dmc_tensor
        second_term = 0.92 + self._power(0.0114 * new_dmc_tensor, 1.7)
        prelim_bui_tensor = new_dmc_tensor - first_term * second_term
        prelim_bui_tensor = K.maximum(prelim_bui_tensor, 0.)

        return tensorflow.where(
            bui_tensor < new_dmc_tensor,
            prelim_bui_tensor,
            bui_tensor
        )

    def call(self, prediction_tensor):
        """Main method.

        :param prediction_tensor: Prediction tensor created by model's final
            layer.
        :return: prediction_tensor: Same as input, except with BUI along last
            slice of channel axis.
        """

        if self.expect_ensemble:
            predicted_dmc_tensor = prediction_tensor[..., self.dmc_index, :]
            predicted_dc_tensor = prediction_tensor[..., self.dc_index, :]
        else:
            predicted_dmc_tensor = prediction_tensor[..., self.dmc_index]
            predicted_dc_tensor = prediction_tensor[..., self.dc_index]

        predicted_bui_tensor = self._compute_bui(
            dmc_tensor=predicted_dmc_tensor,
            dc_tensor=predicted_dc_tensor
        )

        if self.expect_ensemble:
            prediction_tensor = K.concatenate([
                prediction_tensor,
                K.expand_dims(predicted_bui_tensor, axis=-2)
            ], axis=-2)
        else:
            prediction_tensor = K.concatenate([
                prediction_tensor,
                K.expand_dims(predicted_bui_tensor, axis=-1)
            ], axis=-1)

        return prediction_tensor

    def get_config(self):
        """Returns layer configuration."""

        config = super().get_config()
        config.update({
            'dmc_index': self.dmc_index,
            'dc_index': self.dc_index,
            'expect_ensemble': self.expect_ensemble
        })
        return config


class DeriveFWIPredictions(keras.layers.Layer):
    def __init__(self, isi_index, bui_index, expect_ensemble=False, **kwargs):
        """Constructor.

        :param isi_index: Array index for ISI.  This tells the method that ISI
            predictions can be found at
            prediction_tensor[:, :, :, isi_index, ...].
        :param bui_index: Same as above but for BUI.
        :param expect_ensemble: Boolean flag, indicating whether to expect
            ensemble or deterministic predictions.
        :param kwargs: Keyword arguments.
        """

        super(DeriveFWIPredictions, self).__init__(**kwargs)
        self.isi_index = isi_index
        self.bui_index = bui_index
        self.expect_ensemble = expect_ensemble

    def _power(self, input_tensor, exponent):
        """Computes power.

        :param input_tensor: Keras tensor.
        :param exponent: Scalar exponent (float).
        :return: power_tensor: Keras tensor with the same shape as
            input_tensor`.
        """

        power_tensor = K.pow(K.maximum(input_tensor, K.epsilon()), exponent)
        return K.maximum(power_tensor, K.epsilon())

    def _exponential(self, input_tensor):
        """Computes exponential function.

        :param input_tensor: Keras tensor.
        :return: exp_tensor: Keras tensor with the same shape as `input_tensor`.
        """

        return K.maximum(K.exp(input_tensor), K.epsilon())

    def _natural_log(self, input_tensor):
        """Computes natural logarithm.

        :param input_tensor: Keras tensor.
        :return: logarithm_tensor: Keras tensor with the same shape as
            `input_tensor`.
        """

        return K.log(K.maximum(input_tensor, K.epsilon()))

    def call(self, prediction_tensor):
        """Main method.

        :param prediction_tensor: Prediction tensor created by model's final
            layer.
        :return: prediction_tensor: Same as input, except with FWI along last
            slice of channel axis.
        """

        if self.expect_ensemble:
            predicted_isi_tensor = prediction_tensor[..., self.isi_index, :]
            predicted_bui_tensor = prediction_tensor[..., self.bui_index, :]
        else:
            predicted_isi_tensor = prediction_tensor[..., self.isi_index]
            predicted_bui_tensor = prediction_tensor[..., self.bui_index]

        duff_moisture_func = tensorflow.where(
            predicted_bui_tensor <= 80.,
            0.626 * self._power(predicted_bui_tensor, 0.809) + 2,
            1000. / (25 + 108.64 * self._exponential(-0.023 * predicted_bui_tensor))
        )
        prelim_fwi = 0.1 * predicted_isi_tensor * duff_moisture_func
        predicted_fwi_tensor = tensorflow.where(
            prelim_fwi > 1.,
            self._exponential(2.72 * self._power(0.434 * self._natural_log(prelim_fwi), 0.647)),
            prelim_fwi
        )

        predicted_fwi_tensor = tensorflow.where(
            tensorflow.math.is_finite(predicted_fwi_tensor),
            predicted_fwi_tensor,
            tensorflow.zeros_like(predicted_fwi_tensor)
        )
        predicted_fwi_tensor = K.maximum(predicted_fwi_tensor, 0.)

        if self.expect_ensemble:
            prediction_tensor = K.concatenate([
                prediction_tensor,
                K.expand_dims(predicted_fwi_tensor, axis=-2)
            ], axis=-2)
        else:
            prediction_tensor = K.concatenate([
                prediction_tensor,
                K.expand_dims(predicted_fwi_tensor, axis=-1)
            ], axis=-1)

        return prediction_tensor

    def get_config(self):
        """Returns layer configuration."""

        config = super().get_config()
        config.update({
            'isi_index': self.isi_index,
            'bui_index': self.bui_index,
            'expect_ensemble': self.expect_ensemble
        })
        return config


class DeriveDSRPredictions(keras.layers.Layer):
    def __init__(self, fwi_index, expect_ensemble=False, **kwargs):
        """Constructor.

        :param fwi_index: Array index for FWI.  This tells the method that FWI
            predictions can be found at
            prediction_tensor[:, :, :, fwi_index, ...].
        :param expect_ensemble: Boolean flag, indicating whether to expect
            ensemble or deterministic predictions.
        :param kwargs: Keyword arguments.
        """

        super(DeriveDSRPredictions, self).__init__(**kwargs)
        self.fwi_index = fwi_index
        self.expect_ensemble = expect_ensemble

    def _power(self, input_tensor, exponent):
        """Computes power.

        :param input_tensor: Keras tensor.
        :param exponent: Scalar exponent (float).
        :return: power_tensor: Keras tensor with the same shape as
            input_tensor`.
        """

        power_tensor = K.pow(K.maximum(input_tensor, K.epsilon()), exponent)
        return K.maximum(power_tensor, K.epsilon())

    def call(self, prediction_tensor):
        """Main method.

        :param prediction_tensor: Prediction tensor created by model's final
            layer.
        :return: prediction_tensor: Same as input, except with DSR along last
            slice of channel axis.
        """

        if self.expect_ensemble:
            predicted_fwi_tensor = prediction_tensor[..., self.fwi_index, :]
        else:
            predicted_fwi_tensor = prediction_tensor[..., self.fwi_index]

        predicted_dsr_tensor = 0.0272 * self._power(predicted_fwi_tensor, 1.77)
        predicted_dsr_tensor = K.maximum(predicted_dsr_tensor, 0.)

        if self.expect_ensemble:
            prediction_tensor = K.concatenate([
                prediction_tensor,
                K.expand_dims(predicted_dsr_tensor, axis=-2)
            ], axis=-2)
        else:
            prediction_tensor = K.concatenate([
                prediction_tensor,
                K.expand_dims(predicted_dsr_tensor, axis=-1)
            ], axis=-1)

        return prediction_tensor

    def get_config(self):
        """Returns layer configuration."""

        config = super().get_config()
        config.update({
            'fwi_index': self.fwi_index,
            'expect_ensemble': self.expect_ensemble
        })
        return config


def __dimension_to_int(dimension_object):
    """Converts `tensorflow.Dimension` object to integer.

    :param dimension_object: `tensorflow.Dimension` object.
    :return: dimension_int: Integer.
    """

    try:
        return dimension_object.value
    except:
        return dimension_object


def __get_2d_convnext_block(
        input_layer_object, num_conv_layers, filter_size_px, num_filters,
        do_time_distributed_conv, regularizer_object,
        do_activation, dropout_rate, basic_layer_name):
    """Creates ConvNext block for data with 2 spatial dimensions.

    L = number of conv layers

    :param input_layer_object: See documentation for `_get_2d_conv_block`.
    :param num_conv_layers: Same.
    :param filter_size_px: Same.
    :param num_filters: Same.
    :param do_time_distributed_conv: Same.
    :param regularizer_object: Same.
    :param do_activation: Same.
    :param dropout_rate: Same.
    :param basic_layer_name: Same.
    :return: output_layer_object: Same.
    """

    # TODO(thunderhoser): HACK.
    if filter_size_px == 3:
        actual_filter_size_px = 7
    else:
        actual_filter_size_px = filter_size_px + 0

    current_layer_object = None

    for i in range(num_conv_layers):
        if i == 0:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = current_layer_object

        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)
        current_layer_object = architecture_utils.get_2d_depthwise_conv_layer(
            num_kernel_rows=actual_filter_size_px,
            num_kernel_columns=actual_filter_size_px,
            num_rows_per_stride=1,
            num_columns_per_stride=1,
            num_filters=num_filters,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        if do_time_distributed_conv:
            current_layer_object = keras.layers.TimeDistributed(
                current_layer_object, name=this_name
            )(this_input_layer_object)
        else:
            current_layer_object = current_layer_object(this_input_layer_object)

        this_name = '{0:s}_lyrnorm{1:d}'.format(basic_layer_name, i)
        current_layer_object = keras.layers.LayerNormalization(
            epsilon=EPSILON_FOR_LAYER_NORM, name=this_name
        )(
            current_layer_object
        )

        this_name = '{0:s}_dense{1:d}a'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=EXPANSION_FACTOR_FOR_CONVNEXT * num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )
        current_layer_object = dense_layer_object(current_layer_object)

        if do_activation:
            this_name = '{0:s}_gelu{1:d}'.format(basic_layer_name, i)
            current_layer_object = keras.layers.Activation(
                'gelu', name=this_name
            )(current_layer_object)

        this_name = '{0:s}_dense{1:d}b'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )
        current_layer_object = dense_layer_object(current_layer_object)

        this_name = '{0:s}_lyrscale{1:d}'.format(basic_layer_name, i)
        current_layer_object = LayerScale(
            INIT_VALUE_FOR_LAYER_SCALE, num_filters, name=this_name
        )(current_layer_object)

        if i != num_conv_layers - 1:
            continue

        if input_layer_object.shape[-1] == num_filters:
            new_layer_object = input_layer_object
        else:
            this_name = '{0:s}_preresidual_conv'.format(basic_layer_name)
            new_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=1,
                num_kernel_columns=1,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )

            if do_time_distributed_conv:
                new_layer_object = keras.layers.TimeDistributed(
                    new_layer_object, name=this_name
                )(input_layer_object)
            else:
                new_layer_object = new_layer_object(input_layer_object)

        this_name = '{0:s}_residual'.format(basic_layer_name)

        if dropout_rate > 0:
            current_layer_object = StochasticDepth(
                survival_prob=1. - dropout_rate, name=this_name
            )([new_layer_object, current_layer_object])
        else:
            current_layer_object = keras.layers.Add(name=this_name)([
                new_layer_object, current_layer_object
            ])

    return current_layer_object


def __get_3d_convnext_block(
        input_layer_object, num_time_steps, num_conv_layers, filter_size_px,
        regularizer_object, do_activation, dropout_rate, basic_layer_name):
    """Creates ConvNext block for data with 3 spatial dimensions.

    :param input_layer_object: See documentation for `_get_3d_conv_block`.
    :param num_time_steps: Same.
    :param num_conv_layers: Same.
    :param filter_size_px: Same.
    :param regularizer_object: Same.
    :param do_activation: Same.
    :param dropout_rate: Same.
    :param basic_layer_name: Same.
    :return: output_layer_object: Same.
    """

    current_layer_object = None
    num_filters = __dimension_to_int(input_layer_object.shape[-1])

    for i in range(num_conv_layers):
        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)

        if i == 0:
            current_layer_object = architecture_utils.get_3d_conv_layer(
                num_kernel_rows=filter_size_px,
                num_kernel_columns=filter_size_px,
                num_kernel_heights=num_time_steps,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_heights_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.NO_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )
            current_layer_object = current_layer_object(input_layer_object)

            new_dims = (
                current_layer_object.shape[1:3] +
                (current_layer_object.shape[-1],)
            )

            this_name = '{0:s}_remove-time-dim'.format(basic_layer_name)
            current_layer_object = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(current_layer_object)
        else:
            dwconv_layer_object = (
                architecture_utils.get_2d_depthwise_conv_layer(
                    num_kernel_rows=filter_size_px,
                    num_kernel_columns=filter_size_px,
                    num_rows_per_stride=1,
                    num_columns_per_stride=1,
                    num_filters=num_filters,
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )
            )
            current_layer_object = dwconv_layer_object(current_layer_object)

        this_name = '{0:s}_lyrnorm{1:d}'.format(basic_layer_name, i)
        current_layer_object = keras.layers.LayerNormalization(
            epsilon=EPSILON_FOR_LAYER_NORM, name=this_name
        )(
            current_layer_object
        )

        this_name = '{0:s}_dense{1:d}a'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=EXPANSION_FACTOR_FOR_CONVNEXT * num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )
        current_layer_object = dense_layer_object(current_layer_object)

        if do_activation:
            this_name = '{0:s}_gelu{1:d}'.format(basic_layer_name, i)
            current_layer_object = keras.layers.Activation(
                'gelu', name=this_name
            )(current_layer_object)

        this_name = '{0:s}_dense{1:d}b'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )
        current_layer_object = dense_layer_object(current_layer_object)

        this_name = '{0:s}_lyrscale{1:d}'.format(basic_layer_name, i)
        current_layer_object = LayerScale(
            INIT_VALUE_FOR_LAYER_SCALE, num_filters, name=this_name
        )(current_layer_object)

        if i != num_conv_layers - 1:
            continue

        this_name = '{0:s}_preresidual_avg'.format(basic_layer_name)
        this_layer_object = architecture_utils.get_3d_pooling_layer(
            num_rows_in_window=1,
            num_columns_in_window=1,
            num_heights_in_window=num_time_steps,
            num_rows_per_stride=1,
            num_columns_per_stride=1,
            num_heights_per_stride=num_time_steps,
            pooling_type_string=architecture_utils.MEAN_POOLING_STRING,
            layer_name=this_name
        )(input_layer_object)

        new_dims = (
            this_layer_object.shape[1:3] +
            (this_layer_object.shape[-1],)
        )

        this_name = '{0:s}_preresidual_squeeze'.format(basic_layer_name)
        this_layer_object = keras.layers.Reshape(
            target_shape=new_dims, name=this_name
        )(this_layer_object)

        this_name = '{0:s}_residual'.format(basic_layer_name)

        if dropout_rate > 0:
            current_layer_object = StochasticDepth(
                survival_prob=1. - dropout_rate, name=this_name
            )([this_layer_object, current_layer_object])
        else:
            current_layer_object = keras.layers.Add(name=this_name)([
                this_layer_object, current_layer_object
            ])

    return current_layer_object


def _get_channel_counts_for_skip_cnxn(input_layer_objects, num_output_channels):
    """Determines number of channels for each input layer to skip connection.

    A = number of input layers.

    :param input_layer_objects: length-A list of input layers (instances of
        subclass of `keras.layers`).
    :param num_output_channels: Number of desired output channels (after
        concatenation).
    :return: desired_channel_counts: length-A numpy array with number of
        desired channels for each input layer.
    """

    current_channel_counts = numpy.array(
        [__dimension_to_int(l.shape[-1]) for l in input_layer_objects],
        dtype=float
    )

    num_input_layers = len(input_layer_objects)
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


def _create_skip_connection(input_layer_objects, num_output_channels,
                            current_level_num, regularizer_object):
    """Creates skip connection.

    :param input_layer_objects: 1-D list of input layers (instances of subclass
        of `keras.layers`).
    :param num_output_channels: Desired number of output channels.
    :param current_level_num: Current level in Chiu-net++ architecture.  This
        should be a zero-based integer index.
    :param regularizer_object: Regularizer for conv layers (instance of
        `keras.regularizers.l1_l2` or similar).
    :return: concat_layer_object: Instance of `keras.layers.Concatenate`.
    """

    desired_input_channel_counts = _get_channel_counts_for_skip_cnxn(
        input_layer_objects=input_layer_objects,
        num_output_channels=num_output_channels
    )
    current_width = len(input_layer_objects) - 1

    for j in range(current_width):
        this_name = 'block{0:d}-{1:d}_preskipconv{2:d}'.format(
            current_level_num, current_width, j
        )

        input_layer_objects[j] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=1,
            num_kernel_columns=1,
            num_rows_per_stride=1,
            num_columns_per_stride=1,
            num_filters=desired_input_channel_counts[j],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )(input_layer_objects[j])

    this_name = 'block{0:d}-{1:d}_skip'.format(current_level_num, current_width)
    return keras.layers.Concatenate(axis=-1, name=this_name)(
        input_layer_objects
    )


def _get_2d_conv_block(
        input_layer_object, do_residual, do_convnext,
        num_conv_layers, filter_size_px, num_filters, do_time_distributed_conv,
        regularizer_object,
        activation_function_name, activation_function_alpha,
        dropout_rates, use_batch_norm, basic_layer_name):
    """Creates convolutional block for data with 2 spatial dimensions.

    L = number of conv layers

    :param input_layer_object: Input layer to block.
    :param do_residual: Boolean flag.  If True, this will be a residual block.
    :param do_convnext: Boolean flag.  If True, this will be a ConvNext block.
        If `do_residual == do_convnext == False`, this will be a basic
        convolutional block.
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
    :return: output_layer_object: Output layer from block.
    """

    # Check input args.
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

    # Handle ConvNext block.
    if do_convnext:
        current_layer_object = __get_2d_convnext_block(
            input_layer_object=input_layer_object,
            num_conv_layers=1,
            filter_size_px=filter_size_px,
            num_filters=num_filters,
            do_time_distributed_conv=do_time_distributed_conv,
            regularizer_object=regularizer_object,
            do_activation=activation_function_name is not None,
            dropout_rate=dropout_rates[0],
            basic_layer_name=(
                basic_layer_name if num_conv_layers == 1
                else basic_layer_name + '_0'
            )
        )

        for i in range(num_conv_layers):
            if i == 0:
                continue

            current_layer_object = __get_2d_convnext_block(
                input_layer_object=input_layer_object,
                num_conv_layers=1,
                filter_size_px=filter_size_px,
                num_filters=num_filters,
                do_time_distributed_conv=do_time_distributed_conv,
                regularizer_object=regularizer_object,
                do_activation=activation_function_name is not None,
                dropout_rate=dropout_rates[0],
                basic_layer_name='{0:s}_{1:d}'.format(basic_layer_name, i)
            )

        return current_layer_object

    # Create residual block or basic conv block.
    current_layer_object = None

    for i in range(num_conv_layers):
        if i == 0:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = current_layer_object

        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)
        current_layer_object = architecture_utils.get_2d_conv_layer(
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
            current_layer_object = keras.layers.TimeDistributed(
                current_layer_object, name=this_name
            )(this_input_layer_object)
        else:
            current_layer_object = current_layer_object(this_input_layer_object)

        if i == num_conv_layers - 1 and do_residual:
            if __dimension_to_int(input_layer_object.shape[-1]) == num_filters:
                new_layer_object = input_layer_object
            else:
                this_name = '{0:s}_preresidual_conv'.format(basic_layer_name)
                new_layer_object = architecture_utils.get_2d_conv_layer(
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
                    new_layer_object = keras.layers.TimeDistributed(
                        new_layer_object, name=this_name
                    )(input_layer_object)
                else:
                    new_layer_object = new_layer_object(input_layer_object)

            this_name = '{0:s}_residual'.format(basic_layer_name)
            current_layer_object = keras.layers.Add(name=this_name)([
                current_layer_object, new_layer_object
            ])

        if activation_function_name is not None:
            this_name = '{0:s}_activ{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_name,
                alpha_for_relu=activation_function_alpha,
                alpha_for_elu=activation_function_alpha,
                layer_name=this_name
            )(current_layer_object)

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


def _get_3d_conv_block(
        input_layer_object, do_residual, do_convnext,
        num_conv_layers, filter_size_px, regularizer_object,
        activation_function_name, activation_function_alpha,
        dropout_rates, use_batch_norm, basic_layer_name):
    """Creates convolutional block for data with 3 spatial dimensions.

    :param input_layer_object: Input layer to block (with 3 spatial dims).
    :param do_residual: See documentation for `_get_2d_conv_block`.
    :param do_convnext: Same.
    :param num_conv_layers: Same.
    :param filter_size_px: Same.
    :param regularizer_object: Same.
    :param activation_function_name: Same.
    :param activation_function_alpha: Same.
    :param dropout_rates: Same.
    :param use_batch_norm: Same.
    :param basic_layer_name: Same.
    :return: output_layer_object: Output layer from block (with 2 spatial dims).
    """

    # Check input args.
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

    # Handle ConvNext block.
    num_time_steps = __dimension_to_int(input_layer_object.shape[-2])

    if do_convnext:
        current_layer_object = __get_3d_convnext_block(
            input_layer_object=input_layer_object,
            num_conv_layers=1,
            filter_size_px=filter_size_px,
            num_time_steps=num_time_steps,
            regularizer_object=regularizer_object,
            do_activation=activation_function_name is not None,
            dropout_rate=dropout_rates[0],
            basic_layer_name=(
                basic_layer_name if num_conv_layers == 1
                else basic_layer_name + '_0'
            )
        )

        for i in range(num_conv_layers):
            if i == 0:
                continue

            current_layer_object = __get_3d_convnext_block(
                input_layer_object=input_layer_object,
                num_conv_layers=1,
                filter_size_px=filter_size_px,
                num_time_steps=num_time_steps,
                regularizer_object=regularizer_object,
                do_activation=activation_function_name is not None,
                dropout_rate=dropout_rates[0],
                basic_layer_name='{0:s}_{1:d}'.format(basic_layer_name, i)
            )

        return current_layer_object

    # Create residual block or basic conv block.
    current_layer_object = None
    num_filters = __dimension_to_int(input_layer_object.shape[-1])

    for i in range(num_conv_layers):
        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)

        if i == 0:
            current_layer_object = architecture_utils.get_3d_conv_layer(
                num_kernel_rows=filter_size_px,
                num_kernel_columns=filter_size_px,
                num_kernel_heights=num_time_steps,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_heights_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.NO_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )(input_layer_object)

            new_dims = (
                __dimension_to_int(current_layer_object.shape[1]),
                __dimension_to_int(current_layer_object.shape[2]),
                __dimension_to_int(current_layer_object.shape[-1])
            )

            this_name = '{0:s}_remove-time-dim'.format(basic_layer_name)
            current_layer_object = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(current_layer_object)
        else:
            current_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=filter_size_px,
                num_kernel_columns=filter_size_px,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(current_layer_object)

        if i == num_conv_layers - 1 and do_residual:
            this_name = '{0:s}_preresidual_avg'.format(basic_layer_name)
            this_layer_object = architecture_utils.get_3d_pooling_layer(
                num_rows_in_window=1,
                num_columns_in_window=1,
                num_heights_in_window=num_time_steps,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_heights_per_stride=num_time_steps,
                pooling_type_string=architecture_utils.MEAN_POOLING_STRING,
                layer_name=this_name
            )(input_layer_object)

            new_dims = (
                __dimension_to_int(this_layer_object.shape[1]),
                __dimension_to_int(this_layer_object.shape[2]),
                __dimension_to_int(this_layer_object.shape[-1])
            )

            this_name = '{0:s}_preresidual_squeeze'.format(basic_layer_name)
            this_layer_object = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(this_layer_object)

            this_name = '{0:s}_residual'.format(basic_layer_name)
            current_layer_object = keras.layers.Add(name=this_name)([
                current_layer_object, this_layer_object
            ])

        this_name = '{0:s}_activ{1:d}'.format(basic_layer_name, i)
        current_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=activation_function_name,
            alpha_for_relu=activation_function_alpha,
            alpha_for_elu=activation_function_alpha,
            layer_name=this_name
        )(current_layer_object)

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


def _pad_2d_layer(source_layer_object, target_layer_object, padding_layer_name):
    """Pads layer with 2 spatial dimensions.

    :param source_layer_object: Source layer.
    :param target_layer_object: Target layer.  The source layer will be padded,
        if necessary, to have the same dimensions as the target layer.
    :param padding_layer_name: Name of padding layer.
    :return: source_layer_object: Same as input, except maybe with different
        spatial dimensions.
    """

    num_source_rows = __dimension_to_int(source_layer_object.shape[1])
    num_target_rows = __dimension_to_int(target_layer_object.shape[1])
    num_padding_rows = num_target_rows - num_source_rows

    num_source_columns = __dimension_to_int(source_layer_object.shape[2])
    num_target_columns = __dimension_to_int(target_layer_object.shape[2])
    num_padding_columns = num_target_columns - num_source_columns

    if num_padding_rows + num_padding_columns > 0:
        padding_arg = ((0, num_padding_rows), (0, num_padding_columns))

        return keras.layers.ZeroPadding2D(
            padding=padding_arg, name=padding_layer_name
        )(source_layer_object)

    return source_layer_object


def create_model(option_dict, omit_model_summary=False):
    """Creates Chiu-net++.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    :param option_dict: See doc for `chiu_net_architecture.check_args`.
    :param omit_model_summary: Boolean flag.
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
    error_checking.assert_is_boolean(omit_model_summary)

    input_dimensions_gfs_3d = option_dict[GFS_3D_DIMENSIONS_KEY]
    input_dimensions_gfs_2d = option_dict[GFS_2D_DIMENSIONS_KEY]
    input_dimensions_era5 = option_dict[ERA5_CONST_DIMENSIONS_KEY]
    input_dimensions_lagged_target = option_dict[LAGTGT_DIMENSIONS_KEY]
    input_dimensions_predn_baseline = option_dict[PREDN_BASELINE_DIMENSIONS_KEY]
    use_residual_blocks = option_dict[USE_RESIDUAL_BLOCKS_KEY]
    use_convnext_blocks = option_dict[USE_CONVNEXT_BLOCKS_KEY]
    use_lead_time_as_predictor = option_dict[USE_LEAD_TIME_AS_PRED_KEY]
    dmc_dc_isi_indices_for_constraints = option_dict[CONSTRAINT_INDICES_KEY]

    # TODO(thunderhoser): Error-checking for this input arg and other keys
    # not defined in chiu_net_architecture.py.
    num_free_target_fields = option_dict[NUM_FREE_TARGETS_KEY]

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
    loss_function = option_dict[LOSS_FUNCTION_KEY]
    metric_functions = option_dict[METRIC_FUNCTIONS_KEY]

    num_gfs_lead_times = None
    num_grid_rows = None
    num_grid_columns = None

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

        num_grid_rows = input_dimensions_gfs_3d[0]
        num_grid_columns = input_dimensions_gfs_3d[1]
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

        num_grid_rows = input_dimensions_gfs_2d[0]
        num_grid_columns = input_dimensions_gfs_2d[1]
        num_gfs_lead_times = input_dimensions_gfs_2d[-2]

    layer_objects_gfs = [layer_object_gfs_3d, layer_object_gfs_2d]
    layer_objects_gfs = [l for l in layer_objects_gfs if l is not None]

    if len(layer_objects_gfs) > 1:
        layer_object_gfs = keras.layers.Concatenate(
            axis=-1, name='gfs_concat-2d-and-3d'
        )(layer_objects_gfs)
    elif len(layer_objects_gfs) == 1:
        layer_object_gfs = layer_objects_gfs[0]
    else:
        layer_object_gfs = None

    num_target_lag_times = None

    if input_dimensions_lagged_target is None:
        input_layer_object_lagged_target = None
        layer_object_lagged_target = None
    else:
        input_layer_object_lagged_target = keras.layers.Input(
            shape=tuple(input_dimensions_lagged_target.tolist()),
            name='lagged_target_inputs'
        )
        layer_object_lagged_target = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='lagged_targets_put-time-first'
        )(input_layer_object_lagged_target)

        num_grid_rows = input_dimensions_lagged_target[0]
        num_grid_columns = input_dimensions_lagged_target[1]
        num_target_lag_times = input_dimensions_lagged_target[-2]

    if use_lead_time_as_predictor:
        input_layer_object_lead_time = keras.layers.Input(
            shape=(1,), name='lead_time'
        )

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
    else:
        input_layer_object_lead_time = None
        layer_object_lead_time = None

    if input_dimensions_predn_baseline is None:
        input_layer_object_predn_baseline = None
    else:
        input_layer_object_predn_baseline = keras.layers.Input(
            shape=tuple(input_dimensions_predn_baseline.tolist()),
            name='predn_baseline_inputs'
        )

    if input_dimensions_era5 is None:
        input_layer_object_era5 = None
    else:
        input_layer_object_era5 = keras.layers.Input(
            shape=input_dimensions_era5, name='era5_inputs'
        )

    if input_layer_object_era5 is not None and layer_object_lead_time is not None:
        layer_object_constants = keras.layers.Concatenate(
            axis=-1, name='concat_constants'
        )(
            [input_layer_object_era5, layer_object_lead_time]
        )
        new_dims = (
            1, num_grid_rows, num_grid_columns, 1 + input_dimensions_era5[-1]
        )
    elif input_layer_object_era5 is not None:
        layer_object_constants = input_layer_object_era5
        new_dims = (
            1, num_grid_rows, num_grid_columns, input_dimensions_era5[-1]
        )
    elif layer_object_lead_time is not None:
        layer_object_constants = layer_object_lead_time
        new_dims = (1, num_grid_rows, num_grid_columns, 1)
    else:
        layer_object_constants = None
        new_dims = ()

    if layer_object_constants is not None:
        layer_object_constants = keras.layers.Reshape(
            target_shape=new_dims, name='const_add-time-dim'
        )(layer_object_constants)

        if layer_object_gfs is not None:
            this_layer_object = keras.layers.Concatenate(
                axis=-4, name='const_add-gfs-times'
            )(
                num_gfs_lead_times * [layer_object_constants]
            )

            layer_object_gfs = keras.layers.Concatenate(
                axis=-1, name='gfs_concat-const'
            )(
                [layer_object_gfs, this_layer_object]
            )

        if input_dimensions_lagged_target is not None:
            this_layer_object = keras.layers.Concatenate(
                axis=-4, name='const_add-lag-times'
            )(
                num_target_lag_times * [layer_object_constants]
            )

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
    loop_max = 0 if layer_object_gfs is None else num_levels + 1

    for i in range(loop_max):
        if i == 0:
            this_input_layer_object = layer_object_gfs
        else:
            this_input_layer_object = gfs_encoder_pooling_layer_objects[i - 1]

        gfs_encoder_conv_layer_objects[i] = _get_2d_conv_block(
            input_layer_object=this_input_layer_object,
            do_residual=use_residual_blocks,
            do_convnext=use_convnext_blocks,
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

        this_name = 'gfs_fcst_level{0:d}_put-time-last'.format(i)
        gfs_fcst_module_layer_objects[i] = keras.layers.Permute(
            dims=(2, 3, 1, 4), name=this_name
        )(gfs_encoder_conv_layer_objects[i])

        if num_gfs_lead_times > 1:
            orig_dims = gfs_fcst_module_layer_objects[i].shape
            orig_dims = numpy.array([__dimension_to_int(d) for d in orig_dims], dtype=int)
            new_dims = orig_dims[1:-2] + (orig_dims[-1],)

            this_name = 'gfs_fcst_level{0:d}_remove-time-dim'.format(i)
            gfs_fcst_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(gfs_fcst_module_layer_objects[i])

        elif gfs_fcst_use_3d_conv:
            gfs_fcst_module_layer_objects[i] = _get_3d_conv_block(
                input_layer_object=gfs_fcst_module_layer_objects[i],
                do_residual=use_residual_blocks,
                do_convnext=use_convnext_blocks,
                num_conv_layers=gfs_fcst_num_conv_layers,
                filter_size_px=1,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=gfs_fcst_dropout_rates,
                use_batch_norm=use_batch_normalization,
                basic_layer_name='gfs_fcst_level{0:d}'.format(i)
            )
        else:
            orig_dims = gfs_fcst_module_layer_objects[i].shape
            orig_dims = numpy.array([__dimension_to_int(d) for d in orig_dims], dtype=int)
            new_dims = orig_dims[1:-2] + (orig_dims[-2] * orig_dims[-1],)

            this_name = 'gfs_fcst_level{0:d}_remove-time-dim'.format(i)
            gfs_fcst_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(gfs_fcst_module_layer_objects[i])

            gfs_fcst_module_layer_objects[i] = _get_2d_conv_block(
                input_layer_object=gfs_fcst_module_layer_objects[i],
                do_residual=use_residual_blocks,
                do_convnext=use_convnext_blocks,
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
    loop_max = 0 if input_dimensions_lagged_target is None else num_levels + 1

    for i in range(loop_max):
        if i == 0:
            this_input_layer_object = layer_object_lagged_target
        else:
            this_input_layer_object = (
                lagtgt_encoder_pooling_layer_objects[i - 1]
            )

        lagtgt_encoder_conv_layer_objects[i] = _get_2d_conv_block(
            input_layer_object=this_input_layer_object,
            do_residual=use_residual_blocks,
            do_convnext=use_convnext_blocks,
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

        this_name = 'lagtgt_fcst_level{0:d}_put-time-last'.format(i)
        lagtgt_fcst_module_layer_objects[i] = keras.layers.Permute(
            dims=(2, 3, 1, 4), name=this_name
        )(lagtgt_encoder_conv_layer_objects[i])

        if num_target_lag_times > 1:
            orig_dims = lagtgt_fcst_module_layer_objects[i].shape
            print(orig_dims)
            orig_dims = numpy.array([__dimension_to_int(d) for d in orig_dims], dtype=int)
            new_dims = orig_dims[1:-2] + (orig_dims[-1],)

            this_name = 'lagtgt_fcst_level{0:d}_remove-time-dim'.format(i)
            lagtgt_fcst_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(lagtgt_fcst_module_layer_objects[i])

        elif lagtgt_fcst_use_3d_conv:
            lagtgt_fcst_module_layer_objects[i] = _get_3d_conv_block(
                input_layer_object=lagtgt_fcst_module_layer_objects[i],
                do_residual=use_residual_blocks,
                do_convnext=use_convnext_blocks,
                num_conv_layers=lagtgt_fcst_num_conv_layers,
                filter_size_px=1,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=lagtgt_fcst_dropout_rates,
                use_batch_norm=use_batch_normalization,
                basic_layer_name='lagtgt_fcst_level{0:d}'.format(i)
            )
        else:
            orig_dims = lagtgt_fcst_module_layer_objects[i].shape
            new_dims = orig_dims[1:-2] + (orig_dims[-2] * orig_dims[-1],)

            this_name = 'lagtgt_fcst_level{0:d}_remove-time-dim'.format(i)
            lagtgt_fcst_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(lagtgt_fcst_module_layer_objects[i])

            lagtgt_fcst_module_layer_objects[i] = _get_2d_conv_block(
                input_layer_object=lagtgt_fcst_module_layer_objects[i],
                do_residual=use_residual_blocks,
                do_convnext=use_convnext_blocks,
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

        these_layer_objects = [
            gfs_fcst_module_layer_objects[i],
            lagtgt_fcst_module_layer_objects[i]
        ]
        these_layer_objects = [l for l in these_layer_objects if l is not None]

        if len(these_layer_objects) > 1:
            last_conv_layer_matrix[i, 0] = keras.layers.Concatenate(
                axis=-1, name=this_name
            )(these_layer_objects)
        else:
            last_conv_layer_matrix[i, 0] = these_layer_objects[0]

        i_new = i + 0
        j = 0

        while i_new > 0:
            i_new -= 1
            j += 1

            this_name = 'block{0:d}-{1:d}_upsampling'.format(i_new, j)
            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2), name=this_name
            )(last_conv_layer_matrix[i_new + 1, j - 1])

            this_layer_object = _pad_2d_layer(
                source_layer_object=this_layer_object,
                target_layer_object=last_conv_layer_matrix[i_new, 0],
                padding_layer_name='block{0:d}-{1:d}_padding'.format(i_new, j)
            )

            this_num_channels = int(numpy.round(
                0.5 * decoder_num_channels_by_level[i_new]
            ))

            last_conv_layer_matrix[i_new, j] = _get_2d_conv_block(
                input_layer_object=this_layer_object,
                do_residual=use_residual_blocks,
                do_convnext=use_convnext_blocks,
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

            last_conv_layer_matrix[i_new, j] = _create_skip_connection(
                input_layer_objects=
                last_conv_layer_matrix[i_new, :(j + 1)].tolist(),
                num_output_channels=decoder_num_channels_by_level[i_new],
                current_level_num=i_new,
                regularizer_object=regularizer_object
            )

            last_conv_layer_matrix[i_new, j] = _get_2d_conv_block(
                input_layer_object=last_conv_layer_matrix[i_new, j],
                do_residual=use_residual_blocks,
                do_convnext=use_convnext_blocks,
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

    if include_penultimate_conv:
        last_conv_layer_matrix[0, -1] = _get_2d_conv_block(
            input_layer_object=last_conv_layer_matrix[0, -1],
            do_residual=use_residual_blocks,
            do_convnext=use_convnext_blocks,
            num_conv_layers=1,
            filter_size_px=3,
            num_filters=2 * num_free_target_fields * ensemble_size,
            do_time_distributed_conv=False,
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=penultimate_conv_dropout_rate,
            use_batch_norm=use_batch_normalization,
            basic_layer_name='penultimate'
        )

    output_layer_object = _get_2d_conv_block(
        input_layer_object=last_conv_layer_matrix[0, -1],
        do_residual=use_residual_blocks,
        do_convnext=use_convnext_blocks,
        num_conv_layers=1,
        filter_size_px=1,
        num_filters=num_free_target_fields * ensemble_size,
        do_time_distributed_conv=False,
        regularizer_object=regularizer_object,
        activation_function_name=None,
        activation_function_alpha=output_activ_function_alpha,
        dropout_rates=-1.,
        use_batch_norm=False,
        basic_layer_name='output'
    )

    if ensemble_size > 1:
        new_dims = (
            num_grid_rows,
            num_grid_columns,
            num_free_target_fields,
            ensemble_size
        )
        output_layer_object = keras.layers.Reshape(
            target_shape=new_dims, name='reshape_output'
        )(output_layer_object)

    if input_layer_object_predn_baseline is not None:
        layer_object_predn_baseline = (
            input_layer_object_predn_baseline[..., :num_free_target_fields]
        )

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
            )(layer_object_predn_baseline)

        output_layer_object = keras.layers.Add(name='output_add_baseline')([
            output_layer_object, layer_object_predn_baseline
        ])

    if output_activ_function_name is not None:
        output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=output_activ_function_name,
            alpha_for_relu=output_activ_function_alpha,
            alpha_for_elu=output_activ_function_alpha,
            layer_name='output_activation'
        )(output_layer_object)

    if dmc_dc_isi_indices_for_constraints is not None:
        output_layer_object = DeriveBUIPredictions(
            dmc_index=dmc_dc_isi_indices_for_constraints[0],
            dc_index=dmc_dc_isi_indices_for_constraints[1],
            expect_ensemble=ensemble_size > 1,
            name='constrain_bui'
        )(output_layer_object)

        output_layer_object = DeriveFWIPredictions(
            isi_index=dmc_dc_isi_indices_for_constraints[2],
            bui_index=-1,
            expect_ensemble=ensemble_size > 1,
            name='constrain_fwi'
        )(output_layer_object)

        output_layer_object = DeriveDSRPredictions(
            fwi_index=-1,
            expect_ensemble=ensemble_size > 1,
            name='constrain_dsr'
        )(output_layer_object)

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
        metrics=metric_functions
    )

    if not omit_model_summary:
        model_object.summary()

    return model_object
