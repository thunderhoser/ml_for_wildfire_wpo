"""Makes templates for Experiment 7 with seven target variables."""

import os
import sys
import copy
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import custom_losses
import custom_metrics
import neural_net
import chiu_net_pp_architecture as chiu_net_pp_arch
import architecture_utils
import file_system_utils

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_models/'
    'experiment07_bigger-batches_7var/templates'
)

CHANNEL_WEIGHTS = numpy.array([0.00633568, 0.00029226, 0.00000163, 0.79002088, 0.00018185, 0.04245499, 0.16071271])
MAX_DUAL_WEIGHTS = numpy.array([96.4105, 303.9126, 1741.9033, 23.1660, 361.6984, 61.3669, 40.1856])

LOSS_FUNCTION = custom_losses.dual_weighted_mse(
    channel_weights=CHANNEL_WEIGHTS,
    max_dual_weight_by_channel=MAX_DUAL_WEIGHTS,
    expect_ensemble=False,
    function_name='loss_dwmse'
)

LOSS_FUNCTION_STRING = (
    'custom_losses.dual_weighted_mse('
    'channel_weights=numpy.array([0.00633568, 0.00029226, 0.00000163, 0.79002088, 0.00018185, 0.04245499, 0.16071271]), '
    'max_dual_weight_by_channel=numpy.array([96.4105, 303.9126, 1741.9033, 23.1660, 361.6984, 61.3669, 40.1856]), '
    'expect_ensemble=False, '
    'function_name="loss_dwmse")'
)

METRIC_FUNCTIONS = [
    custom_metrics.max_prediction_unmasked(channel_index=0, expect_ensemble=False, function_name='ffmc_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=1, expect_ensemble=False, function_name='dmc_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=2, expect_ensemble=False, function_name='dc_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=3, expect_ensemble=False, function_name='isi_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=4, expect_ensemble=False, function_name='bui_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=5, expect_ensemble=False, function_name='fwi_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=6, expect_ensemble=False, function_name='dsr_max_prediction'),
    custom_metrics.max_prediction_anywhere(channel_index=0, expect_ensemble=False, function_name='ffmc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=1, expect_ensemble=False, function_name='dmc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=2, expect_ensemble=False, function_name='dc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=3, expect_ensemble=False, function_name='isi_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=4, expect_ensemble=False, function_name='bui_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=5, expect_ensemble=False, function_name='fwi_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=6, expect_ensemble=False, function_name='dsr_max_prediction_anywhere'),
    custom_metrics.mean_squared_error_unmasked(channel_index=0, expect_ensemble=False, function_name='ffmc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=1, expect_ensemble=False, function_name='dmc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=2, expect_ensemble=False, function_name='dc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=3, expect_ensemble=False, function_name='isi_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=4, expect_ensemble=False, function_name='bui_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=5, expect_ensemble=False, function_name='fwi_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=6, expect_ensemble=False, function_name='dsr_mse'),
    custom_metrics.mean_squared_error_anywhere(channel_index=0, expect_ensemble=False, function_name='ffmc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=1, expect_ensemble=False, function_name='dmc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=2, expect_ensemble=False, function_name='dc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=3, expect_ensemble=False, function_name='isi_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=4, expect_ensemble=False, function_name='bui_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=5, expect_ensemble=False, function_name='fwi_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=6, expect_ensemble=False, function_name='dsr_mse_anywhere'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=0, expect_ensemble=False, function_name='ffmc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=1, expect_ensemble=False, function_name='dmc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=2, expect_ensemble=False, function_name='dc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=3, expect_ensemble=False, function_name='isi_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=4, expect_ensemble=False, function_name='bui_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=5, expect_ensemble=False, function_name='fwi_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=6, expect_ensemble=False, function_name='dsr_dwmse'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=0, expect_ensemble=False, function_name='ffmc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=1, expect_ensemble=False, function_name='dmc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=2, expect_ensemble=False, function_name='dc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=3, expect_ensemble=False, function_name='isi_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=4, expect_ensemble=False, function_name='bui_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=5, expect_ensemble=False, function_name='fwi_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=6, expect_ensemble=False, function_name='dsr_dwmse_anywhere'),
    custom_losses.dual_weighted_mse_1channel(channel_index=0, channel_weight=0.00633568, max_dual_weight=96.4105, expect_ensemble=False, function_name='ffmc_dwmse_in_loss'),
    custom_losses.dual_weighted_mse_1channel(channel_index=1, channel_weight=0.00029226, max_dual_weight=303.9126, expect_ensemble=False, function_name='dmc_dwmse_in_loss'),
    custom_losses.dual_weighted_mse_1channel(channel_index=2, channel_weight=0.00000163, max_dual_weight=1741.9033, expect_ensemble=False, function_name='dc_dwmse_in_loss'),
    custom_losses.dual_weighted_mse_1channel(channel_index=3, channel_weight=0.79002088, max_dual_weight=23.1660, expect_ensemble=False, function_name='isi_dwmse_in_loss'),
    custom_losses.dual_weighted_mse_1channel(channel_index=4, channel_weight=0.00018185, max_dual_weight=361.6984, expect_ensemble=False, function_name='bui_dwmse_in_loss'),
    custom_losses.dual_weighted_mse_1channel(channel_index=5, channel_weight=0.04245499, max_dual_weight=61.3669, expect_ensemble=False, function_name='fwi_dwmse_in_loss'),
    custom_losses.dual_weighted_mse_1channel(channel_index=6, channel_weight=0.16071271, max_dual_weight=40.1856, expect_ensemble=False, function_name='dsr_dwmse_in_loss')
]

METRIC_FUNCTION_STRINGS = [
    'custom_metrics.max_prediction_unmasked(channel_index=0, expect_ensemble=False, function_name="ffmc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=1, expect_ensemble=False, function_name="dmc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=2, expect_ensemble=False, function_name="dc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=3, expect_ensemble=False, function_name="isi_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=4, expect_ensemble=False, function_name="bui_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=5, expect_ensemble=False, function_name="fwi_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=6, expect_ensemble=False, function_name="dsr_max_prediction")',
    'custom_metrics.max_prediction_anywhere(channel_index=0, expect_ensemble=False, function_name="ffmc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=1, expect_ensemble=False, function_name="dmc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=2, expect_ensemble=False, function_name="dc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=3, expect_ensemble=False, function_name="isi_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=4, expect_ensemble=False, function_name="bui_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=5, expect_ensemble=False, function_name="fwi_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=6, expect_ensemble=False, function_name="dsr_max_prediction_anywhere")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=0, expect_ensemble=False, function_name="ffmc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=1, expect_ensemble=False, function_name="dmc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=2, expect_ensemble=False, function_name="dc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=3, expect_ensemble=False, function_name="isi_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=4, expect_ensemble=False, function_name="bui_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=5, expect_ensemble=False, function_name="fwi_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=6, expect_ensemble=False, function_name="dsr_mse")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=0, expect_ensemble=False, function_name="ffmc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=1, expect_ensemble=False, function_name="dmc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=2, expect_ensemble=False, function_name="dc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=3, expect_ensemble=False, function_name="isi_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=4, expect_ensemble=False, function_name="bui_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=5, expect_ensemble=False, function_name="fwi_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=6, expect_ensemble=False, function_name="dsr_mse_anywhere")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=0, expect_ensemble=False, function_name="ffmc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=1, expect_ensemble=False, function_name="dmc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=2, expect_ensemble=False, function_name="dc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=3, expect_ensemble=False, function_name="isi_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=4, expect_ensemble=False, function_name="bui_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=5, expect_ensemble=False, function_name="fwi_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=6, expect_ensemble=False, function_name="dsr_dwmse")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=0, expect_ensemble=False, function_name="ffmc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=1, expect_ensemble=False, function_name="dmc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=2, expect_ensemble=False, function_name="dc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=3, expect_ensemble=False, function_name="isi_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=4, expect_ensemble=False, function_name="bui_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=5, expect_ensemble=False, function_name="fwi_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=6, expect_ensemble=False, function_name="dsr_dwmse_anywhere")',
    'custom_losses.dual_weighted_mse_1channel(channel_index=0, channel_weight=0.00633568, max_dual_weight=96.4105, expect_ensemble=False, function_name="ffmc_dwmse_in_loss")',
    'custom_losses.dual_weighted_mse_1channel(channel_index=1, channel_weight=0.00029226, max_dual_weight=303.9126, expect_ensemble=False, function_name="dmc_dwmse_in_loss")',
    'custom_losses.dual_weighted_mse_1channel(channel_index=2, channel_weight=0.00000163, max_dual_weight=1741.9033, expect_ensemble=False, function_name="dc_dwmse_in_loss")',
    'custom_losses.dual_weighted_mse_1channel(channel_index=3, channel_weight=0.79002088, max_dual_weight=23.1660, expect_ensemble=False, function_name="isi_dwmse_in_loss")',
    'custom_losses.dual_weighted_mse_1channel(channel_index=4, channel_weight=0.00018185, max_dual_weight=361.6984, expect_ensemble=False, function_name="bui_dwmse_in_loss")',
    'custom_losses.dual_weighted_mse_1channel(channel_index=5, channel_weight=0.04245499, max_dual_weight=61.3669, expect_ensemble=False, function_name="fwi_dwmse_in_loss")',
    'custom_losses.dual_weighted_mse_1channel(channel_index=6, channel_weight=0.16071271, max_dual_weight=40.1856, expect_ensemble=False, function_name="dsr_dwmse_in_loss")'
]

NUM_CONV_LAYERS_PER_BLOCK = 2
NUM_GFS_LEAD_TIMES = 2

DEFAULT_OPTION_DICT = {
    chiu_net_pp_arch.GFS_3D_DIMENSIONS_KEY: numpy.array(
        [265, 537, 2, NUM_GFS_LEAD_TIMES, 5], dtype=int
    ),
    chiu_net_pp_arch.GFS_2D_DIMENSIONS_KEY: numpy.array(
        [265, 537, NUM_GFS_LEAD_TIMES, 7], dtype=int
    ),
    # chiu_net_pp_arch.ERA5_CONST_DIMENSIONS_KEY: None,
    chiu_net_pp_arch.ERA5_CONST_DIMENSIONS_KEY: numpy.array(
        [265, 537, 7], dtype=int
    ),
    chiu_net_pp_arch.LAGTGT_DIMENSIONS_KEY: numpy.array(
        [265, 537, 6, 7], dtype=int
    ),
    chiu_net_pp_arch.GFS_FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    chiu_net_pp_arch.GFS_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    chiu_net_pp_arch.GFS_FC_MODULE_USE_3D_CONV: True,
    chiu_net_pp_arch.LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    chiu_net_pp_arch.LAGTGT_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    chiu_net_pp_arch.LAGTGT_FC_MODULE_USE_3D_CONV: True,
    chiu_net_pp_arch.NUM_LEVELS_KEY: 6,
    chiu_net_pp_arch.GFS_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(
        7, NUM_CONV_LAYERS_PER_BLOCK, dtype=int
    ),
    # chiu_net_pp_arch.GFS_ENCODER_NUM_CHANNELS_KEY:
    #     NUM_FIRST_LAYER_FILTERS * numpy.array([1, 2, 3, 4, 5, 6, 7], dtype=int),
    chiu_net_pp_arch.GFS_ENCODER_NUM_CHANNELS_KEY:
        numpy.array([10, 15, 20, 25, 30, 35, 40], dtype=int),
    chiu_net_pp_arch.GFS_ENCODER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(
        7, NUM_CONV_LAYERS_PER_BLOCK, dtype=int
    ),
    chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY:
        numpy.array([10, 12, 14, 16, 18, 20, 22], dtype=int),
    chiu_net_pp_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    chiu_net_pp_arch.DECODER_NUM_CONV_LAYERS_KEY: numpy.full(6, 2, dtype=int),
    # chiu_net_pp_arch.DECODER_NUM_CHANNELS_KEY:
    #     int(numpy.round(1.25 * NUM_FIRST_LAYER_FILTERS)) *
    #     numpy.array([1, 2, 3, 4, 5, 6], dtype=int),
    chiu_net_pp_arch.DECODER_NUM_CHANNELS_KEY:
        numpy.array([20, 27, 34, 41, 48, 55], dtype=int),
    # chiu_net_pp_arch.UPSAMPLING_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    # chiu_net_pp_arch.SKIP_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    chiu_net_pp_arch.INCLUDE_PENULTIMATE_KEY: False,
    chiu_net_pp_arch.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    chiu_net_pp_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    chiu_net_pp_arch.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    chiu_net_pp_arch.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    chiu_net_pp_arch.L1_WEIGHT_KEY: 0.,
    chiu_net_pp_arch.L2_WEIGHT_KEY: 1e-6,
    chiu_net_pp_arch.USE_BATCH_NORM_KEY: True,
    chiu_net_pp_arch.ENSEMBLE_SIZE_KEY: 1
}

NUM_SKIP_LAYERS_WITH_DROPOUT = 3
NUM_UPCONV_LAYERS_WITH_DROPOUT = 3

BATCH_SIZES = numpy.array([24, 32, 40, 48], dtype=int)
DROPOUT_RATES = numpy.array([0.00, 0.05, 0.10, 0.15, 0.20])

# Subbatch size is 8.
BATCH_SIZE_TO_NUM_GRAD_ACCUM_STEPS = {
    24: 3,
    32: 4,
    40: 5,
    48: 6
}


def _run():
    """Makes templates for Experiment 7 with seven target variables.

    This is effectively the main method.
    """

    for i in range(len(BATCH_SIZES)):
        for k in range(len(DROPOUT_RATES)):
            option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

            num_grad_accum_steps = (
                BATCH_SIZE_TO_NUM_GRAD_ACCUM_STEPS[BATCH_SIZES[i]]
            )

            if num_grad_accum_steps == 1:
                optimizer_function = keras.optimizers.Nadam()
                optimizer_function_string = 'keras.optimizers.Nadam()'
            else:
                optimizer_function = keras.optimizers.Nadam(
                    gradient_accumulation_steps=num_grad_accum_steps
                )

                optimizer_function_string = (
                    'keras.optimizers.Nadam('
                    'gradient_accumulation_steps={0:d}'
                    ')'
                ).format(num_grad_accum_steps)

            upsampling_dropout_rates = numpy.full(6, 0.)
            skip_dropout_rates = numpy.full(6, 0.)
            upsampling_dropout_rates[:NUM_UPCONV_LAYERS_WITH_DROPOUT] = (
                DROPOUT_RATES[k]
            )
            skip_dropout_rates[:NUM_SKIP_LAYERS_WITH_DROPOUT] = (
                DROPOUT_RATES[k]
            )

            option_dict.update({
                chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY: optimizer_function,
                chiu_net_pp_arch.UPSAMPLING_DROPOUT_RATES_KEY:
                    upsampling_dropout_rates,
                chiu_net_pp_arch.SKIP_DROPOUT_RATES_KEY: skip_dropout_rates
            })

            model_object = chiu_net_pp_arch.create_model(
                option_dict=option_dict,
                loss_function=LOSS_FUNCTION,
                metric_list=METRIC_FUNCTIONS
            )

            output_file_name = (
                '{0:s}/batch-size={1:03d}_dropout-rate={2:.2f}/model.keras'
            ).format(
                OUTPUT_DIR_NAME,
                BATCH_SIZES[i],
                DROPOUT_RATES[k]
            )

            file_system_utils.mkdir_recursive_if_necessary(
                file_name=output_file_name
            )

            print('Writing model to: "{0:s}"...'.format(output_file_name))
            model_object.save(
                filepath=output_file_name, overwrite=True,
                include_optimizer=True
            )

            metafile_name = neural_net.find_metafile(
                model_file_name=output_file_name,
                raise_error_if_missing=False
            )
            option_dict[neural_net.LOSS_FUNCTION_KEY] = LOSS_FUNCTION_STRING
            option_dict[neural_net.METRIC_FUNCTIONS_KEY] = (
                METRIC_FUNCTION_STRINGS
            )
            option_dict[neural_net.OPTIMIZER_FUNCTION_KEY] = (
                optimizer_function_string
            )

            neural_net.write_metafile(
                pickle_file_name=metafile_name,
                num_epochs=100,
                num_training_batches_per_epoch=32,
                training_option_dict={},
                num_validation_batches_per_epoch=16,
                validation_option_dict={},
                loss_function_string=LOSS_FUNCTION_STRING,
                optimizer_function_string=optimizer_function_string,
                metric_function_strings=METRIC_FUNCTION_STRINGS,
                plateau_patience_epochs=10,
                plateau_learning_rate_multiplier=0.6,
                early_stopping_patience_epochs=50
            )


if __name__ == '__main__':
    _run()
