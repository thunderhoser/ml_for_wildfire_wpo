"""Tries making templates for recurrent evidential NN."""

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
import chiu_net_pp_architecture_recurrent as chiu_net_pp_arch
import architecture_utils
import file_system_utils

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_models/'
    'test_recurrent_enn_template'
)

CHANNEL_WEIGHTS = numpy.array([0.03502606, 0.00522109, 0.00012253, 0.83691845, 0.00334096, 0.11937091])
MAX_DUAL_WEIGHTS = numpy.array([96.4105, 303.9126, 1741.9033, 23.1660, 361.6984, 61.3669])

METRIC_FUNCTIONS = [
    custom_metrics.max_prediction_unmasked(channel_index=0, expect_ensemble=True, is_nn_evidential=False, function_name='ffmc_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=1, expect_ensemble=True, is_nn_evidential=False, function_name='dmc_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=2, expect_ensemble=True, is_nn_evidential=False, function_name='dc_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=3, expect_ensemble=True, is_nn_evidential=False, function_name='isi_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=4, expect_ensemble=True, is_nn_evidential=False, function_name='bui_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=5, expect_ensemble=True, is_nn_evidential=False, function_name='fwi_max_prediction'),
    custom_metrics.max_prediction_anywhere(channel_index=0, expect_ensemble=True, is_nn_evidential=False, function_name='ffmc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=1, expect_ensemble=True, is_nn_evidential=False, function_name='dmc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=2, expect_ensemble=True, is_nn_evidential=False, function_name='dc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=3, expect_ensemble=True, is_nn_evidential=False, function_name='isi_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=4, expect_ensemble=True, is_nn_evidential=False, function_name='bui_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=5, expect_ensemble=True, is_nn_evidential=False, function_name='fwi_max_prediction_anywhere'),
    custom_metrics.mean_squared_error_unmasked(channel_index=0, expect_ensemble=True, is_nn_evidential=False, function_name='ffmc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=1, expect_ensemble=True, is_nn_evidential=False, function_name='dmc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=2, expect_ensemble=True, is_nn_evidential=False, function_name='dc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=3, expect_ensemble=True, is_nn_evidential=False, function_name='isi_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=4, expect_ensemble=True, is_nn_evidential=False, function_name='bui_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=5, expect_ensemble=True, is_nn_evidential=False, function_name='fwi_mse'),
    custom_metrics.mean_squared_error_anywhere(channel_index=0, expect_ensemble=True, is_nn_evidential=False, function_name='ffmc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=1, expect_ensemble=True, is_nn_evidential=False, function_name='dmc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=2, expect_ensemble=True, is_nn_evidential=False, function_name='dc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=3, expect_ensemble=True, is_nn_evidential=False, function_name='isi_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=4, expect_ensemble=True, is_nn_evidential=False, function_name='bui_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=5, expect_ensemble=True, is_nn_evidential=False, function_name='fwi_mse_anywhere'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=0, expect_ensemble=True, is_nn_evidential=False, function_name='ffmc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=1, expect_ensemble=True, is_nn_evidential=False, function_name='dmc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=2, expect_ensemble=True, is_nn_evidential=False, function_name='dc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=3, expect_ensemble=True, is_nn_evidential=False, function_name='isi_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=4, expect_ensemble=True, is_nn_evidential=False, function_name='bui_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=5, expect_ensemble=True, is_nn_evidential=False, function_name='fwi_dwmse'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=0, expect_ensemble=True, is_nn_evidential=False, function_name='ffmc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=1, expect_ensemble=True, is_nn_evidential=False, function_name='dmc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=2, expect_ensemble=True, is_nn_evidential=False, function_name='dc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=3, expect_ensemble=True, is_nn_evidential=False, function_name='isi_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=4, expect_ensemble=True, is_nn_evidential=False, function_name='bui_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=5, expect_ensemble=True, is_nn_evidential=False, function_name='fwi_dwmse_anywhere')
]

METRIC_FUNCTION_STRINGS = [
    'custom_metrics.max_prediction_unmasked(channel_index=0, expect_ensemble=True, is_nn_evidential=False, function_name="ffmc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=1, expect_ensemble=True, is_nn_evidential=False, function_name="dmc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=2, expect_ensemble=True, is_nn_evidential=False, function_name="dc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=3, expect_ensemble=True, is_nn_evidential=False, function_name="isi_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=4, expect_ensemble=True, is_nn_evidential=False, function_name="bui_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=5, expect_ensemble=True, is_nn_evidential=False, function_name="fwi_max_prediction")',
    'custom_metrics.max_prediction_anywhere(channel_index=0, expect_ensemble=True, is_nn_evidential=False, function_name="ffmc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=1, expect_ensemble=True, is_nn_evidential=False, function_name="dmc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=2, expect_ensemble=True, is_nn_evidential=False, function_name="dc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=3, expect_ensemble=True, is_nn_evidential=False, function_name="isi_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=4, expect_ensemble=True, is_nn_evidential=False, function_name="bui_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=5, expect_ensemble=True, is_nn_evidential=False, function_name="fwi_max_prediction_anywhere")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=0, expect_ensemble=True, is_nn_evidential=False, function_name="ffmc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=1, expect_ensemble=True, is_nn_evidential=False, function_name="dmc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=2, expect_ensemble=True, is_nn_evidential=False, function_name="dc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=3, expect_ensemble=True, is_nn_evidential=False, function_name="isi_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=4, expect_ensemble=True, is_nn_evidential=False, function_name="bui_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=5, expect_ensemble=True, is_nn_evidential=False, function_name="fwi_mse")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=0, expect_ensemble=True, is_nn_evidential=False, function_name="ffmc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=1, expect_ensemble=True, is_nn_evidential=False, function_name="dmc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=2, expect_ensemble=True, is_nn_evidential=False, function_name="dc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=3, expect_ensemble=True, is_nn_evidential=False, function_name="isi_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=4, expect_ensemble=True, is_nn_evidential=False, function_name="bui_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=5, expect_ensemble=True, is_nn_evidential=False, function_name="fwi_mse_anywhere")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=0, expect_ensemble=True, is_nn_evidential=False, function_name="ffmc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=1, expect_ensemble=True, is_nn_evidential=False, function_name="dmc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=2, expect_ensemble=True, is_nn_evidential=False, function_name="dc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=3, expect_ensemble=True, is_nn_evidential=False, function_name="isi_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=4, expect_ensemble=True, is_nn_evidential=False, function_name="bui_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=5, expect_ensemble=True, is_nn_evidential=False, function_name="fwi_dwmse")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=0, expect_ensemble=True, is_nn_evidential=False, function_name="ffmc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=1, expect_ensemble=True, is_nn_evidential=False, function_name="dmc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=2, expect_ensemble=True, is_nn_evidential=False, function_name="dc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=3, expect_ensemble=True, is_nn_evidential=False, function_name="isi_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=4, expect_ensemble=True, is_nn_evidential=False, function_name="bui_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=5, expect_ensemble=True, is_nn_evidential=False, function_name="fwi_dwmse_anywhere")'
]

NUM_CONV_LAYERS_PER_BLOCK = 1

OPTIMIZER_FUNCTION = keras.optimizers.Nadam(gradient_accumulation_steps=24)
OPTIMIZER_FUNCTION_STRING = (
    'keras.optimizers.Nadam(gradient_accumulation_steps=24)'
)

DEFAULT_OPTION_DICT = {
    # chiu_net_pp_arch.GFS_3D_DIMENSIONS_KEY: numpy.array(
    #     [265, 537, 2, NUM_GFS_LEAD_TIMES, 5], dtype=int
    # ),
    # chiu_net_pp_arch.GFS_2D_DIMENSIONS_KEY: numpy.array(
    #     [265, 537, NUM_GFS_LEAD_TIMES, 7], dtype=int
    # ),
    chiu_net_pp_arch.ERA5_CONST_DIMENSIONS_KEY: numpy.array(
        [265, 537, 7], dtype=int
    ),
    # chiu_net_pp_arch.LAGTGT_DIMENSIONS_KEY: numpy.array(
    #     [265, 537, 6, 7], dtype=int
    # ),
    chiu_net_pp_arch.PREDN_BASELINE_DIMENSIONS_KEY: numpy.array(
        [265, 537, 6], dtype=int
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
    chiu_net_pp_arch.GFS_ENCODER_NUM_CHANNELS_KEY: numpy.array(
        [25, 37, 50, 62, 75, 87, 100], dtype=int
    ),
    chiu_net_pp_arch.GFS_ENCODER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(
        7, NUM_CONV_LAYERS_PER_BLOCK, dtype=int
    ),
    chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY: numpy.array(
        [15, 20, 25, 30, 35, 40, 45], dtype=int
    ),
    chiu_net_pp_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    chiu_net_pp_arch.DECODER_NUM_CONV_LAYERS_KEY: numpy.full(6, 2, dtype=int),
    chiu_net_pp_arch.DECODER_NUM_CHANNELS_KEY: numpy.array(
        [20, 28, 37, 46, 55, 63], dtype=int
    ),
    chiu_net_pp_arch.UPSAMPLING_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    chiu_net_pp_arch.SKIP_DROPOUT_RATES_KEY: numpy.full(6, 0.),
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
    chiu_net_pp_arch.ENSEMBLE_SIZE_KEY: 25,
    chiu_net_pp_arch.USE_EVIDENTIAL_KEY: True,
    # chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION,
    chiu_net_pp_arch.USE_RESIDUAL_BLOCKS_KEY: True
}

NUM_GFS_PRESSURE_LEVELS = 2
NUM_GFS_2D_VARS = 5
NUM_GFS_3D_VARS = 5

# NN_LEAD_TIMES_DAYS = numpy.linspace(1, 10, num=10, dtype=int)
NN_LEAD_TIMES_DAYS = numpy.array([2], dtype=int)
LF_REGULARIZATION_WEIGHTS = numpy.logspace(-3, 1, num=17)

NN_LEAD_TIME_DAYS_TO_TARGET_LAG_TIMES_DAYS = {
    1: numpy.array([1, 2, 3], dtype=int),
    2: numpy.array([1, 2, 3], dtype=int),
    3: numpy.array([1, 2, 3], dtype=int),
    4: numpy.array([1, 2, 3, 4], dtype=int),
    5: numpy.array([1, 3, 5], dtype=int),
    6: numpy.array([2, 4, 6], dtype=int),
    7: numpy.array([1, 3, 5, 7], dtype=int),
    8: numpy.array([2, 4, 6, 8], dtype=int),
    9: numpy.array([1, 3, 5, 7, 9], dtype=int),
    10: numpy.array([2, 4, 6, 8, 10], dtype=int)
}

NN_LEAD_TIME_DAYS_TO_TARGET_LEAD_TIMES_DAYS = {
    1: numpy.array([1, 2, 3], dtype=int),
    2: numpy.array([1, 2, 3], dtype=int),
    3: numpy.array([1, 2, 3], dtype=int),
    4: numpy.array([1, 2, 3, 4], dtype=int),
    5: numpy.array([1, 3, 5], dtype=int),
    6: numpy.array([2, 4, 6], dtype=int),
    7: numpy.array([1, 3, 5, 7], dtype=int),
    8: numpy.array([2, 4, 6, 8], dtype=int),
    9: numpy.array([1, 3, 5, 7, 9], dtype=int),
    10: numpy.array([2, 4, 6, 8, 10], dtype=int)
}

NN_LEAD_TIME_DAYS_TO_GFS_LEAD_TIMES_HOURS = {
    1: numpy.array([0, 6, 12, 18, 24], dtype=int),
    2: numpy.array([0, 12, 24, 36, 48], dtype=int),
    3: numpy.array([0, 24, 48, 72], dtype=int),
    4: numpy.array([0, 24, 48, 72, 96], dtype=int),
    5: numpy.array([0, 36, 72, 108, 144], dtype=int),
    6: numpy.array([0, 36, 72, 108, 144], dtype=int),
    7: numpy.array([0, 48, 96, 144, 192], dtype=int),
    8: numpy.array([0, 48, 96, 144, 192], dtype=int),
    9: numpy.array([0, 60, 120, 168, 216], dtype=int),
    10: numpy.array([0, 60, 120, 168, 240], dtype=int)  # TODO(thunderhoser): weird but probably okay
}


def _run():
    """Tries making templates for recurrent evidential NN.

    This is effectively the main method.
    """

    for i in range(len(NN_LEAD_TIMES_DAYS)):
        for j in range(len(LF_REGULARIZATION_WEIGHTS)):
            option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

            this_num_lead_times = len(
                NN_LEAD_TIME_DAYS_TO_GFS_LEAD_TIMES_HOURS[NN_LEAD_TIMES_DAYS[i]]
            )
            these_dim_3d = numpy.array([
                265, 537,
                NUM_GFS_PRESSURE_LEVELS, this_num_lead_times, NUM_GFS_3D_VARS
            ], dtype=int)

            these_dim_2d = numpy.array([
                265, 537, this_num_lead_times, NUM_GFS_2D_VARS
            ], dtype=int)

            this_num_lag_times = len(
                NN_LEAD_TIME_DAYS_TO_TARGET_LAG_TIMES_DAYS[NN_LEAD_TIMES_DAYS[i]]
            )
            this_num_lead_times = len(
                NN_LEAD_TIME_DAYS_TO_TARGET_LAG_TIMES_DAYS[NN_LEAD_TIMES_DAYS[i]]
            )
            these_dim_laglead = numpy.array([
                265, 537, this_num_lag_times + this_num_lead_times, 6
            ], dtype=int)

            option_dict.update({
                chiu_net_pp_arch.GFS_3D_DIMENSIONS_KEY:
                    None if numpy.any(these_dim_3d == 0) else these_dim_3d,
                chiu_net_pp_arch.GFS_2D_DIMENSIONS_KEY: these_dim_2d,
                chiu_net_pp_arch.LAGTGT_DIMENSIONS_KEY: these_dim_laglead,
                chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION
            })

            this_loss_function = custom_losses.dual_weighted_evidential_loss(
                channel_weights=CHANNEL_WEIGHTS,
                max_dual_weight_by_channel=MAX_DUAL_WEIGHTS,
                regularization_weight=LF_REGULARIZATION_WEIGHTS[j],
                function_name='loss_evidential'
            )

            this_loss_function_string = (
                'custom_losses.dual_weighted_evidential_loss('
                'channel_weights=numpy.array([0.03502606, 0.00522109, 0.00012253, 0.83691845, 0.00334096, 0.11937091]),'
                'max_dual_weight_by_channel=numpy.array([96.4105, 303.9126, 1741.9033, 23.1660, 361.6984, 61.3669]),'
                'regularization_weight={0:.10f},'
                'function_name="loss_evidential"'
                ')'
            ).format(LF_REGULARIZATION_WEIGHTS[j])

            model_object = chiu_net_pp_arch.create_model(
                option_dict=option_dict,
                loss_function=this_loss_function,
                metric_list=METRIC_FUNCTIONS
            )

            output_file_name = (
                '{0:s}/nn-lead-time-days={1:02d}_lf-regularization-weight={2:013.10f}/'
                'model.keras'
            ).format(
                OUTPUT_DIR_NAME,
                NN_LEAD_TIMES_DAYS[i],
                LF_REGULARIZATION_WEIGHTS[j]
            )

            file_system_utils.mkdir_recursive_if_necessary(
                file_name=output_file_name
            )

            print('Writing model to: "{0:s}"...'.format(output_file_name))
            model_object.save(
                filepath=output_file_name,
                overwrite=True,
                include_optimizer=True
            )

            metafile_name = neural_net.find_metafile(
                model_file_name=output_file_name,
                raise_error_if_missing=False
            )
            option_dict[neural_net.LOSS_FUNCTION_KEY] = (
                this_loss_function_string
            )
            option_dict[neural_net.METRIC_FUNCTIONS_KEY] = (
                METRIC_FUNCTION_STRINGS
            )
            option_dict[neural_net.OPTIMIZER_FUNCTION_KEY] = (
                OPTIMIZER_FUNCTION_STRING
            )

            neural_net.write_metafile(
                pickle_file_name=metafile_name,
                num_epochs=100,
                num_training_batches_per_epoch=32,
                training_option_dict={},
                num_validation_batches_per_epoch=16,
                validation_option_dict={},
                loss_function_string=this_loss_function_string,
                optimizer_function_string=OPTIMIZER_FUNCTION_STRING,
                metric_function_strings=METRIC_FUNCTION_STRINGS,
                plateau_patience_epochs=10,
                plateau_learning_rate_multiplier=0.6,
                early_stopping_patience_epochs=50
            )


if __name__ == '__main__':
    _run()