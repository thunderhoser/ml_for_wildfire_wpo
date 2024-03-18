"""Makes templates for Experiment 9 (with multiple lead times)."""

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
    'experiment09_multi_lead_times/templates'
)

CHANNEL_WEIGHTS = numpy.array([0.02562263, 0.00373885, 0.00008940, 0.60291427, 0.00251213, 0.08761268, 0.27751004])
MAX_DUAL_WEIGHTS = numpy.array([96.4105, 303.9126, 1741.9033, 23.1660, 361.6984, 61.3669, 40.1856])

LOSS_FUNCTION = custom_losses.dual_weighted_mse_constrained_dsr(
    channel_weights=CHANNEL_WEIGHTS,
    max_dual_weight_by_channel=MAX_DUAL_WEIGHTS,
    fwi_index=5,
    expect_ensemble=False,
    function_name='loss_dwmse'
)

LOSS_FUNCTION_STRING = (
    'custom_losses.dual_weighted_mse_constrained_dsr('
    'channel_weights=numpy.array([0.02562263, 0.00373885, 0.00008940, 0.60291427, 0.00251213, 0.08761268, 0.27751004]), '
    'max_dual_weight_by_channel=numpy.array([96.4105, 303.9126, 1741.9033, 23.1660, 361.6984, 61.3669, 40.1856]), '
    'fwi_index=5, '
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
    # custom_metrics.max_prediction_unmasked(channel_index=6, expect_ensemble=False, function_name='dsr_max_prediction'),
    custom_metrics.max_prediction_anywhere(channel_index=0, expect_ensemble=False, function_name='ffmc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=1, expect_ensemble=False, function_name='dmc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=2, expect_ensemble=False, function_name='dc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=3, expect_ensemble=False, function_name='isi_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=4, expect_ensemble=False, function_name='bui_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=5, expect_ensemble=False, function_name='fwi_max_prediction_anywhere'),
    # custom_metrics.max_prediction_anywhere(channel_index=6, expect_ensemble=False, function_name='dsr_max_prediction_anywhere'),
    custom_metrics.mean_squared_error_unmasked(channel_index=0, expect_ensemble=False, function_name='ffmc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=1, expect_ensemble=False, function_name='dmc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=2, expect_ensemble=False, function_name='dc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=3, expect_ensemble=False, function_name='isi_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=4, expect_ensemble=False, function_name='bui_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=5, expect_ensemble=False, function_name='fwi_mse'),
    # custom_metrics.mean_squared_error_unmasked(channel_index=6, expect_ensemble=False, function_name='dsr_mse'),
    custom_metrics.mean_squared_error_anywhere(channel_index=0, expect_ensemble=False, function_name='ffmc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=1, expect_ensemble=False, function_name='dmc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=2, expect_ensemble=False, function_name='dc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=3, expect_ensemble=False, function_name='isi_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=4, expect_ensemble=False, function_name='bui_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=5, expect_ensemble=False, function_name='fwi_mse_anywhere'),
    # custom_metrics.mean_squared_error_anywhere(channel_index=6, expect_ensemble=False, function_name='dsr_mse_anywhere'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=0, expect_ensemble=False, function_name='ffmc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=1, expect_ensemble=False, function_name='dmc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=2, expect_ensemble=False, function_name='dc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=3, expect_ensemble=False, function_name='isi_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=4, expect_ensemble=False, function_name='bui_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=5, expect_ensemble=False, function_name='fwi_dwmse'),
    # custom_metrics.dual_weighted_mse_unmasked(channel_index=6, expect_ensemble=False, function_name='dsr_dwmse'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=0, expect_ensemble=False, function_name='ffmc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=1, expect_ensemble=False, function_name='dmc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=2, expect_ensemble=False, function_name='dc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=3, expect_ensemble=False, function_name='isi_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=4, expect_ensemble=False, function_name='bui_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=5, expect_ensemble=False, function_name='fwi_dwmse_anywhere'),
    # custom_metrics.dual_weighted_mse_anywhere(channel_index=6, expect_ensemble=False, function_name='dsr_dwmse_anywhere'),
    custom_losses.dual_weighted_mse_1channel(channel_index=0, channel_weight=0.02562263, max_dual_weight=96.4105, expect_ensemble=False, function_name='ffmc_dwmse_in_loss'),
    custom_losses.dual_weighted_mse_1channel(channel_index=1, channel_weight=0.00373885, max_dual_weight=303.9126, expect_ensemble=False, function_name='dmc_dwmse_in_loss'),
    custom_losses.dual_weighted_mse_1channel(channel_index=2, channel_weight=0.00008940, max_dual_weight=1741.9033, expect_ensemble=False, function_name='dc_dwmse_in_loss'),
    custom_losses.dual_weighted_mse_1channel(channel_index=3, channel_weight=0.60291427, max_dual_weight=23.1660, expect_ensemble=False, function_name='isi_dwmse_in_loss'),
    custom_losses.dual_weighted_mse_1channel(channel_index=4, channel_weight=0.00251213, max_dual_weight=361.6984, expect_ensemble=False, function_name='bui_dwmse_in_loss'),
    custom_losses.dual_weighted_mse_1channel(channel_index=5, channel_weight=0.08761268, max_dual_weight=61.3669, expect_ensemble=False, function_name='fwi_dwmse_in_loss'),
    # custom_losses.dual_weighted_mse_1channel(channel_index=6, channel_weight=0.27751004, max_dual_weight=40.1856, expect_ensemble=False, function_name='dsr_dwmse_in_loss')
]

METRIC_FUNCTION_STRINGS = [
    'custom_metrics.max_prediction_unmasked(channel_index=0, expect_ensemble=False, function_name="ffmc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=1, expect_ensemble=False, function_name="dmc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=2, expect_ensemble=False, function_name="dc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=3, expect_ensemble=False, function_name="isi_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=4, expect_ensemble=False, function_name="bui_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=5, expect_ensemble=False, function_name="fwi_max_prediction")',
    # 'custom_metrics.max_prediction_unmasked(channel_index=6, expect_ensemble=False, function_name="dsr_max_prediction")',
    'custom_metrics.max_prediction_anywhere(channel_index=0, expect_ensemble=False, function_name="ffmc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=1, expect_ensemble=False, function_name="dmc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=2, expect_ensemble=False, function_name="dc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=3, expect_ensemble=False, function_name="isi_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=4, expect_ensemble=False, function_name="bui_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=5, expect_ensemble=False, function_name="fwi_max_prediction_anywhere")',
    # 'custom_metrics.max_prediction_anywhere(channel_index=6, expect_ensemble=False, function_name="dsr_max_prediction_anywhere")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=0, expect_ensemble=False, function_name="ffmc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=1, expect_ensemble=False, function_name="dmc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=2, expect_ensemble=False, function_name="dc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=3, expect_ensemble=False, function_name="isi_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=4, expect_ensemble=False, function_name="bui_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=5, expect_ensemble=False, function_name="fwi_mse")',
    # 'custom_metrics.mean_squared_error_unmasked(channel_index=6, expect_ensemble=False, function_name="dsr_mse")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=0, expect_ensemble=False, function_name="ffmc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=1, expect_ensemble=False, function_name="dmc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=2, expect_ensemble=False, function_name="dc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=3, expect_ensemble=False, function_name="isi_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=4, expect_ensemble=False, function_name="bui_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=5, expect_ensemble=False, function_name="fwi_mse_anywhere")',
    # 'custom_metrics.mean_squared_error_anywhere(channel_index=6, expect_ensemble=False, function_name="dsr_mse_anywhere")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=0, expect_ensemble=False, function_name="ffmc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=1, expect_ensemble=False, function_name="dmc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=2, expect_ensemble=False, function_name="dc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=3, expect_ensemble=False, function_name="isi_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=4, expect_ensemble=False, function_name="bui_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=5, expect_ensemble=False, function_name="fwi_dwmse")',
    # 'custom_metrics.dual_weighted_mse_unmasked(channel_index=6, expect_ensemble=False, function_name="dsr_dwmse")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=0, expect_ensemble=False, function_name="ffmc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=1, expect_ensemble=False, function_name="dmc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=2, expect_ensemble=False, function_name="dc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=3, expect_ensemble=False, function_name="isi_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=4, expect_ensemble=False, function_name="bui_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=5, expect_ensemble=False, function_name="fwi_dwmse_anywhere")',
    # 'custom_metrics.dual_weighted_mse_anywhere(channel_index=6, expect_ensemble=False, function_name="dsr_dwmse_anywhere")',
    'custom_losses.dual_weighted_mse_1channel(channel_index=0, channel_weight=0.02562263, max_dual_weight=96.4105, expect_ensemble=False, function_name="ffmc_dwmse_in_loss")',
    'custom_losses.dual_weighted_mse_1channel(channel_index=1, channel_weight=0.00373885, max_dual_weight=303.9126, expect_ensemble=False, function_name="dmc_dwmse_in_loss")',
    'custom_losses.dual_weighted_mse_1channel(channel_index=2, channel_weight=0.00008940, max_dual_weight=1741.9033, expect_ensemble=False, function_name="dc_dwmse_in_loss")',
    'custom_losses.dual_weighted_mse_1channel(channel_index=3, channel_weight=0.60291427, max_dual_weight=23.1660, expect_ensemble=False, function_name="isi_dwmse_in_loss")',
    'custom_losses.dual_weighted_mse_1channel(channel_index=4, channel_weight=0.00251213, max_dual_weight=361.6984, expect_ensemble=False, function_name="bui_dwmse_in_loss")',
    'custom_losses.dual_weighted_mse_1channel(channel_index=5, channel_weight=0.08761268, max_dual_weight=61.3669, expect_ensemble=False, function_name="fwi_dwmse_in_loss")',
    # 'custom_losses.dual_weighted_mse_1channel(channel_index=6, channel_weight=0.27751004, max_dual_weight=40.1856, expect_ensemble=False, function_name="dsr_dwmse_in_loss")'
]

NUM_CONV_LAYERS_PER_BLOCK = 1

# TODO(thunderhoser): Make sure to adjust Slurm script for this!
OPTIMIZER_FUNCTION = keras.optimizers.Nadam(gradient_accumulation_steps=6)
OPTIMIZER_FUNCTION_STRING = (
    'keras.optimizers.Nadam(gradient_accumulation_steps=6)'
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
        [50, 75, 100, 125, 150, 175, 200], dtype=int
    ),
    chiu_net_pp_arch.GFS_ENCODER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(
        7, NUM_CONV_LAYERS_PER_BLOCK, dtype=int
    ),
    chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY: numpy.array(
        [30, 40, 50, 60, 70, 80, 90], dtype=int
    ),
    chiu_net_pp_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    chiu_net_pp_arch.DECODER_NUM_CONV_LAYERS_KEY: numpy.full(6, 2, dtype=int),
    chiu_net_pp_arch.DECODER_NUM_CHANNELS_KEY: numpy.array(
        [40, 58, 75, 93, 110, 128], dtype=int
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
    chiu_net_pp_arch.ENSEMBLE_SIZE_KEY: 1,
    # chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION
}

NUM_GFS_PRESSURE_LEVELS = 2
NUM_GFS_2D_VARS = 5
NUM_GFS_3D_VARS = 5

NN_LEAD_TIMES_DAYS = numpy.linspace(1, 10, num=10, dtype=int)
RANDOM_SEED_INDICES = numpy.linspace(1, 5, num=5, dtype=int)

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
    """Makes templates for Experiment 9 (with multiple lead times).

    This is effectively the main method.
    """

    for i in range(len(NN_LEAD_TIMES_DAYS)):
        for j in range(len(RANDOM_SEED_INDICES)):
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

            model_object = chiu_net_pp_arch.create_model(
                option_dict=option_dict,
                loss_function=LOSS_FUNCTION,
                metric_list=METRIC_FUNCTIONS
            )

            output_file_name = (
                '{0:s}/nn-lead-time-days={1:02d}_random-seed-index={2:d}/'
                'model.keras'
            ).format(
                OUTPUT_DIR_NAME,
                NN_LEAD_TIMES_DAYS[i],
                RANDOM_SEED_INDICES[j]
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
                LOSS_FUNCTION_STRING
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
                loss_function_string=LOSS_FUNCTION_STRING,
                optimizer_function_string=OPTIMIZER_FUNCTION_STRING,
                metric_function_strings=METRIC_FUNCTION_STRINGS,
                plateau_patience_epochs=10,
                plateau_learning_rate_multiplier=0.6,
                early_stopping_patience_epochs=50
            )


if __name__ == '__main__':
    _run()
