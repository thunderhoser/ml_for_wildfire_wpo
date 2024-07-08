"""Makes test template with flexible lead time."""

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
import chiu_net_pp_flexi_architecture as chiu_net_pp_flexi_arch
import architecture_utils
import file_system_utils

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_models/'
    'flexible_lead_time_test/template'
)

CHANNEL_WEIGHTS = numpy.array([0.02562263, 0.00373885, 0.00008940, 0.60291427, 0.00251213, 0.08761268, 0.27751004])
MAX_DUAL_WEIGHTS = numpy.array([96.4105, 303.9126, 1741.9033, 23.1660, 361.6984, 61.3669, 40.1856])

LOSS_FUNCTION = custom_losses.dual_weighted_crps_constrained_dsr(
    channel_weights=CHANNEL_WEIGHTS,
    max_dual_weight_by_channel=MAX_DUAL_WEIGHTS,
    fwi_index=5,
    function_name='loss_dwcrps'
)

LOSS_FUNCTION_STRING = (
    'custom_losses.dual_weighted_crps_constrained_dsr('
    'channel_weights=numpy.array([0.02562263, 0.00373885, 0.00008940, 0.60291427, 0.00251213, 0.08761268, 0.27751004]), '
    'max_dual_weight_by_channel=numpy.array([96.4105, 303.9126, 1741.9033, 23.1660, 361.6984, 61.3669, 40.1856]), '
    'fwi_index=5, '
    'function_name="loss_dwcrps")'
)

METRIC_FUNCTIONS = [
    custom_metrics.max_prediction_unmasked(channel_index=0, expect_ensemble=True, function_name='ffmc_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=1, expect_ensemble=True, function_name='dmc_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=2, expect_ensemble=True, function_name='dc_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=3, expect_ensemble=True, function_name='isi_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=4, expect_ensemble=True, function_name='bui_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=5, expect_ensemble=True, function_name='fwi_max_prediction'),
    # custom_metrics.max_prediction_unmasked(channel_index=6, expect_ensemble=True, function_name='dsr_max_prediction'),
    custom_metrics.max_prediction_anywhere(channel_index=0, expect_ensemble=True, function_name='ffmc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=1, expect_ensemble=True, function_name='dmc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=2, expect_ensemble=True, function_name='dc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=3, expect_ensemble=True, function_name='isi_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=4, expect_ensemble=True, function_name='bui_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=5, expect_ensemble=True, function_name='fwi_max_prediction_anywhere'),
    # custom_metrics.max_prediction_anywhere(channel_index=6, expect_ensemble=True, function_name='dsr_max_prediction_anywhere'),
    custom_metrics.mean_squared_error_unmasked(channel_index=0, expect_ensemble=True, function_name='ffmc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=1, expect_ensemble=True, function_name='dmc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=2, expect_ensemble=True, function_name='dc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=3, expect_ensemble=True, function_name='isi_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=4, expect_ensemble=True, function_name='bui_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=5, expect_ensemble=True, function_name='fwi_mse'),
    # custom_metrics.mean_squared_error_unmasked(channel_index=6, expect_ensemble=True, function_name='dsr_mse'),
    custom_metrics.mean_squared_error_anywhere(channel_index=0, expect_ensemble=True, function_name='ffmc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=1, expect_ensemble=True, function_name='dmc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=2, expect_ensemble=True, function_name='dc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=3, expect_ensemble=True, function_name='isi_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=4, expect_ensemble=True, function_name='bui_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=5, expect_ensemble=True, function_name='fwi_mse_anywhere'),
    # custom_metrics.mean_squared_error_anywhere(channel_index=6, expect_ensemble=True, function_name='dsr_mse_anywhere'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=0, expect_ensemble=True, function_name='ffmc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=1, expect_ensemble=True, function_name='dmc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=2, expect_ensemble=True, function_name='dc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=3, expect_ensemble=True, function_name='isi_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=4, expect_ensemble=True, function_name='bui_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=5, expect_ensemble=True, function_name='fwi_dwmse'),
    # custom_metrics.dual_weighted_mse_unmasked(channel_index=6, expect_ensemble=True, function_name='dsr_dwmse'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=0, expect_ensemble=True, function_name='ffmc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=1, expect_ensemble=True, function_name='dmc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=2, expect_ensemble=True, function_name='dc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=3, expect_ensemble=True, function_name='isi_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=4, expect_ensemble=True, function_name='bui_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=5, expect_ensemble=True, function_name='fwi_dwmse_anywhere'),
    # custom_metrics.dual_weighted_mse_anywhere(channel_index=6, expect_ensemble=True, function_name='dsr_dwmse_anywhere')
]

METRIC_FUNCTION_STRINGS = [
    'custom_metrics.max_prediction_unmasked(channel_index=0, expect_ensemble=True, function_name="ffmc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=1, expect_ensemble=True, function_name="dmc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=2, expect_ensemble=True, function_name="dc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=3, expect_ensemble=True, function_name="isi_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=4, expect_ensemble=True, function_name="bui_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=5, expect_ensemble=True, function_name="fwi_max_prediction")',
    # 'custom_metrics.max_prediction_unmasked(channel_index=6, expect_ensemble=True, function_name="dsr_max_prediction")',
    'custom_metrics.max_prediction_anywhere(channel_index=0, expect_ensemble=True, function_name="ffmc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=1, expect_ensemble=True, function_name="dmc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=2, expect_ensemble=True, function_name="dc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=3, expect_ensemble=True, function_name="isi_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=4, expect_ensemble=True, function_name="bui_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=5, expect_ensemble=True, function_name="fwi_max_prediction_anywhere")',
    # 'custom_metrics.max_prediction_anywhere(channel_index=6, expect_ensemble=True, function_name="dsr_max_prediction_anywhere")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=0, expect_ensemble=True, function_name="ffmc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=1, expect_ensemble=True, function_name="dmc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=2, expect_ensemble=True, function_name="dc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=3, expect_ensemble=True, function_name="isi_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=4, expect_ensemble=True, function_name="bui_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=5, expect_ensemble=True, function_name="fwi_mse")',
    # 'custom_metrics.mean_squared_error_unmasked(channel_index=6, expect_ensemble=True, function_name="dsr_mse")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=0, expect_ensemble=True, function_name="ffmc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=1, expect_ensemble=True, function_name="dmc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=2, expect_ensemble=True, function_name="dc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=3, expect_ensemble=True, function_name="isi_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=4, expect_ensemble=True, function_name="bui_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=5, expect_ensemble=True, function_name="fwi_mse_anywhere")',
    # 'custom_metrics.mean_squared_error_anywhere(channel_index=6, expect_ensemble=True, function_name="dsr_mse_anywhere")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=0, expect_ensemble=True, function_name="ffmc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=1, expect_ensemble=True, function_name="dmc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=2, expect_ensemble=True, function_name="dc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=3, expect_ensemble=True, function_name="isi_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=4, expect_ensemble=True, function_name="bui_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=5, expect_ensemble=True, function_name="fwi_dwmse")',
    # 'custom_metrics.dual_weighted_mse_unmasked(channel_index=6, expect_ensemble=True, function_name="dsr_dwmse")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=0, expect_ensemble=True, function_name="ffmc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=1, expect_ensemble=True, function_name="dmc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=2, expect_ensemble=True, function_name="dc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=3, expect_ensemble=True, function_name="isi_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=4, expect_ensemble=True, function_name="bui_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=5, expect_ensemble=True, function_name="fwi_dwmse_anywhere")',
    # 'custom_metrics.dual_weighted_mse_anywhere(channel_index=6, expect_ensemble=True, function_name="dsr_dwmse_anywhere")'
]

NUM_CONV_LAYERS_PER_BLOCK = 1

OPTIMIZER_FUNCTION = keras.optimizers.Nadam(gradient_accumulation_steps=24)
OPTIMIZER_FUNCTION_STRING = (
    'keras.optimizers.Nadam(gradient_accumulation_steps=24)'
)

DEFAULT_OPTION_DICT = {
    # chiu_net_pp_flexi_arch.GFS_3D_DIMENSIONS_KEY: numpy.array(
    #     [265, 537, 2, NUM_GFS_LEAD_TIMES, 5], dtype=int
    # ),
    # chiu_net_pp_flexi_arch.GFS_2D_DIMENSIONS_KEY: numpy.array(
    #     [265, 537, NUM_GFS_LEAD_TIMES, 7], dtype=int
    # ),
    chiu_net_pp_flexi_arch.ERA5_CONST_DIMENSIONS_KEY: numpy.array(
        [265, 537, 7], dtype=int
    ),
    # chiu_net_pp_flexi_arch.LAGTGT_DIMENSIONS_KEY: numpy.array(
    #     [265, 537, 6, 7], dtype=int
    # ),
    chiu_net_pp_flexi_arch.PREDN_BASELINE_DIMENSIONS_KEY: numpy.array(
        [265, 537, 6], dtype=int
    ),
    chiu_net_pp_flexi_arch.GFS_FC_MODULE_NUM_LSTM_LAYERS_KEY: 2,
    chiu_net_pp_flexi_arch.GFS_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0., 0.]),
    chiu_net_pp_flexi_arch.LAGTGT_FC_MODULE_NUM_LSTM_LAYERS_KEY: 2,
    chiu_net_pp_flexi_arch.LAGTGT_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0., 0.]),
    chiu_net_pp_flexi_arch.NUM_LEVELS_KEY: 6,
    chiu_net_pp_flexi_arch.GFS_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(
        7, NUM_CONV_LAYERS_PER_BLOCK, dtype=int
    ),
    chiu_net_pp_flexi_arch.GFS_ENCODER_NUM_CHANNELS_KEY: numpy.array(
        [25, 37, 50, 62, 75, 87, 100], dtype=int
    ),
    chiu_net_pp_flexi_arch.GFS_ENCODER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    chiu_net_pp_flexi_arch.LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(
        7, NUM_CONV_LAYERS_PER_BLOCK, dtype=int
    ),
    chiu_net_pp_flexi_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY: numpy.array(
        [15, 20, 25, 30, 35, 40, 45], dtype=int
    ),
    chiu_net_pp_flexi_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    chiu_net_pp_flexi_arch.DECODER_NUM_CONV_LAYERS_KEY: numpy.full(6, 2, dtype=int),
    chiu_net_pp_flexi_arch.DECODER_NUM_CHANNELS_KEY: numpy.array(
        [20, 28, 37, 46, 55, 63], dtype=int
    ),
    chiu_net_pp_flexi_arch.UPSAMPLING_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    chiu_net_pp_flexi_arch.SKIP_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    chiu_net_pp_flexi_arch.INCLUDE_PENULTIMATE_KEY: False,
    chiu_net_pp_flexi_arch.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    chiu_net_pp_flexi_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    chiu_net_pp_flexi_arch.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    chiu_net_pp_flexi_arch.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    chiu_net_pp_flexi_arch.L1_WEIGHT_KEY: 0.,
    chiu_net_pp_flexi_arch.L2_WEIGHT_KEY: 1e-6,
    chiu_net_pp_flexi_arch.USE_BATCH_NORM_KEY: True,
    chiu_net_pp_flexi_arch.ENSEMBLE_SIZE_KEY: 25,
    # chiu_net_pp_flexi_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION,
    chiu_net_pp_flexi_arch.USE_RESIDUAL_BLOCKS_KEY: True
}

NUM_GFS_PRESSURE_LEVELS = 2
NUM_GFS_2D_VARS = 5
NUM_GFS_3D_VARS = 5


def _run():
    """Makes test template with flexible lead time.

    This is effectively the main method.
    """

    option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

    these_dim_3d = numpy.array(
        [265, 537, NUM_GFS_PRESSURE_LEVELS, -1, NUM_GFS_3D_VARS], dtype=int
    )
    these_dim_2d = numpy.array([265, 537, -1, NUM_GFS_2D_VARS], dtype=int)
    these_dim_laglead = numpy.array([265, 537, -1, 6], dtype=int)

    option_dict.update({
        chiu_net_pp_flexi_arch.GFS_3D_DIMENSIONS_KEY:
            None if numpy.any(these_dim_3d == 0) else these_dim_3d,
        chiu_net_pp_flexi_arch.GFS_2D_DIMENSIONS_KEY: these_dim_2d,
        chiu_net_pp_flexi_arch.LAGTGT_DIMENSIONS_KEY: these_dim_laglead,
        chiu_net_pp_flexi_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION
    })

    model_object = chiu_net_pp_flexi_arch.create_model(
        option_dict=option_dict,
        loss_function=LOSS_FUNCTION,
        metric_list=METRIC_FUNCTIONS
    )

    output_file_name = '{0:s}/model.keras'.format(OUTPUT_DIR_NAME)
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
