"""Makes templates for Experiment 23c (third step of ablation)."""

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
    'experiment23c_ablation/templates'
)

# FFMC, DMC, DC, ISI, BUI, FWI, DSR
CHANNEL_WEIGHTS = numpy.array([0.02562263, 0.00373885, 0.00008940, 0.60291427, 0.00251213, 0.08761268, 0.27751004])
MAX_DUAL_WEIGHTS = numpy.array([96.4105, 303.9126, 1741.9033, 23.1660, 361.6984, 61.3669, 40.1856])

LOSS_FUNCTION = custom_losses.dual_weighted_crpss(
    channel_weights=CHANNEL_WEIGHTS,
    max_dual_weight_by_channel=MAX_DUAL_WEIGHTS,
    dmc_dc_isi_indices_for_constraints=None,
    function_name='loss_dwcrpss'
)

LOSS_FUNCTION_STRING = (
    'custom_losses.dual_weighted_crpss('
    'channel_weights=numpy.array([0.02562263, 0.00373885, 0.00008940, 0.60291427, 0.00251213, 0.08761268, 0.27751004]), '
    'max_dual_weight_by_channel=numpy.array([96.4105, 303.9126, 1741.9033, 23.1660, 361.6984, 61.3669, 40.1856]), '
    'dmc_dc_isi_indices_for_constraints=None, '
    'function_name="loss_dwcrpss")'
)

METRIC_FUNCTIONS = [
    custom_metrics.max_prediction_unmasked(channel_index=0, expect_ensemble=True, function_name='ffmc_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=1, expect_ensemble=True, function_name='dmc_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=2, expect_ensemble=True, function_name='dc_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=3, expect_ensemble=True, function_name='isi_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=4, expect_ensemble=True, function_name='bui_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=5, expect_ensemble=True, function_name='fwi_max_prediction'),
    custom_metrics.max_prediction_unmasked(channel_index=6, expect_ensemble=True, function_name='dsr_max_prediction'),

    custom_metrics.max_prediction_anywhere(channel_index=0, expect_ensemble=True, function_name='ffmc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=1, expect_ensemble=True, function_name='dmc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=2, expect_ensemble=True, function_name='dc_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=3, expect_ensemble=True, function_name='isi_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=4, expect_ensemble=True, function_name='bui_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=5, expect_ensemble=True, function_name='fwi_max_prediction_anywhere'),
    custom_metrics.max_prediction_anywhere(channel_index=6, expect_ensemble=True, function_name='dsr_max_prediction_anywhere'),

    custom_metrics.min_prediction_unmasked(channel_index=0, expect_ensemble=True, function_name='ffmc_min_prediction'),
    custom_metrics.min_prediction_unmasked(channel_index=1, expect_ensemble=True, function_name='dmc_min_prediction'),
    custom_metrics.min_prediction_unmasked(channel_index=2, expect_ensemble=True, function_name='dc_min_prediction'),
    custom_metrics.min_prediction_unmasked(channel_index=3, expect_ensemble=True, function_name='isi_min_prediction'),
    custom_metrics.min_prediction_unmasked(channel_index=4, expect_ensemble=True, function_name='bui_min_prediction'),
    custom_metrics.min_prediction_unmasked(channel_index=5, expect_ensemble=True, function_name='fwi_min_prediction'),
    custom_metrics.min_prediction_unmasked(channel_index=6, expect_ensemble=True, function_name='dsr_min_prediction'),

    custom_metrics.min_prediction_anywhere(channel_index=0, expect_ensemble=True, function_name='ffmc_min_prediction_anywhere'),
    custom_metrics.min_prediction_anywhere(channel_index=1, expect_ensemble=True, function_name='dmc_min_prediction_anywhere'),
    custom_metrics.min_prediction_anywhere(channel_index=2, expect_ensemble=True, function_name='dc_min_prediction_anywhere'),
    custom_metrics.min_prediction_anywhere(channel_index=3, expect_ensemble=True, function_name='isi_min_prediction_anywhere'),
    custom_metrics.min_prediction_anywhere(channel_index=4, expect_ensemble=True, function_name='bui_min_prediction_anywhere'),
    custom_metrics.min_prediction_anywhere(channel_index=5, expect_ensemble=True, function_name='fwi_min_prediction_anywhere'),
    custom_metrics.min_prediction_anywhere(channel_index=6, expect_ensemble=True, function_name='dsr_min_prediction_anywhere'),

    custom_metrics.max_target_unmasked(channel_index=0, expect_ensemble=True, function_name='ffmc_max_target'),
    custom_metrics.max_target_unmasked(channel_index=1, expect_ensemble=True, function_name='dmc_max_target'),
    custom_metrics.max_target_unmasked(channel_index=2, expect_ensemble=True, function_name='dc_max_target'),
    custom_metrics.max_target_unmasked(channel_index=3, expect_ensemble=True, function_name='isi_max_target'),
    custom_metrics.max_target_unmasked(channel_index=4, expect_ensemble=True, function_name='bui_max_target'),
    custom_metrics.max_target_unmasked(channel_index=5, expect_ensemble=True, function_name='fwi_max_target'),
    custom_metrics.max_target_unmasked(channel_index=6, expect_ensemble=True, function_name='dsr_max_target'),

    custom_metrics.max_target_anywhere(channel_index=0, expect_ensemble=True, function_name='ffmc_max_target_anywhere'),
    custom_metrics.max_target_anywhere(channel_index=1, expect_ensemble=True, function_name='dmc_max_target_anywhere'),
    custom_metrics.max_target_anywhere(channel_index=2, expect_ensemble=True, function_name='dc_max_target_anywhere'),
    custom_metrics.max_target_anywhere(channel_index=3, expect_ensemble=True, function_name='isi_max_target_anywhere'),
    custom_metrics.max_target_anywhere(channel_index=4, expect_ensemble=True, function_name='bui_max_target_anywhere'),
    custom_metrics.max_target_anywhere(channel_index=5, expect_ensemble=True, function_name='fwi_max_target_anywhere'),
    custom_metrics.max_target_anywhere(channel_index=6, expect_ensemble=True, function_name='dsr_max_target_anywhere'),

    custom_metrics.min_target_unmasked(channel_index=0, expect_ensemble=True, function_name='ffmc_min_target'),
    custom_metrics.min_target_unmasked(channel_index=1, expect_ensemble=True, function_name='dmc_min_target'),
    custom_metrics.min_target_unmasked(channel_index=2, expect_ensemble=True, function_name='dc_min_target'),
    custom_metrics.min_target_unmasked(channel_index=3, expect_ensemble=True, function_name='isi_min_target'),
    custom_metrics.min_target_unmasked(channel_index=4, expect_ensemble=True, function_name='bui_min_target'),
    custom_metrics.min_target_unmasked(channel_index=5, expect_ensemble=True, function_name='fwi_min_target'),
    custom_metrics.min_target_unmasked(channel_index=6, expect_ensemble=True, function_name='dsr_min_target'),

    custom_metrics.min_target_anywhere(channel_index=0, expect_ensemble=True, function_name='ffmc_min_target_anywhere'),
    custom_metrics.min_target_anywhere(channel_index=1, expect_ensemble=True, function_name='dmc_min_target_anywhere'),
    custom_metrics.min_target_anywhere(channel_index=2, expect_ensemble=True, function_name='dc_min_target_anywhere'),
    custom_metrics.min_target_anywhere(channel_index=3, expect_ensemble=True, function_name='isi_min_target_anywhere'),
    custom_metrics.min_target_anywhere(channel_index=4, expect_ensemble=True, function_name='bui_min_target_anywhere'),
    custom_metrics.min_target_anywhere(channel_index=5, expect_ensemble=True, function_name='fwi_min_target_anywhere'),
    custom_metrics.min_target_anywhere(channel_index=6, expect_ensemble=True, function_name='dsr_min_target_anywhere'),

    custom_metrics.mean_squared_error_unmasked(channel_index=0, expect_ensemble=True, function_name='ffmc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=1, expect_ensemble=True, function_name='dmc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=2, expect_ensemble=True, function_name='dc_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=3, expect_ensemble=True, function_name='isi_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=4, expect_ensemble=True, function_name='bui_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=5, expect_ensemble=True, function_name='fwi_mse'),
    custom_metrics.mean_squared_error_unmasked(channel_index=6, expect_ensemble=True, function_name='dsr_mse'),

    custom_metrics.mean_squared_error_anywhere(channel_index=0, expect_ensemble=True, function_name='ffmc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=1, expect_ensemble=True, function_name='dmc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=2, expect_ensemble=True, function_name='dc_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=3, expect_ensemble=True, function_name='isi_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=4, expect_ensemble=True, function_name='bui_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=5, expect_ensemble=True, function_name='fwi_mse_anywhere'),
    custom_metrics.mean_squared_error_anywhere(channel_index=6, expect_ensemble=True, function_name='dsr_mse_anywhere'),

    custom_metrics.dual_weighted_mse_unmasked(channel_index=0, expect_ensemble=True, function_name='ffmc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=1, expect_ensemble=True, function_name='dmc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=2, expect_ensemble=True, function_name='dc_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=3, expect_ensemble=True, function_name='isi_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=4, expect_ensemble=True, function_name='bui_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=5, expect_ensemble=True, function_name='fwi_dwmse'),
    custom_metrics.dual_weighted_mse_unmasked(channel_index=6, expect_ensemble=True, function_name='dsr_dwmse'),

    custom_metrics.dual_weighted_mse_anywhere(channel_index=0, expect_ensemble=True, function_name='ffmc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=1, expect_ensemble=True, function_name='dmc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=2, expect_ensemble=True, function_name='dc_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=3, expect_ensemble=True, function_name='isi_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=4, expect_ensemble=True, function_name='bui_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=5, expect_ensemble=True, function_name='fwi_dwmse_anywhere'),
    custom_metrics.dual_weighted_mse_anywhere(channel_index=6, expect_ensemble=True, function_name='dsr_dwmse_anywhere')
]

METRIC_FUNCTION_STRINGS = [
    'custom_metrics.max_prediction_unmasked(channel_index=0, expect_ensemble=True, function_name="ffmc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=1, expect_ensemble=True, function_name="dmc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=2, expect_ensemble=True, function_name="dc_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=3, expect_ensemble=True, function_name="isi_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=4, expect_ensemble=True, function_name="bui_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=5, expect_ensemble=True, function_name="fwi_max_prediction")',
    'custom_metrics.max_prediction_unmasked(channel_index=6, expect_ensemble=True, function_name="dsr_max_prediction")',

    'custom_metrics.max_prediction_anywhere(channel_index=0, expect_ensemble=True, function_name="ffmc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=1, expect_ensemble=True, function_name="dmc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=2, expect_ensemble=True, function_name="dc_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=3, expect_ensemble=True, function_name="isi_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=4, expect_ensemble=True, function_name="bui_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=5, expect_ensemble=True, function_name="fwi_max_prediction_anywhere")',
    'custom_metrics.max_prediction_anywhere(channel_index=6, expect_ensemble=True, function_name="dsr_max_prediction_anywhere")',

    'custom_metrics.min_prediction_unmasked(channel_index=0, expect_ensemble=True, function_name="ffmc_min_prediction")',
    'custom_metrics.min_prediction_unmasked(channel_index=1, expect_ensemble=True, function_name="dmc_min_prediction")',
    'custom_metrics.min_prediction_unmasked(channel_index=2, expect_ensemble=True, function_name="dc_min_prediction")',
    'custom_metrics.min_prediction_unmasked(channel_index=3, expect_ensemble=True, function_name="isi_min_prediction")',
    'custom_metrics.min_prediction_unmasked(channel_index=4, expect_ensemble=True, function_name="bui_min_prediction")',
    'custom_metrics.min_prediction_unmasked(channel_index=5, expect_ensemble=True, function_name="fwi_min_prediction")',
    'custom_metrics.min_prediction_unmasked(channel_index=6, expect_ensemble=True, function_name="dsr_min_prediction")',

    'custom_metrics.min_prediction_anywhere(channel_index=0, expect_ensemble=True, function_name="ffmc_min_prediction_anywhere")',
    'custom_metrics.min_prediction_anywhere(channel_index=1, expect_ensemble=True, function_name="dmc_min_prediction_anywhere")',
    'custom_metrics.min_prediction_anywhere(channel_index=2, expect_ensemble=True, function_name="dc_min_prediction_anywhere")',
    'custom_metrics.min_prediction_anywhere(channel_index=3, expect_ensemble=True, function_name="isi_min_prediction_anywhere")',
    'custom_metrics.min_prediction_anywhere(channel_index=4, expect_ensemble=True, function_name="bui_min_prediction_anywhere")',
    'custom_metrics.min_prediction_anywhere(channel_index=5, expect_ensemble=True, function_name="fwi_min_prediction_anywhere")',
    'custom_metrics.min_prediction_anywhere(channel_index=6, expect_ensemble=True, function_name="dsr_min_prediction_anywhere")',

    'custom_metrics.max_target_unmasked(channel_index=0, expect_ensemble=True, function_name="ffmc_max_target")',
    'custom_metrics.max_target_unmasked(channel_index=1, expect_ensemble=True, function_name="dmc_max_target")',
    'custom_metrics.max_target_unmasked(channel_index=2, expect_ensemble=True, function_name="dc_max_target")',
    'custom_metrics.max_target_unmasked(channel_index=3, expect_ensemble=True, function_name="isi_max_target")',
    'custom_metrics.max_target_unmasked(channel_index=4, expect_ensemble=True, function_name="bui_max_target")',
    'custom_metrics.max_target_unmasked(channel_index=5, expect_ensemble=True, function_name="fwi_max_target")',
    'custom_metrics.max_target_unmasked(channel_index=6, expect_ensemble=True, function_name="dsr_max_target")',

    'custom_metrics.max_target_anywhere(channel_index=0, expect_ensemble=True, function_name="ffmc_max_target_anywhere")',
    'custom_metrics.max_target_anywhere(channel_index=1, expect_ensemble=True, function_name="dmc_max_target_anywhere")',
    'custom_metrics.max_target_anywhere(channel_index=2, expect_ensemble=True, function_name="dc_max_target_anywhere")',
    'custom_metrics.max_target_anywhere(channel_index=3, expect_ensemble=True, function_name="isi_max_target_anywhere")',
    'custom_metrics.max_target_anywhere(channel_index=4, expect_ensemble=True, function_name="bui_max_target_anywhere")',
    'custom_metrics.max_target_anywhere(channel_index=5, expect_ensemble=True, function_name="fwi_max_target_anywhere")',
    'custom_metrics.max_target_anywhere(channel_index=6, expect_ensemble=True, function_name="dsr_max_target_anywhere")',

    'custom_metrics.min_target_unmasked(channel_index=0, expect_ensemble=True, function_name="ffmc_min_target")',
    'custom_metrics.min_target_unmasked(channel_index=1, expect_ensemble=True, function_name="dmc_min_target")',
    'custom_metrics.min_target_unmasked(channel_index=2, expect_ensemble=True, function_name="dc_min_target")',
    'custom_metrics.min_target_unmasked(channel_index=3, expect_ensemble=True, function_name="isi_min_target")',
    'custom_metrics.min_target_unmasked(channel_index=4, expect_ensemble=True, function_name="bui_min_target")',
    'custom_metrics.min_target_unmasked(channel_index=5, expect_ensemble=True, function_name="fwi_min_target")',
    'custom_metrics.min_target_unmasked(channel_index=6, expect_ensemble=True, function_name="dsr_min_target")',

    'custom_metrics.min_target_anywhere(channel_index=0, expect_ensemble=True, function_name="ffmc_min_target_anywhere")',
    'custom_metrics.min_target_anywhere(channel_index=1, expect_ensemble=True, function_name="dmc_min_target_anywhere")',
    'custom_metrics.min_target_anywhere(channel_index=2, expect_ensemble=True, function_name="dc_min_target_anywhere")',
    'custom_metrics.min_target_anywhere(channel_index=3, expect_ensemble=True, function_name="isi_min_target_anywhere")',
    'custom_metrics.min_target_anywhere(channel_index=4, expect_ensemble=True, function_name="bui_min_target_anywhere")',
    'custom_metrics.min_target_anywhere(channel_index=5, expect_ensemble=True, function_name="fwi_min_target_anywhere")',
    'custom_metrics.min_target_anywhere(channel_index=6, expect_ensemble=True, function_name="dsr_min_target_anywhere")',

    'custom_metrics.mean_squared_error_unmasked(channel_index=0, expect_ensemble=True, function_name="ffmc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=1, expect_ensemble=True, function_name="dmc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=2, expect_ensemble=True, function_name="dc_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=3, expect_ensemble=True, function_name="isi_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=4, expect_ensemble=True, function_name="bui_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=5, expect_ensemble=True, function_name="fwi_mse")',
    'custom_metrics.mean_squared_error_unmasked(channel_index=6, expect_ensemble=True, function_name="dsr_mse")',

    'custom_metrics.mean_squared_error_anywhere(channel_index=0, expect_ensemble=True, function_name="ffmc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=1, expect_ensemble=True, function_name="dmc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=2, expect_ensemble=True, function_name="dc_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=3, expect_ensemble=True, function_name="isi_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=4, expect_ensemble=True, function_name="bui_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=5, expect_ensemble=True, function_name="fwi_mse_anywhere")',
    'custom_metrics.mean_squared_error_anywhere(channel_index=6, expect_ensemble=True, function_name="dsr_mse_anywhere")',

    'custom_metrics.dual_weighted_mse_unmasked(channel_index=0, expect_ensemble=True, function_name="ffmc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=1, expect_ensemble=True, function_name="dmc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=2, expect_ensemble=True, function_name="dc_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=3, expect_ensemble=True, function_name="isi_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=4, expect_ensemble=True, function_name="bui_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=5, expect_ensemble=True, function_name="fwi_dwmse")',
    'custom_metrics.dual_weighted_mse_unmasked(channel_index=6, expect_ensemble=True, function_name="dsr_dwmse")',

    'custom_metrics.dual_weighted_mse_anywhere(channel_index=0, expect_ensemble=True, function_name="ffmc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=1, expect_ensemble=True, function_name="dmc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=2, expect_ensemble=True, function_name="dc_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=3, expect_ensemble=True, function_name="isi_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=4, expect_ensemble=True, function_name="bui_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=5, expect_ensemble=True, function_name="fwi_dwmse_anywhere")',
    'custom_metrics.dual_weighted_mse_anywhere(channel_index=6, expect_ensemble=True, function_name="dsr_dwmse_anywhere")'
]

NUM_CONV_LAYERS_PER_BLOCK = 1
MODEL_DEPTH = 5

OPTIMIZER_FUNCTION = keras.optimizers.AdamW(gradient_accumulation_steps=int(10 * 44 * 0.2))
OPTIMIZER_FUNCTION_STRING = (
    'keras.optimizers.AdamW(gradient_accumulation_steps=int(10 * 44 * 0.2))'
)

DEFAULT_OPTION_DICT = {
    # chiu_net_pp_arch.GFS_3D_DIMENSIONS_KEY: numpy.array(
    #     [160, 160, 2, -1, 5], dtype=int
    # ),
    chiu_net_pp_arch.GFS_2D_DIMENSIONS_KEY: numpy.array(
        [160, 160, 1, 5], dtype=int
    ),
    # chiu_net_pp_arch.ERA5_CONST_DIMENSIONS_KEY: numpy.array(
    #     [160, 160, 7], dtype=int
    # ),
    # chiu_net_pp_arch.LAGTGT_DIMENSIONS_KEY: None,
    # chiu_net_pp_arch.PREDN_BASELINE_DIMENSIONS_KEY: numpy.array(
    #     [160, 160, 4], dtype=int
    # ),
    chiu_net_pp_arch.USE_RESIDUAL_BLOCKS_KEY: False,
    chiu_net_pp_arch.USE_CONVNEXT_BLOCKS_KEY: True,
    chiu_net_pp_arch.NUM_FREE_TARGETS_KEY: 4,
    chiu_net_pp_arch.CONSTRAINT_INDICES_KEY: numpy.array([1, 2, 3], dtype=int),
    chiu_net_pp_arch.GFS_FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    chiu_net_pp_arch.GFS_FC_MODULE_USE_3D_CONV: True,
    chiu_net_pp_arch.GFS_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    chiu_net_pp_arch.LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    chiu_net_pp_arch.LAGTGT_FC_MODULE_USE_3D_CONV: True,
    chiu_net_pp_arch.LAGTGT_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    chiu_net_pp_arch.NUM_LEVELS_KEY: MODEL_DEPTH,
    chiu_net_pp_arch.GFS_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(
        MODEL_DEPTH + 1, NUM_CONV_LAYERS_PER_BLOCK, dtype=int
    ),
    # chiu_net_pp_arch.GFS_ENCODER_NUM_CHANNELS_KEY: numpy.array(
    #     [50, 75, 100, 125, 150, 175, 200], dtype=int
    # ),
    chiu_net_pp_arch.GFS_ENCODER_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH + 1, 0.),
    chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(
        MODEL_DEPTH + 1, NUM_CONV_LAYERS_PER_BLOCK, dtype=int
    ),
    # chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY: numpy.array(
    #     [30, 40, 50, 60, 70, 80, 90], dtype=int
    # ),
    chiu_net_pp_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH + 1, 0.),
    chiu_net_pp_arch.DECODER_NUM_CONV_LAYERS_KEY: numpy.full(MODEL_DEPTH, 2, dtype=int),
    # chiu_net_pp_arch.DECODER_NUM_CHANNELS_KEY: numpy.array(
    #     [40, 58, 75, 93, 110, 128], dtype=int
    # ),
    chiu_net_pp_arch.UPSAMPLING_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH, 0.),
    chiu_net_pp_arch.SKIP_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH, 0.),
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
    # chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION
}

SPECTRAL_COMPLEXITY = 25
EXTRA_PREDICTOR_TYPE_STRINGS = [
    'gfs_3d', 'lagged_targets', 'era5_constants', 'residual_baseline'
]


def _run():
    """Makes templates for Experiment 23c (third step of ablation).

    This is effectively the main method.
    """

    for i in range(len(EXTRA_PREDICTOR_TYPE_STRINGS)):
        these_multipliers = numpy.linspace(
            0, MODEL_DEPTH, num=MODEL_DEPTH + 1, dtype=float
        )
        these_multipliers = numpy.round(
            2 ** these_multipliers
        ).astype(int)

        gfs_encoder_channel_counts = numpy.round(
            SPECTRAL_COMPLEXITY * these_multipliers
        ).astype(int)

        lagtgt_encoder_channel_counts = numpy.round(
            SPECTRAL_COMPLEXITY * these_multipliers
        ).astype(int)

        decoder_channel_counts = (
            gfs_encoder_channel_counts +
            lagtgt_encoder_channel_counts
        )[:-1]

        option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
        option_dict.update({
            chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION,
            chiu_net_pp_arch.LOSS_FUNCTION_KEY: LOSS_FUNCTION,
            chiu_net_pp_arch.METRIC_FUNCTIONS_KEY: METRIC_FUNCTIONS,
            chiu_net_pp_arch.GFS_ENCODER_NUM_CHANNELS_KEY:
                gfs_encoder_channel_counts,
            chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY:
                lagtgt_encoder_channel_counts,
            chiu_net_pp_arch.DECODER_NUM_CHANNELS_KEY:
                decoder_channel_counts,
            chiu_net_pp_arch.GFS_3D_DIMENSIONS_KEY: None,
            chiu_net_pp_arch.LAGTGT_DIMENSIONS_KEY:
                numpy.array([160, 160, 1, 7], dtype=int),
            chiu_net_pp_arch.USE_LEAD_TIME_AS_PRED_KEY: True,
            chiu_net_pp_arch.PREDN_BASELINE_DIMENSIONS_KEY: None,
            chiu_net_pp_arch.ERA5_CONST_DIMENSIONS_KEY: None
        })

        if EXTRA_PREDICTOR_TYPE_STRINGS[i] == 'gfs_3d':
            option_dict[chiu_net_pp_arch.GFS_3D_DIMENSIONS_KEY] = numpy.array(
                [160, 160, 1, 1, 5], dtype=int
            )
        elif EXTRA_PREDICTOR_TYPE_STRINGS[i] == 'era5_constants':
            option_dict[chiu_net_pp_arch.ERA5_CONST_DIMENSIONS_KEY] = (
                numpy.array([160, 160, 7], dtype=int)
            )
        elif EXTRA_PREDICTOR_TYPE_STRINGS[i] == 'residual_baseline':
            option_dict[chiu_net_pp_arch.PREDN_BASELINE_DIMENSIONS_KEY] = (
                numpy.array([160, 160, 7], dtype=int)
            )
        else:
            option_dict[chiu_net_pp_arch.LAGTGT_DIMENSIONS_KEY] = numpy.array(
                [160, 160, 2, 7], dtype=int
            )

        model_object = chiu_net_pp_arch.create_model(option_dict)
        output_file_name = '{0:s}/extra_predictor_type={1:s}/model.keras'.format(
            OUTPUT_DIR_NAME, EXTRA_PREDICTOR_TYPE_STRINGS[i]
        )

        print('Writing model to: "{0:s}"...'.format(output_file_name))
        file_system_utils.mkdir_recursive_if_necessary(
            file_name=output_file_name
        )
        model_object.save(
            filepath=output_file_name, overwrite=True, include_optimizer=True
        )

        metafile_name = neural_net.find_metafile(
            model_file_name=output_file_name,
            raise_error_if_missing=False
        )
        option_dict[chiu_net_pp_arch.LOSS_FUNCTION_KEY] = (
            LOSS_FUNCTION_STRING
        )
        option_dict[chiu_net_pp_arch.METRIC_FUNCTIONS_KEY] = (
            METRIC_FUNCTION_STRINGS
        )
        option_dict[chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY] = (
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
            metric_function_strings=METRIC_FUNCTION_STRINGS,
            optimizer_function_string=OPTIMIZER_FUNCTION_STRING,
            chiu_net_architecture_dict=None,
            chiu_net_pp_architecture_dict=option_dict,
            plateau_patience_epochs=10,
            plateau_learning_rate_multiplier=0.6,
            early_stopping_patience_epochs=50
        )


if __name__ == '__main__':
    _run()
