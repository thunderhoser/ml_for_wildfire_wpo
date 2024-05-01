"""Trains neural net."""

import os
import sys
import copy
import argparse
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import neural_net
import chiu_net_pp_architecture_recurrent as chiu_net_pp_arch
import architecture_utils
import custom_losses
import training_args

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = training_args.add_input_args(parser_object=INPUT_ARG_PARSER)

CHANNEL_WEIGHTS = numpy.array([0.02562263, 0.00373885, 0.00008940, 0.60291427, 0.00251213, 0.08761268, 0.27751004])
MAX_DUAL_WEIGHTS = numpy.array([96.4105, 303.9126, 1741.9033, 23.1660, 361.6984, 61.3669, 40.1856])

LOSS_FUNCTION = custom_losses.dual_weighted_mse_constrained_dsr(
    channel_weights=CHANNEL_WEIGHTS,
    max_dual_weight_by_channel=MAX_DUAL_WEIGHTS,
    fwi_index=5,
    function_name='loss_dwmse',
    expect_ensemble=False,
    is_nn_evidential=False
)

LOSS_FUNCTION_STRING = (
    'custom_losses.dual_weighted_mse_constrained_dsr('
    'channel_weights=numpy.array([0.02562263, 0.00373885, 0.00008940, 0.60291427, 0.00251213, 0.08761268, 0.27751004]), '
    'max_dual_weight_by_channel=numpy.array([96.4105, 303.9126, 1741.9033, 23.1660, 361.6984, 61.3669, 40.1856]), '
    'fwi_index=5, '
    'function_name="loss_dwcrps",'
    'expect_ensemble=False,'
    'is_nn_evidential=False)'
)

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
    chiu_net_pp_arch.ENSEMBLE_SIZE_KEY: 1,
    chiu_net_pp_arch.USE_EVIDENTIAL_KEY: False,
    # chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION,
    chiu_net_pp_arch.USE_RESIDUAL_BLOCKS_KEY: True,
}

NUM_GFS_PRESSURE_LEVELS = 2
NUM_GFS_2D_VARS = 5
NUM_GFS_3D_VARS = 5

NN_LEAD_TIMES_DAYS = numpy.linspace(1, 9, num=9, dtype=int)

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


def _create_model():
    """Creates model.

    :return: model_object: Model.
    """

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

    return chiu_net_pp_arch.create_model(
        option_dict=option_dict,
        loss_function=LOSS_FUNCTION,
        metric_list=[]
    )


def _run(template_file_name, output_dir_name,
         inner_latitude_limits_deg_n, inner_longitude_limits_deg_e,
         outer_latitude_buffer_deg, outer_longitude_buffer_deg,
         gfs_predictor_field_names, gfs_pressure_levels_mb,
         gfs_predictor_lead_times_hours, gfs_normalization_file_name,
         gfs_use_quantile_norm,
         era5_constant_file_name, era5_constant_predictor_field_names,
         era5_normalization_file_name, era5_use_quantile_norm,
         target_field_names, model_lead_times_days, target_lag_times_days,
         gfs_forecast_target_lead_times_days, target_normalization_file_name,
         targets_use_quantile_norm,
         num_examples_per_batch, sentinel_value, do_residual_prediction,
         gfs_dir_name_for_training, target_dir_name_for_training,
         gfs_forecast_target_dir_name_for_training,
         init_date_limit_strings_for_training,
         gfs_dir_name_for_validation, target_dir_name_for_validation,
         gfs_forecast_target_dir_name_for_validation,
         init_date_limit_strings_for_validation,
         num_epochs, num_training_batches_per_epoch,
         num_validation_batches_per_epoch,
         plateau_patience_epochs, plateau_learning_rate_multiplier,
         early_stopping_patience_epochs):
    """Trains neural net.

    This is effectively the main method.

    :param template_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    :param inner_latitude_limits_deg_n: Same.
    :param inner_longitude_limits_deg_e: Same.
    :param outer_latitude_buffer_deg: Same.
    :param outer_longitude_buffer_deg: Same.
    :param gfs_predictor_field_names: Same.
    :param gfs_pressure_levels_mb: Same.
    :param gfs_predictor_lead_times_hours: Same.
    :param gfs_normalization_file_name: Same.
    :param gfs_use_quantile_norm: Same.
    :param era5_constant_file_name: Same.
    :param era5_constant_predictor_field_names: Same.
    :param era5_normalization_file_name: Same.
    :param era5_use_quantile_norm: Same.
    :param target_field_names: Same.
    :param model_lead_times_days: Same.
    :param target_lag_times_days: Same.
    :param gfs_forecast_target_lead_times_days: Same.
    :param target_normalization_file_name: Same.
    :param targets_use_quantile_norm: Same.
    :param num_examples_per_batch: Same.
    :param sentinel_value: Same.
    :param do_residual_prediction: Same.
    :param gfs_dir_name_for_training: Same.
    :param target_dir_name_for_training: Same.
    :param gfs_forecast_target_dir_name_for_training: Same.
    :param init_date_limit_strings_for_training: Same.
    :param gfs_dir_name_for_validation: Same.
    :param target_dir_name_for_validation: Same.
    :param gfs_forecast_target_dir_name_for_validation: Same.
    :param init_date_limit_strings_for_validation: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    """

    if len(gfs_pressure_levels_mb) == 1 and gfs_pressure_levels_mb[0] <= 0:
        gfs_pressure_levels_mb = None
    if gfs_normalization_file_name == '':
        gfs_normalization_file_name = None
    if target_normalization_file_name == '':
        target_normalization_file_name = None
    if era5_normalization_file_name == '':
        era5_normalization_file_name = None

    if era5_constant_file_name == '':
        era5_constant_file_name = None
        era5_constant_predictor_field_names = None
    elif (
            len(era5_constant_predictor_field_names) == 1 and
            era5_constant_predictor_field_names[0] in ['', 'None']
    ):
        era5_constant_predictor_field_names = None

    if (
            len(gfs_forecast_target_lead_times_days) == 1 and
            gfs_forecast_target_lead_times_days[0] <= 0
    ):
        gfs_forecast_target_lead_times_days = None
        gfs_forecast_target_dir_name_for_training = None
        gfs_forecast_target_dir_name_for_validation = None

    if (
            gfs_forecast_target_dir_name_for_training is None or
            gfs_forecast_target_dir_name_for_validation is None
    ):
        gfs_forecast_target_lead_times_days = None
        gfs_forecast_target_dir_name_for_training = None
        gfs_forecast_target_dir_name_for_validation = None

    training_option_dict = {
        neural_net.INNER_LATITUDE_LIMITS_KEY: inner_latitude_limits_deg_n,
        neural_net.INNER_LONGITUDE_LIMITS_KEY: inner_longitude_limits_deg_e,
        neural_net.OUTER_LATITUDE_BUFFER_KEY: outer_latitude_buffer_deg,
        neural_net.OUTER_LONGITUDE_BUFFER_KEY: outer_longitude_buffer_deg,
        neural_net.GFS_PREDICTOR_FIELDS_KEY: gfs_predictor_field_names,
        neural_net.GFS_PRESSURE_LEVELS_KEY: gfs_pressure_levels_mb,
        neural_net.GFS_PREDICTOR_LEADS_KEY: gfs_predictor_lead_times_hours,
        neural_net.GFS_NORM_FILE_KEY: gfs_normalization_file_name,
        neural_net.GFS_USE_QUANTILE_NORM_KEY: gfs_use_quantile_norm,
        neural_net.ERA5_CONSTANT_PREDICTOR_FIELDS_KEY:
            era5_constant_predictor_field_names,
        neural_net.ERA5_CONSTANT_FILE_KEY: era5_constant_file_name,
        neural_net.ERA5_NORM_FILE_KEY: era5_normalization_file_name,
        neural_net.ERA5_USE_QUANTILE_NORM_KEY: era5_use_quantile_norm,
        neural_net.TARGET_FIELDS_KEY: target_field_names,
        neural_net.MODEL_LEAD_TIMES_KEY: model_lead_times_days,
        neural_net.TARGET_LAG_TIMES_KEY: target_lag_times_days,
        neural_net.GFS_FCST_TARGET_LEAD_TIMES_KEY:
            gfs_forecast_target_lead_times_days,
        neural_net.TARGET_NORM_FILE_KEY: target_normalization_file_name,
        neural_net.TARGETS_USE_QUANTILE_NORM_KEY: targets_use_quantile_norm,
        neural_net.BATCH_SIZE_KEY: num_examples_per_batch,
        neural_net.SENTINEL_VALUE_KEY: sentinel_value,
        neural_net.DO_RESIDUAL_PREDICTION_KEY: do_residual_prediction,
        neural_net.INIT_DATE_LIMITS_KEY: init_date_limit_strings_for_training,
        neural_net.GFS_DIRECTORY_KEY: gfs_dir_name_for_training,
        neural_net.TARGET_DIRECTORY_KEY: target_dir_name_for_training,
        neural_net.GFS_FORECAST_TARGET_DIR_KEY:
            gfs_forecast_target_dir_name_for_training
    }

    validation_option_dict = {
        neural_net.INIT_DATE_LIMITS_KEY: init_date_limit_strings_for_validation,
        neural_net.GFS_DIRECTORY_KEY: gfs_dir_name_for_validation,
        neural_net.TARGET_DIRECTORY_KEY: target_dir_name_for_validation,
        neural_net.GFS_FORECAST_TARGET_DIR_KEY:
            gfs_forecast_target_dir_name_for_validation
    }

    # print('Reading model template from: "{0:s}"...'.format(template_file_name))
    # model_object = neural_net.read_model(hdf5_file_name=template_file_name)

    model_object = _create_model()

    model_metafile_name = neural_net.find_metafile(
        model_file_name=template_file_name, raise_error_if_missing=True
    )
    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    neural_net.train_model(
        model_object=model_object,
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        loss_function_string=model_metadata_dict[neural_net.LOSS_FUNCTION_KEY],
        optimizer_function_string=
        model_metadata_dict[neural_net.OPTIMIZER_FUNCTION_KEY],
        metric_function_strings=
        model_metadata_dict[neural_net.METRIC_FUNCTIONS_KEY],
        plateau_patience_epochs=plateau_patience_epochs,
        plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        output_dir_name=output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        template_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.TEMPLATE_FILE_ARG_NAME
        ),
        output_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.OUTPUT_DIR_ARG_NAME
        ),
        inner_latitude_limits_deg_n=numpy.array(
            getattr(
                INPUT_ARG_OBJECT, training_args.INNER_LATITUDE_LIMITS_ARG_NAME
            ),
            dtype=float
        ),
        inner_longitude_limits_deg_e=numpy.array(
            getattr(
                INPUT_ARG_OBJECT, training_args.INNER_LONGITUDE_LIMITS_ARG_NAME
            ),
            dtype=float
        ),
        outer_latitude_buffer_deg=getattr(
            INPUT_ARG_OBJECT, training_args.OUTER_LATITUDE_BUFFER_ARG_NAME
        ),
        outer_longitude_buffer_deg=getattr(
            INPUT_ARG_OBJECT, training_args.OUTER_LONGITUDE_BUFFER_ARG_NAME
        ),
        gfs_predictor_field_names=getattr(
            INPUT_ARG_OBJECT, training_args.GFS_PREDICTORS_ARG_NAME
        ),
        gfs_pressure_levels_mb=numpy.array(
            getattr(
                INPUT_ARG_OBJECT, training_args.GFS_PRESSURE_LEVELS_ARG_NAME
            ),
            dtype=int
        ),
        gfs_predictor_lead_times_hours=numpy.array(
            getattr(
                INPUT_ARG_OBJECT, training_args.GFS_PREDICTOR_LEADS_ARG_NAME
            ),
            dtype=int
        ),
        gfs_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.GFS_NORM_FILE_ARG_NAME
        ),
        gfs_use_quantile_norm=bool(getattr(
            INPUT_ARG_OBJECT, training_args.GFS_USE_QUANTILE_NORM_ARG_NAME
        )),
        era5_constant_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.ERA5_CONSTANT_FILE_ARG_NAME
        ),
        era5_constant_predictor_field_names=getattr(
            INPUT_ARG_OBJECT, training_args.ERA5_CONSTANT_PREDICTORS_ARG_NAME
        ),
        era5_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.ERA5_NORM_FILE_ARG_NAME
        ),
        era5_use_quantile_norm=bool(getattr(
            INPUT_ARG_OBJECT, training_args.ERA5_USE_QUANTILE_NORM_ARG_NAME
        )),
        target_field_names=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_FIELDS_ARG_NAME
        ),
        model_lead_times_days=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.MODEL_LEAD_TIMES_ARG_NAME),
            dtype=int
        ),
        target_lag_times_days=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.TARGET_LAG_TIMES_ARG_NAME),
            dtype=int
        ),
        gfs_forecast_target_lead_times_days=numpy.array(
            getattr(
                INPUT_ARG_OBJECT,
                training_args.GFS_FCST_TARGET_LEAD_TIMES_ARG_NAME
            ),
            dtype=int
        ),
        target_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_NORM_FILE_ARG_NAME
        ),
        targets_use_quantile_norm=bool(getattr(
            INPUT_ARG_OBJECT, training_args.TARGETS_USE_QUANTILE_NORM_ARG_NAME
        )),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, training_args.BATCH_SIZE_ARG_NAME
        ),
        sentinel_value=getattr(
            INPUT_ARG_OBJECT, training_args.SENTINEL_VALUE_ARG_NAME
        ),
        do_residual_prediction=bool(getattr(
            INPUT_ARG_OBJECT, training_args.DO_RESIDUAL_PRED_ARG_NAME
        )),
        gfs_dir_name_for_training=getattr(
            INPUT_ARG_OBJECT, training_args.GFS_TRAINING_DIR_ARG_NAME
        ),
        target_dir_name_for_training=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_TRAINING_DIR_ARG_NAME
        ),
        gfs_forecast_target_dir_name_for_training=getattr(
            INPUT_ARG_OBJECT,
            training_args.GFS_FCST_TARGET_TRAINING_DIR_ARG_NAME
        ),
        init_date_limit_strings_for_training=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_DATE_LIMITS_ARG_NAME
        ),
        gfs_dir_name_for_validation=getattr(
            INPUT_ARG_OBJECT, training_args.GFS_VALIDATION_DIR_ARG_NAME
        ),
        target_dir_name_for_validation=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_VALIDATION_DIR_ARG_NAME
        ),
        gfs_forecast_target_dir_name_for_validation=getattr(
            INPUT_ARG_OBJECT,
            training_args.GFS_FCST_TARGET_VALIDATION_DIR_ARG_NAME
        ),
        init_date_limit_strings_for_validation=getattr(
            INPUT_ARG_OBJECT, training_args.VALIDATION_DATE_LIMITS_ARG_NAME
        ),
        num_epochs=getattr(INPUT_ARG_OBJECT, training_args.NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_TRAINING_BATCHES_ARG_NAME
        ),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_VALIDATION_BATCHES_ARG_NAME
        ),
        plateau_patience_epochs=getattr(
            INPUT_ARG_OBJECT, training_args.PLATEAU_PATIENCE_ARG_NAME
        ),
        plateau_learning_rate_multiplier=getattr(
            INPUT_ARG_OBJECT, training_args.PLATEAU_MULTIPLIER_ARG_NAME
        ),
        early_stopping_patience_epochs=getattr(
            INPUT_ARG_OBJECT, training_args.EARLY_STOPPING_PATIENCE_ARG_NAME
        )
    )
