"""Trains neural net."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import neural_net
import training_args

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = training_args.add_input_args(parser_object=INPUT_ARG_PARSER)


def _run(template_file_name, output_dir_name,
         inner_latitude_limits_deg_n, inner_longitude_limits_deg_e,
         outer_latitude_buffer_deg, outer_longitude_buffer_deg,
         gfs_predictor_field_names, gfs_pressure_levels_mb,
         gfs_predictor_lead_times_hours, gfs_normalization_file_name,
         gfs_use_quantile_norm,
         era5_constant_file_name, era5_constant_predictor_field_names,
         era5_normalization_file_name, era5_use_quantile_norm,
         target_field_names, target_lead_time_days, target_lag_times_days,
         gfs_forecast_target_lead_times_days, target_normalization_file_name,
         targets_use_quantile_norm,
         num_examples_per_batch, sentinel_value,
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
    :param target_lead_time_days: Same.
    :param target_lag_times_days: Same.
    :param gfs_forecast_target_lead_times_days: Same.
    :param target_normalization_file_name: Same.
    :param targets_use_quantile_norm: Same.
    :param num_examples_per_batch: Same.
    :param sentinel_value: Same.
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
            era5_constant_predictor_field_names[0] == ''
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
        neural_net.TARGET_LEAD_TIME_KEY: target_lead_time_days,
        neural_net.TARGET_LAG_TIMES_KEY: target_lag_times_days,
        neural_net.GFS_FCST_TARGET_LEAD_TIMES_KEY:
            gfs_forecast_target_lead_times_days,
        neural_net.TARGET_NORM_FILE_KEY: target_normalization_file_name,
        neural_net.TARGETS_USE_QUANTILE_NORM_KEY: targets_use_quantile_norm,
        neural_net.BATCH_SIZE_KEY: num_examples_per_batch,
        neural_net.SENTINEL_VALUE_KEY: sentinel_value,
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

    print('Reading model template from: "{0:s}"...'.format(template_file_name))
    model_object = neural_net.read_model(hdf5_file_name=template_file_name)

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
        target_lead_time_days=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_LEAD_TIME_ARG_NAME
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
