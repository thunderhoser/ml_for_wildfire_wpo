"""Contains list of input arguments for training a neural net."""

TEMPLATE_FILE_ARG_NAME = 'input_template_file_name'
OUTPUT_DIR_ARG_NAME = 'output_model_dir_name'

INNER_LATITUDE_LIMITS_ARG_NAME = 'inner_latitude_limits_deg_n'
INNER_LONGITUDE_LIMITS_ARG_NAME = 'inner_longitude_limits_deg_e'
OUTER_LATITUDE_BUFFER_ARG_NAME = 'outer_latitude_buffer_deg'
OUTER_LONGITUDE_BUFFER_ARG_NAME = 'outer_longitude_buffer_deg'
GFS_PREDICTORS_ARG_NAME = 'gfs_predictor_field_names'
GFS_PRESSURE_LEVELS_ARG_NAME = 'gfs_pressure_levels_mb'
GFS_PREDICTOR_LEADS_ARG_NAME = 'gfs_predictor_lead_times_hours'
GFS_NORM_FILE_ARG_NAME = 'gfs_normalization_file_name'
GFS_USE_QUANTILE_NORM_ARG_NAME = 'gfs_use_quantile_norm'
ERA5_CONSTANT_FILE_ARG_NAME = 'era5_constant_file_name'
ERA5_CONSTANT_PREDICTORS_ARG_NAME = 'era5_constant_predictor_field_names'
ERA5_NORM_FILE_ARG_NAME = 'era5_normalization_file_name'
ERA5_USE_QUANTILE_NORM_ARG_NAME = 'era5_use_quantile_norm'
TARGET_FIELDS_ARG_NAME = 'target_field_names'
MODEL_LEAD_TIMES_ARG_NAME = 'model_lead_times_days'
TARGET_LAG_TIMES_ARG_NAME = 'target_lag_times_days'
GFS_FCST_TARGET_LEAD_TIMES_ARG_NAME = 'gfs_forecast_target_lead_times_days'
TARGET_NORM_FILE_ARG_NAME = 'target_normalization_file_name'
TARGETS_USE_QUANTILE_NORM_ARG_NAME = 'targets_use_quantile_norm'
BATCH_SIZE_ARG_NAME = 'num_examples_per_batch'
SENTINEL_VALUE_ARG_NAME = 'sentinel_value'
DO_RESIDUAL_PRED_ARG_NAME = 'do_residual_prediction'

GFS_TRAINING_DIR_ARG_NAME = 'gfs_dir_name_for_training'
TARGET_TRAINING_DIR_ARG_NAME = 'target_dir_name_for_training'
GFS_FCST_TARGET_TRAINING_DIR_ARG_NAME = (
    'gfs_forecast_target_dir_name_for_training'
)
TRAINING_DATE_LIMITS_ARG_NAME = 'gfs_init_date_limit_strings_for_training'

GFS_VALIDATION_DIR_ARG_NAME = 'gfs_dir_name_for_validation'
TARGET_VALIDATION_DIR_ARG_NAME = 'target_dir_name_for_validation'
GFS_FCST_TARGET_VALIDATION_DIR_ARG_NAME = (
    'gfs_forecast_target_dir_name_for_validation'
)
VALIDATION_DATE_LIMITS_ARG_NAME = 'gfs_init_date_limit_strings_for_validation'

NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDATION_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'

PLATEAU_PATIENCE_ARG_NAME = 'plateau_patience_epochs'
PLATEAU_MULTIPLIER_ARG_NAME = 'plateau_learning_rate_multiplier'
EARLY_STOPPING_PATIENCE_ARG_NAME = 'early_stopping_patience_epochs'

TEMPLATE_FILE_HELP_STRING = (
    'Path to template file, containing model architecture.  This will be read '
    'by `neural_net.read_model`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Trained model will be saved here.'
)
INNER_LATITUDE_LIMITS_HELP_STRING = (
    'Length-2 list with meridional limits (deg north) of bounding box for '
    'inner (target) domain.'
)
INNER_LONGITUDE_LIMITS_HELP_STRING = (
    'Same as {0:s} but for longitudes (deg east).'
).format(INNER_LATITUDE_LIMITS_ARG_NAME)

OUTER_LATITUDE_BUFFER_HELP_STRING = (
    'Meridional buffer between inner and outer domains.'
)
OUTER_LONGITUDE_BUFFER_HELP_STRING = (
    'Zonal buffer between inner and outer domains.'
)
GFS_PREDICTORS_HELP_STRING = (
    'List with names of GFS fields to be used as predictors.'
)
GFS_PRESSURE_LEVELS_HELP_STRING = (
    'List of pressure levels to be used for GFS predictors (will be applied to '
    'every 3-D predictor in the list {0:s}).'
).format(GFS_PREDICTORS_ARG_NAME)

GFS_PREDICTOR_LEADS_HELP_STRING = (
    'List of lead times to be used for GFS predictors.'
)
GFS_NORM_FILE_HELP_STRING = (
    'Path to file with normalization params for GFS predictors (will be read '
    'by `gfs_io.read_normalization_file`).  If you do not want to normalize '
    'GFS predictors, leave this argument alone.'
)
GFS_USE_QUANTILE_NORM_HELP_STRING = (
    'Boolean flag.  If 1, GFS predictors will be converted to quantiles and '
    'then quantiles will be converted to standard normal distribution.  If 0, '
    'will convert straight to z-scores.'
)
ERA5_CONSTANT_PREDICTORS_HELP_STRING = (
    'List with names of ERA5-constant fields to be used as predictors.  If you '
    'do not want to use ERA5, leave this argument alone.'
)
ERA5_CONSTANT_FILE_HELP_STRING = (
    'Path to file with ERA5 constants (will be read by '
    '`era5_constant_io.read_file`).  If you do not want to use ERA5, leave '
    'this argument alone.'
)
ERA5_NORM_FILE_HELP_STRING = (
    'Path to file with normalization params for ERA5-constant predictors (will '
    'be read by `era5_constant_io.read_normalization_file`).  If you do not '
    'want to normalize ERA5 predictors, leave this argument alone.'
)
ERA5_USE_QUANTILE_NORM_HELP_STRING = 'Same as {0:s} but for ERA5 data.'.format(
    GFS_USE_QUANTILE_NORM_ARG_NAME
)
TARGET_FIELDS_HELP_STRING = 'List of target fields (fire-weather indices).'
MODEL_LEAD_TIMES_HELP_STRING = '1-D list of model lead times.'
TARGET_LAG_TIMES_HELP_STRING = (
    'List of lag times to be used for lagged-target predictors.'
)
GFS_FCST_TARGET_LEAD_TIMES_HELP_STRING = (
    'List of lead times to be used for lead-target predictors.  A '
    '"lead-target predictor" is the raw-GFS forecast of the target field (fire '
    'weather index) at one lead time.'
)
TARGET_NORM_FILE_HELP_STRING = (
    'Path to file with normalization params for lagged-target predictors (will '
    'be read by `canadian_fwi_io.read_normalization_file`).  If you do not '
    'want to normalize lagged-target predictors, leave this argument alone.'
)
TARGETS_USE_QUANTILE_NORM_HELP_STRING = (
    'Same as {0:s} but for target fields.'
).format(
    GFS_USE_QUANTILE_NORM_ARG_NAME
)
BATCH_SIZE_HELP_STRING = 'Number of data examples per batch.'
SENTINEL_VALUE_HELP_STRING = (
    'All NaN predictors will be replaced with this value.'
)
DO_RESIDUAL_PRED_HELP_STRING = (
    'Boolean flag.  If 1, the NN predicts only the residual, with the baseline '
    '(most recent lagged target fields) being added at the end.  If 0, the NN '
    'just predicts the raw target fields like a normal robot.'
)

GFS_TRAINING_DIR_HELP_STRING = (
    'Name of directory with GFS data for training set.  Files therein will be '
    'found by `gfs_io.find_file` and read by `gfs_io.read_file`.'
)
TARGET_TRAINING_DIR_HELP_STRING = (
    'Name of directory with target fields.  Files therein will be found by '
    '`canadian_fwo_io.find_file` and read by `canadian_fwo_io.read_file`.'
)
GFS_FCST_TARGET_TRAINING_DIR_HELP_STRING = (
    'Name of directory with raw-GFS-forecast target fields.  Files therein '
    'will be found by `gfs_daily_io.find_file` and read by '
    '`gfs_daily_io.read_file`.'
)
TRAINING_DATE_LIMITS_HELP_STRING = (
    'Length-2 list with first and last GFS model runs (init times in format '
    '"yyyymmdd") to be used for training.'
)

GFS_VALIDATION_DIR_HELP_STRING = 'Same as {0:s} but for validation.'.format(
    GFS_TRAINING_DIR_ARG_NAME
)
TARGET_VALIDATION_DIR_HELP_STRING = 'Same as {0:s} but for validation.'.format(
    TARGET_TRAINING_DIR_ARG_NAME
)
GFS_FCST_TARGET_VALIDATION_DIR_HELP_STRING = (
    'Same as {0:s} but for validation.'
).format(
    GFS_FCST_TARGET_TRAINING_DIR_ARG_NAME
)
VALIDATION_DATE_LIMITS_HELP_STRING = 'Same as {0:s} but for validation.'.format(
    TRAINING_DATE_LIMITS_ARG_NAME
)

NUM_EPOCHS_HELP_STRING = 'Number of epochs.'
NUM_TRAINING_BATCHES_HELP_STRING = 'Number of training batches per epoch.'
NUM_VALIDATION_BATCHES_HELP_STRING = 'Number of validation batches per epoch.'

PLATEAU_PATIENCE_HELP_STRING = (
    'Training will be deemed to have reached "plateau" if validation loss has '
    'not decreased in the last N epochs, where N = {0:s}.'
).format(PLATEAU_PATIENCE_ARG_NAME)

PLATEAU_MULTIPLIER_HELP_STRING = (
    'If training reaches "plateau," learning rate will be multiplied by this '
    'value in range (0, 1).'
)
EARLY_STOPPING_PATIENCE_HELP_STRING = (
    'Training will be stopped early if validation loss has not decreased in '
    'the last N epochs, where N = {0:s}.'
).format(EARLY_STOPPING_PATIENCE_ARG_NAME)


def add_input_args(parser_object):
    """Adds input args to ArgumentParser object.

    :param parser_object: Instance of `argparse.ArgumentParser` (may already
        contain some input args).
    :return: parser_object: Same as input but with new args added.
    """

    parser_object.add_argument(
        '--' + TEMPLATE_FILE_ARG_NAME, type=str, required=True,
        help=TEMPLATE_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
        help=OUTPUT_DIR_HELP_STRING
    )

    parser_object.add_argument(
        '--' + INNER_LATITUDE_LIMITS_ARG_NAME, type=float, nargs=2,
        required=True, help=INNER_LATITUDE_LIMITS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + INNER_LONGITUDE_LIMITS_ARG_NAME, type=float, nargs=2,
        required=True, help=INNER_LONGITUDE_LIMITS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + OUTER_LATITUDE_BUFFER_ARG_NAME, type=float,
        required=True, help=OUTER_LATITUDE_BUFFER_HELP_STRING
    )
    parser_object.add_argument(
        '--' + OUTER_LONGITUDE_BUFFER_ARG_NAME, type=float,
        required=True, help=OUTER_LONGITUDE_BUFFER_HELP_STRING
    )
    parser_object.add_argument(
        '--' + GFS_PREDICTORS_ARG_NAME, type=str, nargs='+',
        required=True, help=GFS_PREDICTORS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + GFS_PRESSURE_LEVELS_ARG_NAME, type=int, nargs='+',
        required=False, default=[500], help=GFS_PRESSURE_LEVELS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + GFS_PREDICTOR_LEADS_ARG_NAME, type=int, nargs='+',
        required=True, help=GFS_PREDICTOR_LEADS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + GFS_NORM_FILE_ARG_NAME, type=str,
        required=False, default='', help=GFS_NORM_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + GFS_USE_QUANTILE_NORM_ARG_NAME, type=int,
        required=False, default=0, help=GFS_USE_QUANTILE_NORM_HELP_STRING
    )
    parser_object.add_argument(
        '--' + ERA5_CONSTANT_FILE_ARG_NAME, type=str,
        required=False, default='', help=ERA5_CONSTANT_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + ERA5_CONSTANT_PREDICTORS_ARG_NAME, type=str, nargs='+',
        required=False, default=[''], help=ERA5_CONSTANT_PREDICTORS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + ERA5_NORM_FILE_ARG_NAME, type=str,
        required=False, default='', help=ERA5_NORM_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + ERA5_USE_QUANTILE_NORM_ARG_NAME, type=int,
        required=False, default=0, help=ERA5_USE_QUANTILE_NORM_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TARGET_FIELDS_ARG_NAME, type=str, nargs='+',
        required=True, help=TARGET_FIELDS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + MODEL_LEAD_TIMES_ARG_NAME, type=int, nargs='+',
        required=True, help=MODEL_LEAD_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TARGET_LAG_TIMES_ARG_NAME, type=int, nargs='+',
        required=True, help=TARGET_LAG_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + GFS_FCST_TARGET_LEAD_TIMES_ARG_NAME, type=int, nargs='+',
        required=True, help=GFS_FCST_TARGET_LEAD_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TARGET_NORM_FILE_ARG_NAME, type=str,
        required=False, default='', help=TARGET_NORM_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TARGETS_USE_QUANTILE_NORM_ARG_NAME, type=int,
        required=False, default=0, help=TARGETS_USE_QUANTILE_NORM_HELP_STRING
    )
    parser_object.add_argument(
        '--' + BATCH_SIZE_ARG_NAME, type=int,
        required=True, help=BATCH_SIZE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SENTINEL_VALUE_ARG_NAME, type=float,
        required=True, help=SENTINEL_VALUE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + DO_RESIDUAL_PRED_ARG_NAME, type=int, required=False, default=0,
        help=DO_RESIDUAL_PRED_HELP_STRING
    )

    parser_object.add_argument(
        '--' + GFS_TRAINING_DIR_ARG_NAME, type=str,
        required=True, help=GFS_TRAINING_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TARGET_TRAINING_DIR_ARG_NAME, type=str,
        required=True, help=TARGET_TRAINING_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + GFS_FCST_TARGET_TRAINING_DIR_ARG_NAME, type=str,
        required=True, help=GFS_FCST_TARGET_TRAINING_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_DATE_LIMITS_ARG_NAME, type=str, nargs=2,
        required=True, help=TRAINING_DATE_LIMITS_HELP_STRING
    )

    parser_object.add_argument(
        '--' + GFS_VALIDATION_DIR_ARG_NAME, type=str,
        required=True, help=GFS_VALIDATION_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TARGET_VALIDATION_DIR_ARG_NAME, type=str,
        required=True, help=TARGET_VALIDATION_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + GFS_FCST_TARGET_VALIDATION_DIR_ARG_NAME, type=str,
        required=True, help=GFS_FCST_TARGET_VALIDATION_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VALIDATION_DATE_LIMITS_ARG_NAME, type=str, nargs=2,
        required=True, help=VALIDATION_DATE_LIMITS_HELP_STRING
    )

    parser_object.add_argument(
        '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False, default=1000,
        help=NUM_EPOCHS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=False,
        default=32, help=NUM_TRAINING_BATCHES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_VALIDATION_BATCHES_ARG_NAME, type=int, required=False,
        default=16, help=NUM_VALIDATION_BATCHES_HELP_STRING
    )

    parser_object.add_argument(
        '--' + PLATEAU_PATIENCE_ARG_NAME, type=int, required=False,
        default=10, help=PLATEAU_PATIENCE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PLATEAU_MULTIPLIER_ARG_NAME, type=float, required=False,
        default=0.9, help=PLATEAU_MULTIPLIER_HELP_STRING
    )
    parser_object.add_argument(
        '--' + EARLY_STOPPING_PATIENCE_ARG_NAME, type=int, required=False,
        default=100, help=EARLY_STOPPING_PATIENCE_HELP_STRING
    )

    return parser_object
