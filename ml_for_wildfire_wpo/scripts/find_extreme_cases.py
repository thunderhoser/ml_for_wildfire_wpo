"""Finds extreme cases."""

import re
import argparse
import numpy
import tensorflow
import shap
import shap.explainers
import keras.layers
import keras.models
# from keras import backend as K
from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.io import region_mask_io
from ml_for_wildfire_wpo.io import canadian_fwi_io
from ml_for_wildfire_wpo.io import prediction_io
from ml_for_wildfire_wpo.io import region_mask_io
from ml_for_wildfire_wpo.utils import misc_utils
from ml_for_wildfire_wpo.utils import canadian_fwi_utils
from ml_for_wildfire_wpo.machine_learning import neural_net

DATE_FORMAT = '%Y%m%d'

TARGET_QUANTITY_NAME = 'target'
PREDICTION_QUANTITY_NAME = 'prediction'
MODEL_ERROR_QUANTITY_NAME = 'prediction_minus_target'
VALID_QUANTITY_NAMES = [
    TARGET_QUANTITY_NAME, PREDICTION_QUANTITY_NAME, MODEL_ERROR_QUANTITY_NAME
]

MIN_STATISTIC_NAME = 'minimum'
MAX_STATISTIC_NAME = 'maximum'
MEAN_STATISTIC_NAME = 'mean'
VALID_STATISTIC_NAMES = [
    MIN_STATISTIC_NAME, MAX_STATISTIC_NAME, MEAN_STATISTIC_NAME
]

QUANTITY_ARG_NAME = 'quantity_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
INIT_DATE_LIMITS_ARG_NAME = 'init_date_limit_strings'
NUM_EXTREME_CASES_ARG_NAME = 'num_extreme_cases'
STATISTIC_ARG_NAME = 'statistic_name'
REGION_FILE_ARG_NAME = 'region_mask_file_name'
TARGET_FIELD_ARG_NAME = 'target_field_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

QUANTITY_HELP_STRING = (
    '"Extreme cases" will be defined based on this quantity.  Valid options '
    'are listed below:\n{0:s}'
).format(
    str(VALID_QUANTITY_NAMES)
)
TARGET_DIR_HELP_STRING = (
    'Path to directory with target fields.  Files therein will be found by '
    '`canadian_fwo_io.find_file` and read by `canadian_fwo_io.read_file`.  '
    'This argument is needed only if {0:s} == "{1:s}" or "{2:s}".'
).format(
    QUANTITY_ARG_NAME, TARGET_QUANTITY_NAME, MODEL_ERROR_QUANTITY_NAME
)
PREDICTION_DIR_HELP_STRING = (
    'Path to directory with predicted fields.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.  '
    'This argument is needed only if {0:s} == "{1:s}" or "{2:s}".'
).format(
    QUANTITY_ARG_NAME, PREDICTION_QUANTITY_NAME, MODEL_ERROR_QUANTITY_NAME
)
INIT_DATE_LIMITS_HELP_STRING = (
    'List with first and last init dates in search period (format "yyyymmdd").'
    '  Extreme cases will be based on this date range.'
)
NUM_EXTREME_CASES_HELP_STRING = (
    'Number of extreme cases to find in the date range specified by `{0:s}`.  '
    'Each "extreme case" will be one date.'
).format(
    INIT_DATE_LIMITS_ARG_NAME
)
STATISTIC_HELP_STRING = (
    '"Extreme cases" will be defined based on this spatial statistic, taken '
    'over the region specified by `{0:s}`, for the quantity specified by '
    '`{1:s}` and the field specified by `{2:s}`..  Valid statistic options are '
    'listed below:\n{3:s}'
).format(
    REGION_FILE_ARG_NAME, QUANTITY_ARG_NAME, TARGET_FIELD_ARG_NAME,
    str(VALID_STATISTIC_NAMES)
)
REGION_FILE_HELP_STRING = (
    'Path to file with region mask (will be read by '
    '`region_mask_io.read_file`).'
)
TARGET_FIELD_HELP_STRING = (
    '"Extreme cases" will be based on this target field (must be accepted by '
    '`canadian_fwi_utils.check_field_name`).'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Dates corresponding to extreme cases will be saved '
    'here.'
)

# TODO(thunderhoser): Still need format for output file.

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + QUANTITY_ARG_NAME, type=str, required=True,
    help=QUANTITY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=False, default='',
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=False, default='',
    help=PREDICTION_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_DATE_LIMITS_ARG_NAME, type=str, nargs=2, required=True,
    help=INIT_DATE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXTREME_CASES_ARG_NAME, type=int, required=True,
    help=NUM_EXTREME_CASES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + STATISTIC_ARG_NAME, type=str, required=True,
    help=STATISTIC_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + REGION_FILE_ARG_NAME, type=str, required=True,
    help=REGION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELD_ARG_NAME, type=str, required=True,
    help=TARGET_FIELD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(quantity_name, target_dir_name, prediction_dir_name,
         init_date_limit_strings, num_extreme_cases, statistic_name,
         region_mask_file_name, target_field_name, output_file_name):
    """Finds extreme cases.

    This is effectively the main method.

    :param quantity_name: See documentation at top of this script.
    :param target_dir_name: Same.
    :param prediction_dir_name: Same.
    :param init_date_limit_strings: Same.
    :param num_extreme_cases: Same.
    :param statistic_name: Same.
    :param region_mask_file_name: Same.
    :param target_field_name: Same.
    :param output_file_name: Same.
    """

    # TODO(thunderhoser): FUCK.  I just realized that I don't need to read
    # target values from a separate directory.  The prediction files contain
    # target values.

    assert quantity_name in VALID_QUANTITY_NAMES
    assert statistic_name in VALID_STATISTIC_NAMES
    error_checking.assert_is_greater(num_extreme_cases, 0)
    canadian_fwi_utils.check_field_name(target_field_name)

    if target_dir_name == '':
        target_dir_name = None
    if prediction_dir_name == '':
        prediction_dir_name = None

    if quantity_name in [TARGET_QUANTITY_NAME, MODEL_ERROR_QUANTITY_NAME]:
        assert target_dir_name is not None
    if quantity_name in [PREDICTION_QUANTITY_NAME, MODEL_ERROR_QUANTITY_NAME]:
        assert prediction_dir_name is not None

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    if prediction_dir_name is None:
        prediction_file_names = []
    else:
        prediction_file_names = prediction_io.find_files_for_period(
            directory_name=prediction_dir_name,
            first_init_date_string=init_date_limit_strings[0],
            last_init_date_string=init_date_limit_strings[1],
            raise_error_if_any_missing=False,
            raise_error_if_all_missing=True
        )

    # TODO(thunderhoser): This HACK should be nice code in prediction_io.py.
    regex_match_object = re.search(
        r'lead-time-days=(\d{2})', prediction_dir_name
    )
    assert regex_match_object
    model_lead_time_days = int(regex_match_object.group(1))

    if target_dir_name is None:
        target_file_names = []
    else:
        if prediction_dir_name is None:
            valid_date_strings = time_conversion.get_spc_dates_in_range(
                init_date_limit_strings[0],
                init_date_limit_strings[1]
            )
        else:
            init_date_strings = [
                prediction_io.file_name_to_date(f)
                for f in prediction_file_names
            ]
            init_dates_unix_sec = numpy.array([
                time_conversion.string_to_unix_sec(t, DATE_FORMAT)
                for t in init_date_strings
            ], dtype=int)

            valid_dates_unix_sec = (
                init_dates_unix_sec + model_lead_time_days * DAYS_TO_SECONDS
            )
            valid_date_strings = [
                time_conversion.unix_sec_to_string(t, DATE_FORMAT)
                for t in valid_dates_unix_sec
            ]

        target_file_names = [
            canadian_fwi_io.find_file(
                directory_name=target_dir_name,
                valid_date_string=d,
                raise_error_if_missing=True
            ) for d in valid_date_strings
        ]

    if prediction_dir_name is None:
        init_date_strings = [
            canadian_fwi_io.file_name_to_date(f) for f in target_file_names
        ]
    else:
        init_date_strings = [
            prediction_io.file_name_to_date(f) for f in prediction_file_names
        ]

    print('Reading region mask from: "{0:s}"...'.format(region_mask_file_name))
    mask_table_xarray = region_mask_io.read_file(region_mask_file_name)
    mtx = mask_table_xarray

    num_dates = len(init_date_strings)

    for i in range(num_dates):
        if prediction_dir_name is not None:
            print('Reading data from: "{0:s}"...'.format(
                prediction_file_names[i]
            ))
            prediction_table_xarray = prediction_io.read_file(
                prediction_file_names[i]
            )
            ptx = prediction_table_xarray

            desired_row_indices = misc_utils.desired_latitudes_to_rows(
                grid_latitudes_deg_n=ptx[prediction_io.LATITUDE_KEY].values,
                start_latitude_deg_n=
                numpy.min(mtx[region_mask_io.LATITUDE_KEY].values),
                end_latitude_deg_n=
                numpy.max(mtx[region_mask_io.LATITUDE_KEY].values)
            )
            desired_column_indices = misc_utils.desired_longitudes_to_columns(
                grid_longitudes_deg_e=ptx[prediction_io.LONGITUDE_KEY].values,
                start_longitude_deg_e=
                numpy.min(mtx[region_mask_io.LONGITUDE_KEY].values),
                end_longitude_deg_e=
                numpy.max(mtx[region_mask_io.LONGITUDE_KEY].values)
            )

            ptx = ptx.isel({prediction_io.ROW_DIM: desired_row_indices})
            ptx = ptx.isel({prediction_io.COLUMN_DIM: desired_column_indices})

            desired_field_indices = numpy.where(
                ptx[prediction_io.FIELD_NAME_KEY].values == target_field_name
            )[0]
            ptx = ptx.isel({prediction_io.FIELD_DIM: desired_field_indices})

            prediction_table_xarray = ptx
        else:
            prediction_table_xarray = None

        # TODO(thunderhoser): Find region, then take statistic over region.





if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        quantity_name=getattr(INPUT_ARG_OBJECT, QUANTITY_ARG_NAME),
        target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        prediction_dir_name=getattr(INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME),
        init_date_limit_strings=getattr(
            INPUT_ARG_OBJECT, INIT_DATE_LIMITS_ARG_NAME
        ),
        num_extreme_cases=getattr(INPUT_ARG_OBJECT, NUM_EXTREME_CASES_ARG_NAME),
        statistic_name=getattr(INPUT_ARG_OBJECT, STATISTIC_ARG_NAME),
        region_mask_file_name=getattr(INPUT_ARG_OBJECT, REGION_FILE_ARG_NAME),
        target_field_name=getattr(INPUT_ARG_OBJECT, TARGET_FIELD_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
