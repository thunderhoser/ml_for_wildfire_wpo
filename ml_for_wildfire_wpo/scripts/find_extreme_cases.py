"""Finds extreme cases."""

import re
import copy
import argparse
import numpy
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.io import prediction_io
from ml_for_wildfire_wpo.io import region_mask_io
from ml_for_wildfire_wpo.io import extreme_cases_io
from ml_for_wildfire_wpo.utils import misc_utils

DATE_FORMAT = '%Y%m%d'

QUANTITY_ARG_NAME = 'quantity_name'
PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
INIT_DATE_LIMITS_ARG_NAME = 'init_date_limit_strings'
NUM_EXTREME_CASES_ARG_NAME = 'num_extreme_cases'
SPATIAL_STAT_ARG_NAME = 'spatial_statistic_name'
REGION_FILE_ARG_NAME = 'region_mask_file_name'
REGION_NAME_ARG_NAME = 'region_name'
TARGET_FIELD_ARG_NAME = 'target_field_name'
OUTPUT_FILES_ARG_NAME = 'output_file_names'

QUANTITY_HELP_STRING = (
    '"Extreme cases" will be defined based on this quantity.  Valid options '
    'are listed below:\n{0:s}'
).format(
    str(extreme_cases_io.VALID_QUANTITY_NAMES)
)
PREDICTION_DIR_HELP_STRING = (
    'Path to directory with prediction files (containing both actual and '
    'predicted values).  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
INIT_DATE_LIMITS_HELP_STRING = (
    'List with first and last init dates in search period (format "yyyymmdd").'
    '  Extreme cases will be based on this date range.'
)
NUM_EXTREME_CASES_HELP_STRING = (
    'Number of extreme cases to find in the date range specified by `{0:s}`.  '
    'Each "extreme case" will be one date.  Half of the extreme cases will be '
    'dates with lowest values, and the other half will be dates with highest '
    'values.'
).format(
    INIT_DATE_LIMITS_ARG_NAME
)
SPATIAL_STAT_HELP_STRING = (
    '"Extreme cases" will be defined based on this spatial statistic, taken '
    'over the region specified by `{0:s}`, for the quantity specified by '
    '`{1:s}` and the field specified by `{2:s}`.  Valid statistic options are '
    'listed below:\n{3:s}'
).format(
    REGION_FILE_ARG_NAME, QUANTITY_ARG_NAME, TARGET_FIELD_ARG_NAME,
    str(extreme_cases_io.VALID_SPATIAL_STAT_NAMES)
)
REGION_FILE_HELP_STRING = (
    'Path to file with region mask (will be read by '
    '`region_mask_io.read_file`).'
)
REGION_NAME_HELP_STRING = (
    'Human-readable name of region encoded in `{0:s}`.'
).format(
    REGION_FILE_ARG_NAME
)
TARGET_FIELD_HELP_STRING = (
    '"Extreme cases" will be based on this target field (must be accepted by '
    '`canadian_fwi_utils.check_field_name`).'
)
OUTPUT_FILES_HELP_STRING = (
    'Length-2 list (space-separated) of paths to output files.  Dates with '
    'lowest (highest) values will be written to the first (second) NetCDF file '
    'by `extreme_cases_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + QUANTITY_ARG_NAME, type=str, required=True,
    help=QUANTITY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
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
    '--' + SPATIAL_STAT_ARG_NAME, type=str, required=True,
    help=SPATIAL_STAT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + REGION_FILE_ARG_NAME, type=str, required=True,
    help=REGION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + REGION_NAME_ARG_NAME, type=str, required=True,
    help=REGION_NAME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELD_ARG_NAME, type=str, required=True,
    help=TARGET_FIELD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILES_ARG_NAME, type=str, nargs=2, required=True,
    help=OUTPUT_FILES_HELP_STRING
)


def _run(quantity_name, prediction_dir_name, init_date_limit_strings,
         num_extreme_cases, spatial_statistic_name, region_mask_file_name,
         region_name, target_field_name, output_file_names):
    """Finds extreme cases.

    This is effectively the main method.

    :param quantity_name: See documentation at top of this script.
    :param prediction_dir_name: Same.
    :param init_date_limit_strings: Same.
    :param num_extreme_cases: Same.
    :param spatial_statistic_name: Same.
    :param region_mask_file_name: Same.
    :param region_name: Same.
    :param target_field_name: Same.
    :param output_file_names: Same.
    """

    assert quantity_name in extreme_cases_io.VALID_QUANTITY_NAMES
    assert spatial_statistic_name in extreme_cases_io.VALID_SPATIAL_STAT_NAMES
    error_checking.assert_is_greater(num_extreme_cases, 0)

    num_extreme_cases = int(number_rounding.ceiling_to_nearest(
        num_extreme_cases, 2
    ))

    prediction_file_names = prediction_io.find_files_for_period(
        directory_name=prediction_dir_name,
        first_init_date_string=init_date_limit_strings[0],
        last_init_date_string=init_date_limit_strings[1],
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )

    init_date_strings = [
        prediction_io.file_name_to_date(f) for f in prediction_file_names
    ]

    regex_match_object = re.search(
        r'lead-time-days=(\d{2})', prediction_dir_name
    )
    assert regex_match_object
    model_lead_time_days = int(regex_match_object.group(1))

    print('Reading region mask from: "{0:s}"...'.format(region_mask_file_name))
    mask_table_xarray = region_mask_io.read_file(region_mask_file_name)
    mtx = mask_table_xarray

    num_dates = len(init_date_strings)
    statistic_value_by_date = numpy.full(num_dates, numpy.nan)
    model_file_name = None

    for i in range(num_dates):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        prediction_table_xarray = prediction_io.read_file(
            prediction_file_names[i]
        )
        ptx = prediction_table_xarray

        this_model_file_name = ptx.attrs[prediction_io.MODEL_FILE_KEY]
        if i == 0:
            model_file_name = copy.deepcopy(this_model_file_name)

        assert model_file_name == this_model_file_name

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

        field_idx = numpy.where(
            ptx[prediction_io.FIELD_NAME_KEY].values == target_field_name
        )[0][0]

        if quantity_name == extreme_cases_io.TARGET_QUANTITY_NAME:
            data_matrix = ptx[prediction_io.TARGET_KEY].values[..., field_idx]
        elif quantity_name == extreme_cases_io.PREDICTION_QUANTITY_NAME:
            data_matrix = numpy.mean(
                ptx[prediction_io.PREDICTION_KEY].values[..., field_idx, :],
                axis=-1
            )
        else:
            target_matrix = ptx[prediction_io.TARGET_KEY].values[..., field_idx]
            prediction_matrix = numpy.mean(
                ptx[prediction_io.PREDICTION_KEY].values[..., field_idx, :],
                axis=-1
            )
            data_matrix = prediction_matrix - target_matrix

        region_mask_matrix = mtx[region_mask_io.REGION_MASK_KEY].values
        data_matrix[region_mask_matrix == False] = numpy.nan

        if spatial_statistic_name == extreme_cases_io.SPATIAL_MIN_STAT_NAME:
            statistic_value_by_date[i] = numpy.nanmin(data_matrix)
        elif spatial_statistic_name == extreme_cases_io.SPATIAL_MAX_STAT_NAME:
            statistic_value_by_date[i] = numpy.nanmax(data_matrix)
        else:
            statistic_value_by_date[i] = numpy.nanmean(data_matrix)

    num_extreme_cases = min([num_extreme_cases, num_dates])
    half_num_extreme_cases = int(numpy.round(
        float(num_extreme_cases) / 2
    ))

    min_date_indices = numpy.argsort(statistic_value_by_date)[
        :half_num_extreme_cases
    ]
    max_date_indices = numpy.argsort(-1 * statistic_value_by_date)[
        :half_num_extreme_cases
    ]

    for i in range(len(min_date_indices)):
        print((
            '{0:d}th-lowest value of {1:s} for quantity {2:s} for field {3:s} '
            'for region {4:s} = {5:.3g} at init date {6:s}'
        ).format(
            i + 1,
            spatial_statistic_name.upper(),
            quantity_name.upper(),
            target_field_name.upper(),
            region_name,
            statistic_value_by_date[min_date_indices[i]],
            init_date_strings[min_date_indices[i]]
        ))

    print('\n')

    for i in range(len(max_date_indices)):
        print((
            '{0:d}th-highest value of {1:s} for quantity {2:s} for field {3:s} '
            'for region {4:s} = {5:.3g} at init date {6:s}'
        ).format(
            i + 1,
            spatial_statistic_name.upper(),
            quantity_name.upper(),
            target_field_name.upper(),
            region_name,
            statistic_value_by_date[max_date_indices[i]],
            init_date_strings[max_date_indices[i]]
        ))

    print('\n')

    min_values_file_name = output_file_names[0]
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=min_values_file_name
    )

    print('Writing dates with lowest values to: "{0:s}"...'.format(
        min_values_file_name
    ))
    extreme_cases_io.write_file(
        netcdf_file_name=min_values_file_name,
        init_date_strings=[init_date_strings[k] for k in min_date_indices],
        statistic_value_by_date=statistic_value_by_date[min_date_indices],
        spatial_statistic_name=spatial_statistic_name,
        quantity_name=quantity_name,
        target_field_name=target_field_name,
        region_mask_file_name=region_mask_file_name,
        region_name=region_name,
        prediction_file_names=prediction_file_names,
        model_file_name=model_file_name,
        model_lead_time_days=model_lead_time_days
    )

    max_values_file_name = output_file_names[1]
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=max_values_file_name
    )

    print('Writing dates with highest values to: "{0:s}"...'.format(
        max_values_file_name
    ))
    extreme_cases_io.write_file(
        netcdf_file_name=max_values_file_name,
        init_date_strings=[init_date_strings[k] for k in max_date_indices],
        statistic_value_by_date=statistic_value_by_date[max_date_indices],
        spatial_statistic_name=spatial_statistic_name,
        quantity_name=quantity_name,
        target_field_name=target_field_name,
        region_mask_file_name=region_mask_file_name,
        region_name=region_name,
        prediction_file_names=prediction_file_names,
        model_file_name=model_file_name,
        model_lead_time_days=model_lead_time_days
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        quantity_name=getattr(INPUT_ARG_OBJECT, QUANTITY_ARG_NAME),
        prediction_dir_name=getattr(INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME),
        init_date_limit_strings=getattr(
            INPUT_ARG_OBJECT, INIT_DATE_LIMITS_ARG_NAME
        ),
        num_extreme_cases=getattr(INPUT_ARG_OBJECT, NUM_EXTREME_CASES_ARG_NAME),
        spatial_statistic_name=getattr(INPUT_ARG_OBJECT, SPATIAL_STAT_ARG_NAME),
        region_mask_file_name=getattr(INPUT_ARG_OBJECT, REGION_FILE_ARG_NAME),
        region_name=getattr(INPUT_ARG_OBJECT, REGION_NAME_ARG_NAME),
        target_field_name=getattr(INPUT_ARG_OBJECT, TARGET_FIELD_ARG_NAME),
        output_file_names=getattr(INPUT_ARG_OBJECT, OUTPUT_FILES_ARG_NAME)
    )