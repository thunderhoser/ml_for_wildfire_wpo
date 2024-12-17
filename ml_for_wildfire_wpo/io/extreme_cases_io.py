"""Input/output methods for extreme cases."""

import numpy
import xarray
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.utils import canadian_fwi_utils

DATE_FORMAT = '%Y%m%d'
NUM_DATE_CHARS = 8

TIME_DIM = 'time'
TIME_CHAR_DIM = 'time_char'

INIT_DATE_KEY = 'init_date_string'
STATISTIC_VALUE_KEY = 'statistic_value'

SPATIAL_STATISTIC_KEY = 'spatial_statistic_name'
QUANTITY_KEY = 'quantity_name'
TARGET_FIELDS_KEY = 'target_field_names'
REGION_MASK_FILE_KEY = 'region_mask_file_name'
REGION_NAME_KEY = 'region_name'
PREDICTION_FILES_KEY = 'prediction_file_names'
MODEL_FILE_KEY = 'model_file_name'
MODEL_LEAD_TIME_KEY = 'model_lead_time_days'

TARGET_QUANTITY_NAME = 'target'
PREDICTION_QUANTITY_NAME = 'prediction'
MODEL_ERROR_QUANTITY_NAME = 'prediction_minus_target'
ABS_MODEL_ERROR_QUANTITY_NAME = 'absolute_prediction_minus_target'
VALID_QUANTITY_NAMES = [
    TARGET_QUANTITY_NAME, PREDICTION_QUANTITY_NAME, MODEL_ERROR_QUANTITY_NAME,
    ABS_MODEL_ERROR_QUANTITY_NAME
]

SPATIAL_MIN_STAT_NAME = 'spatial_min'
SPATIAL_MAX_STAT_NAME = 'spatial_max'
SPATIAL_MEAN_STAT_NAME = 'spatial_mean'
VALID_SPATIAL_STAT_NAMES = [
    SPATIAL_MIN_STAT_NAME, SPATIAL_MAX_STAT_NAME, SPATIAL_MEAN_STAT_NAME
]


def write_file(
        netcdf_file_name, init_date_strings, statistic_value_by_date,
        spatial_statistic_name, quantity_name, target_field_names,
        region_mask_file_name, region_name, prediction_file_names,
        model_file_name, model_lead_time_days):
    """Writes extreme cases to NetCDF file.

    E = number of extreme cases

    :param netcdf_file_name: Path to output file.
    :param init_date_strings: length-E list of forecast-init dates (format
        "yyyymmdd").
    :param statistic_value_by_date: length-E numpy array with statistic value
        for every forecast-init date.
    :param spatial_statistic_name: Metadata -- name of spatial statistic.  Must
        be in the list `VALID_SPATIAL_STAT_NAMES`.
    :param quantity_name: Metadata -- name of quantity.  Must be in the list
        `VALID_QUANTITY_NAMES`.
    :param target_field_names: Metadata -- 1-D list with names of target fields
        (each accepted by `canadian_fwi_utils.check_field_name`).  This
        indicates that the given spatial statistic, taken for the given
        quantity, was averaged over all fields in `target_field_names`.
    :param region_mask_file_name: Metadata -- path to file with region mask,
        used to determine which grid points are included in spatial statistic.
        This must be a file readable by `region_mask_io.read_file`.
    :param region_name: Metadata -- human-readable name of region included in
        spatial statistic.
    :param prediction_file_names: Metadata -- length-E list of paths to
        prediction files, from which raw values were taken to compute statistic.
    :param model_file_name: Metadata -- path to file with trained model used to
        generate predictions.
    :param model_lead_time_days: Metadata -- lead time for model predictions.
    """

    # Check input args.
    error_checking.assert_is_string_list(init_date_strings)
    for this_date_string in init_date_strings:
        _ = time_conversion.string_to_unix_sec(this_date_string, DATE_FORMAT)

    num_dates = len(init_date_strings)
    error_checking.assert_is_numpy_array(
        statistic_value_by_date,
        exact_dimensions=numpy.array([num_dates], dtype=int)
    )
    error_checking.assert_is_numpy_array_without_nan(statistic_value_by_date)

    error_checking.assert_is_string(spatial_statistic_name)
    assert spatial_statistic_name in VALID_SPATIAL_STAT_NAMES

    error_checking.assert_is_string(quantity_name)
    assert quantity_name in VALID_QUANTITY_NAMES

    error_checking.assert_is_string_list(target_field_names)
    for f in target_field_names:
        canadian_fwi_utils.check_field_name(f)

    error_checking.assert_is_string(region_mask_file_name)
    error_checking.assert_is_string(region_name)
    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_integer(model_lead_time_days)
    error_checking.assert_is_greater(model_lead_time_days, 0)

    # Do actual stuff.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF4'
    )

    dataset_object.setncattr(SPATIAL_STATISTIC_KEY, spatial_statistic_name)
    dataset_object.setncattr(QUANTITY_KEY, quantity_name)
    dataset_object.setncattr(TARGET_FIELDS_KEY, ' '.join(target_field_names))
    dataset_object.setncattr(REGION_MASK_FILE_KEY, region_mask_file_name)
    dataset_object.setncattr(REGION_NAME_KEY, region_name)
    dataset_object.setncattr(
        PREDICTION_FILES_KEY, ' '.join(prediction_file_names)
    )
    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(MODEL_LEAD_TIME_KEY, model_lead_time_days)

    dataset_object.createVariable(
        STATISTIC_VALUE_KEY, datatype=numpy.float64, dimensions=TIME_DIM
    )
    dataset_object.variables[STATISTIC_VALUE_KEY][:] = statistic_value_by_date

    this_string_format = 'S{0:d}'.format(NUM_DATE_CHARS)
    init_dates_char_array = netCDF4.stringtochar(numpy.array(
        init_date_strings, dtype=this_string_format
    ))

    dataset_object.createVariable(
        INIT_DATE_KEY, datatype='S1', dimensions=(TIME_DIM, TIME_CHAR_DIM)
    )
    dataset_object.variables[INIT_DATE_KEY][:] = numpy.array(
        init_dates_char_array
    )

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads extreme cases from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: extreme_case_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    extreme_case_table_xarray = xarray.open_dataset(netcdf_file_name)
    xct = extreme_case_table_xarray

    init_date_strings = [d.decode('utf-8') for d in xct[INIT_DATE_KEY].values]
    xct = xct.assign({
        INIT_DATE_KEY: (xct[INIT_DATE_KEY].dims, init_date_strings)
    })

    xct.attrs[PREDICTION_FILES_KEY] = xct.attrs[PREDICTION_FILES_KEY].split(' ')
    xct.attrs[TARGET_FIELDS_KEY] = xct.attrs[TARGET_FIELDS_KEY].split(' ')
    return xct
