"""Input/output methods for daily GFS model data.

Daily data are used to compute raw-GFS forecasts of fire-weather indices.
"""

import os
import shutil
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.utils import gfs_utils

DATE_FORMAT = '%Y%m%d'

LATITUDE_DIM = gfs_utils.LATITUDE_DIM
LONGITUDE_DIM = gfs_utils.LONGITUDE_DIM
FIELD_DIM = gfs_utils.FIELD_DIM_2D
LEAD_TIME_DIM = 'lead_time_days'

DATA_KEY_2D = gfs_utils.DATA_KEY_2D


def find_file(directory_name, init_date_string, raise_error_if_missing=True):
    """Finds zarr file with daily GFS data for one model run.

    :param directory_name: Path to input directory.
    :param init_date_string: Initialization date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: gfs_daily_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    _ = time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)
    error_checking.assert_is_boolean(raise_error_if_missing)

    gfs_daily_file_name = '{0:s}/gfs_daily_{1:s}.zarr'.format(
        directory_name, init_date_string
    )

    if raise_error_if_missing and not os.path.isdir(gfs_daily_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            gfs_daily_file_name
        )
        raise ValueError(error_string)

    return gfs_daily_file_name


def read_file(zarr_file_name):
    """Reads daily GFS data from zarr file.

    :param zarr_file_name: Path to input file.
    :return: gfs_daily_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    return xarray.open_zarr(zarr_file_name)


def write_file(
        zarr_file_name, data_matrix, latitudes_deg_n, longitudes_deg_e,
        field_names, lead_times_days):
    """Writes daily GFS data to zarr file.

    L = number of lead times
    M = number of rows in grid
    N = number of columns in grid
    F = number of fields

    :param zarr_file_name: Path to output file.
    :param data_matrix: L-by-M-by-N-by-F numpy array of data values.
    :param latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param field_names: length-F list of field names.
    :param lead_times_days: length-L numpy array of lead times.
    """

    # Check input args.
    error_checking.assert_is_string(zarr_file_name)
    if os.path.isdir(zarr_file_name):
        shutil.rmtree(zarr_file_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=zarr_file_name
    )

    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg_n)
    error_checking.assert_is_numpy_array(latitudes_deg_n, num_dimensions=1)

    error_checking.assert_is_valid_lng_numpy_array(
        longitudes_deg_e,
        positive_in_west_flag=False, negative_in_west_flag=False
    )
    error_checking.assert_is_numpy_array(longitudes_deg_e, num_dimensions=1)

    error_checking.assert_is_string_list(field_names)

    error_checking.assert_is_numpy_array(lead_times_days, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(lead_times_days)
    error_checking.assert_is_geq_numpy_array(lead_times_days, 1)
    error_checking.assert_is_greater_numpy_array(numpy.diff(lead_times_days), 0)

    expected_dim = numpy.array([
        len(lead_times_days), len(latitudes_deg_n), len(longitudes_deg_e),
        len(field_names)
    ], dtype=int)

    # TODO(thunderhoser): Eh, I don't know about this.
    error_checking.assert_is_numpy_array_without_nan(data_matrix)
    error_checking.assert_is_numpy_array(
        data_matrix, exact_dimensions=expected_dim
    )

    # Do actual stuff.
    coord_dict = {
        LEAD_TIME_DIM: lead_times_days,
        LATITUDE_DIM: latitudes_deg_n,
        LONGITUDE_DIM: longitudes_deg_e,
        FIELD_DIM: field_names
    }

    these_dim = (LEAD_TIME_DIM, LATITUDE_DIM, LONGITUDE_DIM, FIELD_DIM)
    main_data_dict = {
        DATA_KEY_2D: (these_dim, data_matrix)
    }

    gfs_daily_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=coord_dict
    )

    encoding_dict = {
        gfs_utils.DATA_KEY_2D: {'dtype': 'float32'}
    }
    gfs_daily_table_xarray.to_zarr(
        store=zarr_file_name, mode='w', encoding=encoding_dict
    )
