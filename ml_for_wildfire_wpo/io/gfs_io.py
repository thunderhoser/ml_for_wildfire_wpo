"""Input/output methods for processed GFS model data.

Processed GFS data are stored with one zarr file per day, containing forecasts
init at 00Z and valid at many forecast hours (lead times).
"""

import os
import shutil
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.utils import gfs_utils

DATE_FORMAT = '%Y%m%d'


def find_file(directory_name, init_date_string, raise_error_if_missing=True):
    """Finds zarr file with GFS forecasts for one day.

    :param directory_name: Path to input directory.
    :param init_date_string: Initialization date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: gfs_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    _ = time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)
    error_checking.assert_is_boolean(raise_error_if_missing)

    gfs_file_name = '{0:s}/gfs_{1:s}.zarr'.format(
        directory_name, init_date_string
    )

    if raise_error_if_missing and not os.path.isdir(gfs_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            gfs_file_name
        )
        raise ValueError(error_string)

    return gfs_file_name


def file_name_to_date(gfs_file_name):
    """Parses date from name of ERA5 file.

    :param gfs_file_name: File path.
    :return: init_date_string: Initialization date (format "yyyymmdd").
    """

    pathless_file_name = os.path.split(gfs_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    init_date_string = extensionless_file_name.split('_')[1]

    _ = time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)

    return init_date_string


def read_file(zarr_file_name):
    """Reads GFS data from zarr file.

    :param zarr_file_name: Path to input file.
    :return: gfs_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    return xarray.open_zarr(zarr_file_name)


def write_file(gfs_table_xarray, zarr_file_name):
    """Writes GFS data to zarr file.

    :param gfs_table_xarray: xarray table in format returned by `read_file`.
    :param zarr_file_name: Path to output file.
    """

    error_checking.assert_is_string(zarr_file_name)
    if os.path.isdir(zarr_file_name):
        shutil.rmtree(zarr_file_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=zarr_file_name
    )

    encoding_dict = {
        gfs_utils.DATA_KEY_2D: {'dtype': 'float32'},
        gfs_utils.DATA_KEY_3D: {'dtype': 'float32'}
    }
    gfs_table_xarray.to_zarr(
        store=zarr_file_name, mode='w', encoding=encoding_dict
    )
