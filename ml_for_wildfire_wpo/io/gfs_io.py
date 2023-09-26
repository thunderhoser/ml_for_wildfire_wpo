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


def find_files_for_period(
        directory_name, first_init_date_string, last_init_date_string,
        raise_error_if_any_missing=False, raise_error_if_all_missing=True):
    """Finds files with GFS fcsts over a time period, one per daily model run.

    :param directory_name: Path to input directory.
    :param first_init_date_string: First date in period (format "yyyymmdd").
    :param last_init_date_string: Last date in period (format "yyyymmdd").
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :return: zarr_file_names: 1-D list of paths to zarr files with GFS
        forecasts, one per daily model run.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    init_date_strings = time_conversion.get_spc_dates_in_range(
        first_init_date_string, last_init_date_string
    )

    zarr_file_names = []

    for this_date_string in init_date_strings:
        this_file_name = find_file(
            directory_name=directory_name,
            init_date_string=this_date_string,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if os.path.isdir(this_file_name):
            zarr_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(zarr_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from dates {1:s} to '
            '{2:s}.'
        ).format(
            directory_name, first_init_date_string, last_init_date_string
        )
        raise ValueError(error_string)

    return zarr_file_names


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
        gfs_utils.DATA_KEY_2D: {'dtype': 'float32'}
    }
    if gfs_utils.DATA_KEY_3D in gfs_table_xarray.data_vars:
        encoding_dict[gfs_utils.DATA_KEY_3D] = {'dtype': 'float32'}

    gfs_table_xarray.to_zarr(
        store=zarr_file_name, mode='w', encoding=encoding_dict
    )


def write_normalization_file(norm_param_table_xarray, netcdf_file_name):
    """Writes normalization parameters for GFS data to NetCDF file.

    :param norm_param_table_xarray: xarray table (metadata and variable names
        should make the table self-explanatory).
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    norm_param_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_normalization_file(netcdf_file_name):
    """Reads normalization parameters for GFS data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: norm_param_table_xarray: xarray table (metadata and variable names
        should make the table self-explanatory).
    """

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)
