"""Input/output methods for processed constants from ERA5 reanalysis."""

import xarray
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking


def read_file(netcdf_file_name):
    """Reads processed ERA5 constants from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: era5_constant_table_xarray: xarray table (metadata and variable
        names should make the table self-explanatory).
    """

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)


def write_file(netcdf_file_name, era5_constant_table_xarray):
    """Writes processed ERA5 constants to NetCDF file.

    :param netcdf_file_name: Path to output file.
    :param era5_constant_table_xarray: xarray table (metadata and variable
        names should make the table self-explanatory).
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    era5_constant_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def write_normalization_file(norm_param_table_xarray, netcdf_file_name):
    """Writes normalization parameters for ERA5 constants to NetCDF file.

    :param norm_param_table_xarray: xarray table (metadata and variable names
        should make the table self-explanatory).
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    norm_param_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_normalization_file(netcdf_file_name):
    """Reads normalization parameters for ERA5 constants from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: norm_param_table_xarray: xarray table (metadata and variable names
        should make the table self-explanatory).
    """

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)
