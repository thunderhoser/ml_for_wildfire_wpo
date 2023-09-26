"""Input/output methods for Canadian fire-weather indices."""

import os
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.utils import canadian_fwi_utils

DATE_FORMAT = '%Y%m%d'


def find_file(directory_name, valid_date_string, raise_error_if_missing=True):
    """Finds NetCDF file with Canadian FWIs for one day.

    :param directory_name: Path to input directory.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: fwi_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)
    error_checking.assert_is_boolean(raise_error_if_missing)

    fwi_file_name = '{0:s}/canadian_fwi_{1:s}.nc'.format(
        directory_name, valid_date_string
    )

    if raise_error_if_missing and not os.path.isfile(fwi_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            fwi_file_name
        )
        raise ValueError(error_string)

    return fwi_file_name


def file_name_to_date(fwi_file_name):
    """Parses date from name of FWI file.

    :param fwi_file_name: File path.
    :return: valid_date_string: Valid date (format "yyyymmdd").
    """

    pathless_file_name = os.path.split(fwi_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    valid_date_string = extensionless_file_name.split('_')[1]

    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    return valid_date_string


def write_file(netcdf_file_name, data_matrix, latitudes_deg_n, longitudes_deg_e,
               field_names):
    """Writes processed FWI data to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid
    F = number of fields

    :param netcdf_file_name: Path to output file.
    :param data_matrix: M-by-N-by-F numpy array of data values.
    :param latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param field_names: length-F list of field names.
    """

    # Check input args.
    error_checking.assert_is_valid_lat_numpy_array(
        latitudes_deg_n, allow_nan=False
    )
    error_checking.assert_is_numpy_array(latitudes_deg_n, num_dimensions=1)

    error_checking.assert_is_valid_lng_numpy_array(
        longitudes_deg_e, allow_nan=False
    )
    error_checking.assert_is_numpy_array(longitudes_deg_e, num_dimensions=1)

    error_checking.assert_is_string_list(field_names)
    for this_field_name in field_names:
        canadian_fwi_utils.check_field_name(this_field_name)

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    # Do actual stuff.
    coord_dict = {
        canadian_fwi_utils.LATITUDE_DIM: latitudes_deg_n,
        canadian_fwi_utils.LONGITUDE_DIM: longitudes_deg_e,
        canadian_fwi_utils.FIELD_DIM: field_names
    }

    these_dim = (
        canadian_fwi_utils.LATITUDE_DIM,
        canadian_fwi_utils.LONGITUDE_DIM,
        canadian_fwi_utils.FIELD_DIM
    )
    main_data_dict = {
        canadian_fwi_utils.DATA_KEY: (these_dim, data_matrix)
    }
    fwi_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=coord_dict
    )

    fwi_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_file(netcdf_file_name):
    """Reads processed FWI data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: fwi_table_xarray: xarray table (metadata and variable names should
        make the table self-explanatory).
    """

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)


def write_normalization_file(norm_param_table_xarray, netcdf_file_name):
    """Writes normalization parameters for FWI data to NetCDF file.

    :param norm_param_table_xarray: xarray table (metadata and variable names
        should make the table self-explanatory).
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    norm_param_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_normalization_file(netcdf_file_name):
    """Reads normalization parameters for FWI data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: norm_param_table_xarray: xarray table (metadata and variable names
        should make the table self-explanatory).
    """

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)
