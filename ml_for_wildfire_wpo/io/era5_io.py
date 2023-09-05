"""Input/output methods for processed ERA5 reanalysis data.

Processed ERA5 data are stored with NetCDF file per day, containing fire-weather
inputs valid at 1200 local standard time everywhere in the grid.
"""

import os
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.utils import era5_utils

DATE_FORMAT = '%Y%m%d'

LATITUDE_DIM = 'latitude_deg_n'
LONGITUDE_DIM = 'longitude_deg_e'
FIELD_DIM = 'field_name'
DATA_KEY = 'data'


def find_file(directory_name, valid_date_string, raise_error_if_missing=True):
    """Finds NetCDF file with ERA5 fire-weather inputs for one day.

    :param directory_name: Path to input directory.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: era5_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)
    error_checking.assert_is_boolean(raise_error_if_missing)

    era5_file_name = '{0:s}/era5_{1:s}.nc'.format(
        directory_name, valid_date_string
    )

    if raise_error_if_missing and not os.path.isfile(era5_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            era5_file_name
        )
        raise ValueError(error_string)

    return era5_file_name


def file_name_to_date(era5_file_name):
    """Parses date from name of ERA5 file.

    :param era5_file_name: File path.
    :return: valid_date_string: Valid date (format "yyyymmdd").
    """

    pathless_file_name = os.path.split(era5_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    valid_date_string = extensionless_file_name.split('_')[1]

    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    return valid_date_string


def write_file(netcdf_file_name, data_matrix, latitudes_deg_n, longitudes_deg_e,
               field_names):
    """Writes processed ERA5 data to NetCDF file.

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
        era5_utils.check_field_name(this_field_name)

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    # Do actual stuff.
    coord_dict = {
        LATITUDE_DIM: latitudes_deg_n,
        LONGITUDE_DIM: longitudes_deg_e,
        FIELD_DIM: field_names
    }
    main_data_dict = {
        DATA_KEY: (
            (LATITUDE_DIM, LONGITUDE_DIM, FIELD_DIM), data_matrix
        )
    }
    era5_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=coord_dict
    )

    era5_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_file(netcdf_file_name):
    """Reads processed ERA5 data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: era5_table_xarray: xarray table (metadata and variable names should
        make the table self-explanatory).
    """

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)
