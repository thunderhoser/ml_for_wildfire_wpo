"""Time zones."""

import os
import xarray
from gewittergefahr.gg_utils import error_checking

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))


def read_file(netcdf_file_name=None):
    """Reads time zones from file.

    :param netcdf_file_name: Path to input file.  To use the default path, leave
        this as None.
    :return: time_zone_table_xarray: xarray table with gridded time zones.
        Metadata and variable names in the xarray table should make it
        self-explanatory.
    """

    if netcdf_file_name is None:
        netcdf_file_name = '{0:s}/time_zone_map.nc'.format(THIS_DIRECTORY_NAME)

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)
