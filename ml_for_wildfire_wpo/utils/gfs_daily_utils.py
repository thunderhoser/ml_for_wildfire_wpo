"""Helper methods for daily GFS data."""

from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.utils import gfs_utils

RELATIVE_HUMIDITY_2METRE_NAME = 'relative_humidity_2m_agl'
TEMPERATURE_2METRE_NAME = gfs_utils.TEMPERATURE_2METRE_NAME
DEWPOINT_2METRE_NAME = gfs_utils.DEWPOINT_2METRE_NAME
U_WIND_10METRE_NAME = gfs_utils.U_WIND_10METRE_NAME
V_WIND_10METRE_NAME = gfs_utils.V_WIND_10METRE_NAME
SURFACE_PRESSURE_NAME = gfs_utils.SURFACE_PRESSURE_NAME
PRECIP_NAME = gfs_utils.PRECIP_NAME

ALL_FIELD_NAMES = [
    RELATIVE_HUMIDITY_2METRE_NAME, TEMPERATURE_2METRE_NAME,
    DEWPOINT_2METRE_NAME, U_WIND_10METRE_NAME, V_WIND_10METRE_NAME,
    SURFACE_PRESSURE_NAME, PRECIP_NAME
]


def check_field_name(field_name):
    """Ensures that field name is valid.

    :param field_name: String (must be in list `ALL_FIELD_NAMES`).
    :raises: ValueError: if `field_name not in ALL_FIELD_NAMES`.
    """

    error_checking.assert_is_string(field_name)
    if field_name in ALL_FIELD_NAMES:
        return

    error_string = (
        'Field name "{0:s}" is not in the list of accepted field names '
        '(below):\n{1:s}'
    ).format(
        field_name, ALL_FIELD_NAMES
    )

    raise ValueError(error_string)
