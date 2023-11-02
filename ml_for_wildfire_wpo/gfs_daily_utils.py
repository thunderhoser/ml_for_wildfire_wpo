"""Helper methods for daily GFS data."""

import os
import sys

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import gfs_utils
import canadian_fwi_utils

RELATIVE_HUMIDITY_2METRE_NAME = 'relative_humidity_2m_agl'
TEMPERATURE_2METRE_NAME = gfs_utils.TEMPERATURE_2METRE_NAME
DEWPOINT_2METRE_NAME = gfs_utils.DEWPOINT_2METRE_NAME
U_WIND_10METRE_NAME = gfs_utils.U_WIND_10METRE_NAME
V_WIND_10METRE_NAME = gfs_utils.V_WIND_10METRE_NAME
SURFACE_PRESSURE_NAME = gfs_utils.SURFACE_PRESSURE_NAME
PRECIP_NAME = gfs_utils.PRECIP_NAME

ALL_NON_FWI_FIELD_NAMES = [
    RELATIVE_HUMIDITY_2METRE_NAME, TEMPERATURE_2METRE_NAME,
    DEWPOINT_2METRE_NAME, U_WIND_10METRE_NAME, V_WIND_10METRE_NAME,
    SURFACE_PRESSURE_NAME, PRECIP_NAME
]

FFMC_NAME = canadian_fwi_utils.FFMC_NAME
DMC_NAME = canadian_fwi_utils.DMC_NAME
DC_NAME = canadian_fwi_utils.DC_NAME
ISI_NAME = canadian_fwi_utils.ISI_NAME
BUI_NAME = canadian_fwi_utils.BUI_NAME
FWI_NAME = canadian_fwi_utils.FWI_NAME
DSR_NAME = canadian_fwi_utils.DSR_NAME

ALL_FWI_FIELD_NAMES = [
    FFMC_NAME, DMC_NAME, DC_NAME, ISI_NAME, BUI_NAME, FWI_NAME, DSR_NAME
]
ALL_FIELD_NAMES = ALL_NON_FWI_FIELD_NAMES + ALL_FWI_FIELD_NAMES


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
        field_name, str(ALL_FIELD_NAMES)
    )

    raise ValueError(error_string)
