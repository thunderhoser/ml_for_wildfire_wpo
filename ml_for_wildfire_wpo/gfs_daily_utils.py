"""Helper methods for daily GFS data."""

import os
import sys

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gfs_utils

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
