"""Helper methods for Canadian fire-weather indices."""

from gewittergefahr.gg_utils import error_checking

LATITUDE_DIM = 'latitude_deg_n'
LONGITUDE_DIM = 'longitude_deg_e'
FIELD_DIM = 'field_name'
DATA_KEY = 'data'

FFMC_NAME = 'fine_fuel_moisture_code'
DMC_NAME = 'duff_moisture_code'
DC_NAME = 'drought_code'
ISI_NAME = 'initial_spread_index'
BUI_NAME = 'buildup_index'
FWI_NAME = 'fire_weather_index'
DSR_NAME = 'daily_severity_rating'

ALL_FIELD_NAMES = [
    FFMC_NAME, DMC_NAME, DC_NAME, ISI_NAME, BUI_NAME, FWI_NAME, DSR_NAME
]


def check_field_name(field_name):
    """Ensures validity of field name.

    :param field_name: Name of weather field.
    :raises: ValueError: if `field_name not in ALL_FIELD_NAMES`.
    """

    error_checking.assert_is_string(field_name)
    if field_name in ALL_FIELD_NAMES:
        return

    error_string = (
        'Field "{0:s}" is not in the list of recognized fields (below):\n{1:s}'
    ).format(
        field_name, str(ALL_FIELD_NAMES)
    )

    raise ValueError(error_string)
