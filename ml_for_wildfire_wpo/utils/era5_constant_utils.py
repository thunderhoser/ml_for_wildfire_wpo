"""Helper methods for constants in ERA5 reanalysis."""

from gewittergefahr.gg_utils import error_checking

LATITUDE_DIM = 'latitude_deg_n'
LONGITUDE_DIM = 'longitude_deg_e'
FIELD_DIM = 'field'
DATA_KEY = 'data'

SUBGRID_OROGRAPHY_SINE_NAME = 'subgrid_orography_angle_sine'
SUBGRID_OROGRAPHY_COSINE_NAME = 'subgrid_orography_angle_cosine'
SUBGRID_OROGRAPHY_ANISOTROPY_NAME = 'subgrid_orography_angle_anisotropy'
GEOPOTENTIAL_NAME = 'geopotential_m2_s02'
LAND_SEA_MASK_NAME = 'land_sea_mask_boolean'
SUBGRID_OROGRAPHY_SLOPE_NAME = 'subgrid_orography_angle_slope'
SUBGRID_OROGRAPHY_STDEV_NAME = 'filtered_subgrid_orography_stdev_metres'
RESOLVED_OROGRAPHY_STDEV_NAME = 'resolved_orography_stdev_metres'

ALL_FIELD_NAMES = [
    SUBGRID_OROGRAPHY_SINE_NAME, SUBGRID_OROGRAPHY_COSINE_NAME,
    SUBGRID_OROGRAPHY_ANISOTROPY_NAME, GEOPOTENTIAL_NAME,
    LAND_SEA_MASK_NAME, SUBGRID_OROGRAPHY_SLOPE_NAME,
    SUBGRID_OROGRAPHY_STDEV_NAME, RESOLVED_OROGRAPHY_STDEV_NAME
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
