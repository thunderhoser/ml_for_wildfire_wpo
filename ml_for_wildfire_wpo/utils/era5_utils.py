"""Helper methods for ERA5 reanalysis."""

import numpy
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6

GRID_LATITUDES_DEG_N = numpy.linspace(17, 73, num=225, dtype=float)
GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E = numpy.linspace(
    -180, 179.75, num=1440, dtype=float
)
GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E = (
    lng_conversion.convert_lng_positive_in_west(
        GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E
    )
)

LONGITUDE_SPACING_DEG = 0.25

DEWPOINT_2METRE_NAME = 'dewpoint_2m_agl_kelvins'
RELATIVE_HUMIDITY_2METRE_NAME = 'relative_humidity_2m_agl'
TEMPERATURE_2METRE_NAME = 'temperature_2m_agl_kelvins'
SURFACE_PRESSURE_NAME = 'surface_pressure_pascals'
U_WIND_10METRE_NAME = 'u_wind_10m_agl_m_s01'
V_WIND_10METRE_NAME = 'v_wind_10m_agl_m_s01'
HOURLY_PRECIP_NAME = 'hourly_precip_metres'

ALL_FIELD_NAMES = [
    DEWPOINT_2METRE_NAME, RELATIVE_HUMIDITY_2METRE_NAME,
    TEMPERATURE_2METRE_NAME, SURFACE_PRESSURE_NAME,
    U_WIND_10METRE_NAME, V_WIND_10METRE_NAME, HOURLY_PRECIP_NAME
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


def desired_longitudes_to_columns(start_longitude_deg_e, end_longitude_deg_e):
    """Converts desired longitudes to desired grid columns.

    :param start_longitude_deg_e: Longitude at start of desired range.  This may
        be in either format (positive or negative values in western hemisphere).
    :param end_longitude_deg_e: Longitude at end of desired range.  This may
        be in either format.
    :return: desired_column_indices: 1-D numpy array with indices of desired
        columns.
    """

    start_longitude_deg_e = number_rounding.floor_to_nearest(
        start_longitude_deg_e, LONGITUDE_SPACING_DEG
    )
    end_longitude_deg_e = number_rounding.ceiling_to_nearest(
        end_longitude_deg_e, LONGITUDE_SPACING_DEG
    )
    error_checking.assert_is_greater(
        numpy.absolute(start_longitude_deg_e - end_longitude_deg_e),
        TOLERANCE
    )

    start_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        start_longitude_deg_e, allow_nan=False
    )
    end_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        end_longitude_deg_e, allow_nan=False
    )

    if end_longitude_deg_e > start_longitude_deg_e:
        are_longitudes_positive_in_west = True
    else:
        start_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            start_longitude_deg_e, allow_nan=False
        )
        end_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            end_longitude_deg_e, allow_nan=False
        )
        are_longitudes_positive_in_west = False

    if are_longitudes_positive_in_west:
        grid_longitudes_deg_e = GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    else:
        grid_longitudes_deg_e = GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E

    num_longitudes = 1 + int(numpy.absolute(numpy.round(
        (end_longitude_deg_e - start_longitude_deg_e) / LONGITUDE_SPACING_DEG
    )))

    if end_longitude_deg_e > start_longitude_deg_e:
        desired_longitudes_deg_e = numpy.linspace(
            start_longitude_deg_e, end_longitude_deg_e, num=num_longitudes
        )

        desired_column_indices = numpy.array([
            numpy.where(
                numpy.absolute(grid_longitudes_deg_e - d) <= TOLERANCE
            )[0][0]
            for d in desired_longitudes_deg_e
        ], dtype=int)
    else:
        undesired_longitudes_deg_e = numpy.linspace(
            end_longitude_deg_e, start_longitude_deg_e, num=num_longitudes
        )[1:-1]

        undesired_column_indices = numpy.array([
            numpy.where(
                numpy.absolute(grid_longitudes_deg_e - d) <= TOLERANCE
            )[0][0]
            for d in undesired_longitudes_deg_e
        ], dtype=int)

        all_column_indices = numpy.linspace(
            0, len(grid_longitudes_deg_e) - 1, num=len(grid_longitudes_deg_e),
            dtype=int
        )
        desired_column_indices = (
            set(all_column_indices.tolist()) -
            set(undesired_column_indices.tolist())
        )
        desired_column_indices = numpy.array(
            list(desired_column_indices), dtype=int
        )

        break_index = 1 + numpy.where(
            numpy.diff(desired_column_indices) > 1
        )[0][0]

        desired_column_indices = numpy.concatenate((
            desired_column_indices[break_index:],
            desired_column_indices[:break_index]
        ))

    return desired_column_indices

    # sort_indices = numpy.argsort(grid_longitudes_deg_e[desired_column_indices])
    # return desired_column_indices[sort_indices]
