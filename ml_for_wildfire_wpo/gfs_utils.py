"""Helper methods for GFS model."""

import os
import sys
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking

FORECAST_HOUR_DIM = 'forecast_hour'
LATITUDE_DIM = 'latitude_deg_n'
LONGITUDE_DIM = 'longitude_deg_e'
PRESSURE_LEVEL_DIM = 'pressure_level_mb'
FIELD_DIM_2D = 'field_name_2d'
FIELD_DIM_3D = 'field_name_3d'

DATA_KEY_2D = 'data_matrix_2d'
DATA_KEY_3D = 'data_matrix_3d'

TEMPERATURE_NAME = 'temperature_kelvins'
SPECIFIC_HUMIDITY_NAME = 'specific_humidity_kg_kg01'
GEOPOTENTIAL_HEIGHT_NAME = 'geopotential_height_m_asl'
U_WIND_NAME = 'u_wind_m_s01'
V_WIND_NAME = 'v_wind_m_s01'

TEMPERATURE_2METRE_NAME = 'temperature_2m_agl_kelvins'
DEWPOINT_2METRE_NAME = 'dewpoint_2m_agl_kelvins'
SPECIFIC_HUMIDITY_2METRE_NAME = 'specific_humidity_2m_agl_kg_kg01'
U_WIND_10METRE_NAME = 'u_wind_10m_agl_m_s01'
V_WIND_10METRE_NAME = 'v_wind_10m_agl_m_s01'
SURFACE_PRESSURE_NAME = 'surface_pressure_pascals'

PRECIP_NAME = 'accumulated_precip_metres'
CONVECTIVE_PRECIP_NAME = 'accumulated_convective_precip_metres'

SOIL_TEMPERATURE_0TO10CM_NAME = 'soil_temperature_0to10cm_kelvins'
SOIL_TEMPERATURE_10TO40CM_NAME = 'soil_temperature_10to40cm_kelvins'
SOIL_TEMPERATURE_40TO100CM_NAME = 'soil_temperature_40to100cm_kelvins'
SOIL_TEMPERATURE_100TO200CM_NAME = 'soil_temperature_100to200cm_kelvins'
SOIL_MOISTURE_0TO10CM_NAME = 'volumetric_soil_moisture_fraction_0to10cm'
SOIL_MOISTURE_10TO40CM_NAME = 'volumetric_soil_moisture_fraction_10to40cm'
SOIL_MOISTURE_40TO100CM_NAME = 'volumetric_soil_moisture_fraction_40to100cm'
SOIL_MOISTURE_100TO200CM_NAME = 'volumetric_soil_moisture_fraction_100to200cm'

PRECIPITABLE_WATER_NAME = 'precipitable_water_metres'
SNOW_DEPTH_NAME = 'snow_depth_metres'
WATER_EQUIV_SNOW_DEPTH_NAME = 'water_equivalent_snow_depth_metres'
CAPE_FULL_COLUMN_NAME = 'conv_avail_pot_energy_full_column_j_kg01'
CAPE_BOTTOM_180MB_NAME = 'conv_avail_pot_energy_bottom_180mb_j_kg01'
CAPE_BOTTOM_255MB_NAME = 'conv_avail_pot_energy_bottom_255mb_j_kg01'

DOWNWARD_SURFACE_SHORTWAVE_FLUX_NAME = 'downward_surface_shortwave_flux_w_m02'
DOWNWARD_SURFACE_LONGWAVE_FLUX_NAME = 'downward_surface_longwave_flux_w_m02'
UPWARD_SURFACE_SHORTWAVE_FLUX_NAME = 'upward_surface_shortwave_flux_w_m02'
UPWARD_SURFACE_LONGWAVE_FLUX_NAME = 'upward_surface_longwave_flux_w_m02'
UPWARD_TOA_SHORTWAVE_FLUX_NAME = 'upward_toa_shortwave_flux_w_m02'
UPWARD_TOA_LONGWAVE_FLUX_NAME = 'upward_toa_longwave_flux_w_m02'

VEGETATION_FRACTION_NAME = 'vegetation_fraction'
SOIL_TYPE_NAME = 'soil_type'

ALL_3D_FIELD_NAMES = [
    TEMPERATURE_NAME, SPECIFIC_HUMIDITY_NAME, GEOPOTENTIAL_HEIGHT_NAME,
    U_WIND_NAME, V_WIND_NAME
]

ALL_2D_FIELD_NAMES = [
    TEMPERATURE_2METRE_NAME, DEWPOINT_2METRE_NAME,
    SPECIFIC_HUMIDITY_2METRE_NAME, U_WIND_10METRE_NAME, V_WIND_10METRE_NAME,
    SURFACE_PRESSURE_NAME,
    PRECIP_NAME, CONVECTIVE_PRECIP_NAME,
    SOIL_TEMPERATURE_0TO10CM_NAME, SOIL_TEMPERATURE_10TO40CM_NAME,
    SOIL_TEMPERATURE_40TO100CM_NAME, SOIL_TEMPERATURE_100TO200CM_NAME,
    SOIL_MOISTURE_0TO10CM_NAME, SOIL_MOISTURE_10TO40CM_NAME,
    SOIL_MOISTURE_40TO100CM_NAME, SOIL_MOISTURE_100TO200CM_NAME,
    PRECIPITABLE_WATER_NAME, SNOW_DEPTH_NAME, WATER_EQUIV_SNOW_DEPTH_NAME,
    CAPE_FULL_COLUMN_NAME, CAPE_BOTTOM_180MB_NAME, CAPE_BOTTOM_255MB_NAME,
    DOWNWARD_SURFACE_SHORTWAVE_FLUX_NAME,
    DOWNWARD_SURFACE_LONGWAVE_FLUX_NAME,
    UPWARD_SURFACE_SHORTWAVE_FLUX_NAME, UPWARD_SURFACE_LONGWAVE_FLUX_NAME,
    UPWARD_TOA_SHORTWAVE_FLUX_NAME, UPWARD_TOA_LONGWAVE_FLUX_NAME,
    VEGETATION_FRACTION_NAME, SOIL_TYPE_NAME
]

ALL_FIELD_NAMES = ALL_3D_FIELD_NAMES + ALL_2D_FIELD_NAMES

NO_0HOUR_ANALYSIS_FIELD_NAMES = [
    PRECIP_NAME, CONVECTIVE_PRECIP_NAME,
    DOWNWARD_SURFACE_SHORTWAVE_FLUX_NAME,
    DOWNWARD_SURFACE_LONGWAVE_FLUX_NAME,
    UPWARD_SURFACE_SHORTWAVE_FLUX_NAME, UPWARD_SURFACE_LONGWAVE_FLUX_NAME,
    UPWARD_TOA_SHORTWAVE_FLUX_NAME, UPWARD_TOA_LONGWAVE_FLUX_NAME
]

MAYBE_MISSING_FIELD_NAMES = [
    PRECIP_NAME, CONVECTIVE_PRECIP_NAME,
    VEGETATION_FRACTION_NAME, SOIL_TYPE_NAME
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


def concat_over_forecast_hours(gfs_tables_xarray):
    """Concatenates GFS tables over forecast hour.

    :param gfs_tables_xarray: 1-D list of xarray tables with GFS data.
    :return: gfs_table_xarray: Single xarray table with GFS data.
    """

    return xarray.concat(
        gfs_tables_xarray, dim=FORECAST_HOUR_DIM, data_vars='all',
        coords='minimal', compat='identical', join='exact'
    )


def subset_by_row(gfs_table_xarray, desired_row_indices):
    """Subsets GFS table by grid row.

    :param gfs_table_xarray: xarray table with GFS forecasts.
    :param desired_row_indices: 1-D numpy array with indices of desired rows.
    :return: gfs_table_xarray: Same as input but maybe with fewer rows.
    """

    error_checking.assert_is_numpy_array(desired_row_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_row_indices)
    error_checking.assert_is_geq_numpy_array(desired_row_indices, 0)

    return gfs_table_xarray.isel({LATITUDE_DIM: desired_row_indices})


def subset_by_column(gfs_table_xarray, desired_column_indices):
    """Subsets GFS table by grid column.

    :param gfs_table_xarray: xarray table with GFS forecasts.
    :param desired_column_indices: 1-D numpy array with indices of desired
        columns.
    :return: gfs_table_xarray: Same as input but maybe with fewer columns.
    """

    error_checking.assert_is_numpy_array(
        desired_column_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)

    return gfs_table_xarray.isel({LONGITUDE_DIM: desired_column_indices})


def subset_by_forecast_hour(gfs_table_xarray, desired_forecast_hours):
    """Subsets GFS table by forecast hour.

    :param gfs_table_xarray: xarray table with GFS forecasts.
    :param desired_forecast_hours: 1-D numpy array of desired forecast hours.
    :return: gfs_table_xarray: Same as input but maybe with fewer forecast
        hours.
    """

    error_checking.assert_is_numpy_array(
        desired_forecast_hours, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_forecast_hours)
    error_checking.assert_is_geq_numpy_array(desired_forecast_hours, 0)

    return gfs_table_xarray.sel({FORECAST_HOUR_DIM: desired_forecast_hours})


def get_field(gfs_table_xarray, field_name, pressure_levels_mb=None):
    """Extracts one field from GFS table.

    M = number of rows in grid
    N = number of columns in grid
    P = number of pressure levels

    :param gfs_table_xarray: xarray table with GFS forecasts.
    :param field_name: Field name.
    :param pressure_levels_mb: length-P numpy array of pressure levels.  If the
        field is 2-D rather than 3-D, leave this argument alone.
    :return: data_matrix: M-by-N (if field is 2-D) or M-by-N-by-P (if field is
        3-D) numpy array of data values.
    """

    check_field_name(field_name)

    if field_name in ALL_2D_FIELD_NAMES:
        k = numpy.where(
            gfs_table_xarray.coords[FIELD_DIM_2D].values == field_name
        )[0][0]
        return gfs_table_xarray[DATA_KEY_2D].values[..., k]

    k = numpy.where(
        gfs_table_xarray.coords[FIELD_DIM_3D].values == field_name
    )[0][0]

    js = numpy.array([
        numpy.where(
            gfs_table_xarray.coords[PRESSURE_LEVEL_DIM].values == p
        )[0][0]
        for p in pressure_levels_mb
    ], dtype=int)

    return gfs_table_xarray[DATA_KEY_3D].values[..., k][..., js]
