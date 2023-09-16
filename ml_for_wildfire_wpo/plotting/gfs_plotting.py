"""Methods for plotting GFS model output."""

from gewittergefahr.gg_utils import temperature_conversions as temperature_conv
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.utils import gfs_utils
from ml_for_wildfire_wpo.plotting import fwi_plotting

FIELD_TO_CONV_FACTOR = {
    gfs_utils.SPECIFIC_HUMIDITY_NAME: 1000.,
    gfs_utils.GEOPOTENTIAL_HEIGHT_NAME: 0.1,
    gfs_utils.U_WIND_NAME: 1.,
    gfs_utils.V_WIND_NAME: 1.,
    gfs_utils.SPECIFIC_HUMIDITY_2METRE_NAME: 1000.,
    gfs_utils.U_WIND_10METRE_NAME: 1.,
    gfs_utils.V_WIND_10METRE_NAME: 1.,
    gfs_utils.SURFACE_PRESSURE_NAME: 0.01,
    gfs_utils.PRECIP_NAME: 1000.,
    gfs_utils.CONVECTIVE_PRECIP_NAME: 1000.,
    gfs_utils.SOIL_MOISTURE_0TO10CM_NAME: 1.,
    gfs_utils.SOIL_MOISTURE_10TO40CM_NAME: 1.,
    gfs_utils.SOIL_MOISTURE_40TO100CM_NAME: 1.,
    gfs_utils.SOIL_MOISTURE_100TO200CM_NAME: 1.,
    gfs_utils.PRECIPITABLE_WATER_NAME: 1000.,
    gfs_utils.SNOW_DEPTH_NAME: 100.,
    gfs_utils.WATER_EQUIV_SNOW_DEPTH_NAME: 1000.,
    gfs_utils.CAPE_FULL_COLUMN_NAME: 1.,
    gfs_utils.CAPE_BOTTOM_180MB_NAME: 1.,
    gfs_utils.CAPE_BOTTOM_255MB_NAME: 1.,
    gfs_utils.DOWNWARD_SURFACE_SHORTWAVE_FLUX_NAME: 1.,
    gfs_utils.DOWNWARD_SURFACE_LONGWAVE_FLUX_NAME: 1.,
    gfs_utils.UPWARD_SURFACE_SHORTWAVE_FLUX_NAME: 1.,
    gfs_utils.UPWARD_SURFACE_LONGWAVE_FLUX_NAME: 1.,
    gfs_utils.UPWARD_TOA_SHORTWAVE_FLUX_NAME: 1.,
    gfs_utils.UPWARD_TOA_LONGWAVE_FLUX_NAME: 1.,
    gfs_utils.VEGETATION_FRACTION_NAME: 1.,
    gfs_utils.SOIL_TYPE_NAME: 1.
}

FIELD_TO_PLOTTING_UNIT_STRING = {
    gfs_utils.TEMPERATURE_NAME: r'$^{\circ}$C',
    gfs_utils.SPECIFIC_HUMIDITY_NAME: r'g kg$^{-1}$',
    gfs_utils.GEOPOTENTIAL_HEIGHT_NAME: r'dam',
    gfs_utils.U_WIND_NAME: r'm s$^{-1}$',
    gfs_utils.V_WIND_NAME: r'm s$^{-1}$',
    gfs_utils.TEMPERATURE_2METRE_NAME: r'$^{\circ}$C',
    gfs_utils.DEWPOINT_2METRE_NAME: r'$^{\circ}$C',
    gfs_utils.SPECIFIC_HUMIDITY_2METRE_NAME: r'g kg$^{-1}$',
    gfs_utils.U_WIND_10METRE_NAME: r'm s$^{-1}$',
    gfs_utils.V_WIND_10METRE_NAME: r'm s$^{-1}$',
    gfs_utils.SURFACE_PRESSURE_NAME: 'hPa',
    gfs_utils.PRECIP_NAME: 'mm',
    gfs_utils.CONVECTIVE_PRECIP_NAME: 'mm',
    gfs_utils.SOIL_TEMPERATURE_0TO10CM_NAME: r'$^{\circ}$C',
    gfs_utils.SOIL_TEMPERATURE_10TO40CM_NAME: r'$^{\circ}$C',
    gfs_utils.SOIL_TEMPERATURE_40TO100CM_NAME: r'$^{\circ}$C',
    gfs_utils.SOIL_TEMPERATURE_100TO200CM_NAME: r'$^{\circ}$C',
    gfs_utils.SOIL_MOISTURE_0TO10CM_NAME: 'fraction',
    gfs_utils.SOIL_MOISTURE_10TO40CM_NAME: 'fraction',
    gfs_utils.SOIL_MOISTURE_40TO100CM_NAME: 'fraction',
    gfs_utils.SOIL_MOISTURE_100TO200CM_NAME: 'fraction',
    gfs_utils.PRECIPITABLE_WATER_NAME: 'mm',
    gfs_utils.SNOW_DEPTH_NAME: 'cm',
    gfs_utils.WATER_EQUIV_SNOW_DEPTH_NAME: 'mm',
    gfs_utils.CAPE_FULL_COLUMN_NAME: r'J kg$^{-1}$',
    gfs_utils.CAPE_BOTTOM_180MB_NAME: r'J kg$^{-1}$',
    gfs_utils.CAPE_BOTTOM_255MB_NAME: r'J kg$^{-1}$',
    gfs_utils.DOWNWARD_SURFACE_SHORTWAVE_FLUX_NAME: r'W m$^{-2}$',
    gfs_utils.DOWNWARD_SURFACE_LONGWAVE_FLUX_NAME: r'W m$^{-2}$',
    gfs_utils.UPWARD_SURFACE_SHORTWAVE_FLUX_NAME: r'W m$^{-2}$',
    gfs_utils.UPWARD_SURFACE_LONGWAVE_FLUX_NAME: r'W m$^{-2}$',
    gfs_utils.UPWARD_TOA_SHORTWAVE_FLUX_NAME: r'W m$^{-2}$',
    gfs_utils.UPWARD_TOA_LONGWAVE_FLUX_NAME: r'W m$^{-2}$',
    gfs_utils.VEGETATION_FRACTION_NAME: 'unitless',
    gfs_utils.SOIL_TYPE_NAME: 'unitless'
}

FIELD_NAME_TO_FANCY = {
    gfs_utils.TEMPERATURE_NAME: 'air temperature',
    gfs_utils.SPECIFIC_HUMIDITY_NAME: 'specific humidity',
    gfs_utils.GEOPOTENTIAL_HEIGHT_NAME: 'geopotential height',
    gfs_utils.U_WIND_NAME: 'zonal wind',
    gfs_utils.V_WIND_NAME: 'meridional wind',
    gfs_utils.TEMPERATURE_2METRE_NAME: '2-metre air temperature',
    gfs_utils.DEWPOINT_2METRE_NAME: '2-metre dewpoint',
    gfs_utils.SPECIFIC_HUMIDITY_2METRE_NAME: '2-metre specific humidity',
    gfs_utils.U_WIND_10METRE_NAME: '10-metre zonal wind',
    gfs_utils.V_WIND_10METRE_NAME: '10-metre meridional wind',
    gfs_utils.SURFACE_PRESSURE_NAME: 'surface pressure',
    gfs_utils.PRECIP_NAME: 'accumulated precip',
    gfs_utils.CONVECTIVE_PRECIP_NAME: 'accumulated convective precip',
    gfs_utils.SOIL_TEMPERATURE_0TO10CM_NAME: '0--10-cm soil temperature',
    gfs_utils.SOIL_TEMPERATURE_10TO40CM_NAME: '10--40-cm soil temperature',
    gfs_utils.SOIL_TEMPERATURE_40TO100CM_NAME: '40--100-cm soil temperature',
    gfs_utils.SOIL_TEMPERATURE_100TO200CM_NAME: '100--200-cm soil temperature',
    gfs_utils.SOIL_MOISTURE_0TO10CM_NAME: '0--10-cm soil moisture',
    gfs_utils.SOIL_MOISTURE_10TO40CM_NAME: '10--40-cm soil moisture',
    gfs_utils.SOIL_MOISTURE_40TO100CM_NAME: '40--100-cm soil moisture',
    gfs_utils.SOIL_MOISTURE_100TO200CM_NAME: '100--200-cm soil moisture',
    gfs_utils.PRECIPITABLE_WATER_NAME: 'precipitable water',
    gfs_utils.SNOW_DEPTH_NAME: 'snow depth',
    gfs_utils.WATER_EQUIV_SNOW_DEPTH_NAME: 'water-equiv snow depth',
    gfs_utils.CAPE_FULL_COLUMN_NAME: 'column-total CAPE',
    gfs_utils.CAPE_BOTTOM_180MB_NAME: 'bottom-180-mb CAPE',
    gfs_utils.CAPE_BOTTOM_255MB_NAME: 'bottom-255-mb CAPE',
    gfs_utils.DOWNWARD_SURFACE_SHORTWAVE_FLUX_NAME: 'downward sfc SW flux',
    gfs_utils.DOWNWARD_SURFACE_LONGWAVE_FLUX_NAME: 'downward sfc LW flux',
    gfs_utils.UPWARD_SURFACE_SHORTWAVE_FLUX_NAME: 'upward sfc SW flux',
    gfs_utils.UPWARD_SURFACE_LONGWAVE_FLUX_NAME: 'upward sfc LW flux',
    gfs_utils.UPWARD_TOA_SHORTWAVE_FLUX_NAME: 'upward TOA SW flux',
    gfs_utils.UPWARD_TOA_LONGWAVE_FLUX_NAME: 'upward TOA LW flux',
    gfs_utils.VEGETATION_FRACTION_NAME: 'vegetation fraction',
    gfs_utils.SOIL_TYPE_NAME: 'soil type'
}


def field_to_plotting_units(data_matrix_default_units, field_name):
    """Converts field to plotting units.

    :param data_matrix_default_units: numpy array of data values in default
        units.
    :param field_name: Field name.
    :return: data_matrix_plotting_units: Same as input but in plotting units.
    :return: plotting_units_string: String describing plotting units.
    """

    error_checking.assert_is_numpy_array(data_matrix_default_units)
    gfs_utils.check_field_name(field_name)

    if field_name in FIELD_TO_CONV_FACTOR:
        data_matrix_plotting_units = (
            FIELD_TO_CONV_FACTOR[field_name] * data_matrix_default_units
        )
    else:
        if 'temperature' in field_name or 'dewpoint' in field_name:
            data_matrix_plotting_units = temperature_conv.kelvins_to_celsius(
                data_matrix_default_units
            )

    return data_matrix_plotting_units, FIELD_TO_PLOTTING_UNIT_STRING[field_name]


def use_diverging_colour_scheme(field_name):
    """Determines whether to use diverging colour scheme.

    :param field_name: Field name.
    :return: use_diverging_scheme: Boolean flag.
    """

    gfs_utils.check_field_name(field_name)

    return field_name in [
        gfs_utils.U_WIND_NAME, gfs_utils.V_WIND_NAME,
        gfs_utils.U_WIND_10METRE_NAME, gfs_utils.V_WIND_10METRE_NAME
    ]


def plot_field(data_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
               colour_map_object, colour_norm_object, axes_object,
               plot_colour_bar):
    """Plots one field on a lat/long grid.

    This is a light wrapper around `fwi_plotting.plot_field`.

    :param data_matrix: See documentation for `fwi_plotting.plot_field`.
    :param grid_latitudes_deg_n: Same.
    :param grid_longitudes_deg_e: Same.
    :param colour_map_object: Same.
    :param colour_norm_object: Same.
    :param axes_object: Same.
    :param plot_colour_bar: Same.
    :return: is_longitude_positive_in_west: Same.
    """

    return fwi_plotting.plot_field(
        data_matrix=data_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        axes_object=axes_object,
        plot_colour_bar=plot_colour_bar
    )
