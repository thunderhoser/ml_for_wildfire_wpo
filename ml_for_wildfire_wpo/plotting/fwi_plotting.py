"""Methods for plotting fire-weather indices."""

import numpy
import matplotlib.colors
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml_for_wildfire_wpo.utils import canadian_fwi_utils

NAN_COLOUR = numpy.full(3, 152. / 255)

FIELD_NAME_TO_FANCY = {
    canadian_fwi_utils.FFMC_NAME: 'Fine-fuel moisture code (FFMC)',
    canadian_fwi_utils.DSR_NAME: 'Daily severity rating (DSR)',
    canadian_fwi_utils.DC_NAME: 'Drought code (DC)',
    canadian_fwi_utils.DMC_NAME: 'Duff moisture code (DMC)',
    canadian_fwi_utils.BUI_NAME: 'Build-up index (BUI)',
    canadian_fwi_utils.FWI_NAME: 'Overall fire-weather index (FWI)',
    canadian_fwi_utils.ISI_NAME: 'Initial-spread index (ISI)'
}


def field_to_colour_scheme(field_name):
    """Returns colour scheme for one field (i.e., one fire-weather index).

    :param field_name: Field name.  Must be accepted by
        `canadian_fwi_utils.check_field_name`.
    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    canadian_fwi_utils.check_field_name(field_name)

    if field_name == canadian_fwi_utils.BUI_NAME:
        colour_list = [
            [1, 0, 254],
            [1, 129, 0],
            [1, 225, 0],
            [254, 254, 0],
            [224, 161, 1],
            [255, 0, 0]
        ]
    else:
        colour_list = [
            [1, 0, 254],
            [1, 225, 0],
            [254, 254, 0],
            [224, 161, 1],
            [255, 0, 0]
        ]

    for i in range(len(colour_list)):
        colour_list[i] = numpy.array(colour_list[i], dtype=float) / 255

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(colour_list[0])
    colour_map_object.set_over(colour_list[-1])
    colour_map_object.set_bad(NAN_COLOUR)

    if field_name == canadian_fwi_utils.FWI_NAME:
        colour_bounds = numpy.array([0, 5, 10, 20, 30, 40.])
    elif field_name == canadian_fwi_utils.FFMC_NAME:
        colour_bounds = numpy.array([0, 74, 84, 88, 91, 100.])
    elif field_name == canadian_fwi_utils.DMC_NAME:
        colour_bounds = numpy.array([0, 21, 27, 40, 60, 70.])
    elif field_name == canadian_fwi_utils.DC_NAME:
        colour_bounds = numpy.array([0, 80, 190, 300, 425, 500.])
    elif field_name == canadian_fwi_utils.ISI_NAME:
        colour_bounds = numpy.array([0, 2, 5, 10, 15, 20.])
    elif field_name == canadian_fwi_utils.BUI_NAME:
        colour_bounds = numpy.array([0, 20, 30, 40, 60, 90, 100.])
    elif field_name == canadian_fwi_utils.DSR_NAME:
        colour_bounds = numpy.array([0, 1, 3, 5, 15, 20.])

    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def plot_field(data_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
               colour_map_object, colour_norm_object, axes_object,
               plot_colour_bar):
    """Plots one field on a lat/long grid.

    M = number of rows in grid
    N = number of columns in grid

    :param data_matrix: M-by-N numpy array of data values.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap` or similar).
    :param colour_norm_object: Colour-normalizer, used to map from physical
        values to colours (instance of `matplotlib.colors.BoundaryNorm` or
        similar).
    :param axes_object: Will plot on this set of axes (instance of
        `matplotlib.axes._subplots.AxesSubplot` or similar).
    :param plot_colour_bar: Boolean flag.
    :return: is_longitude_positive_in_west: Boolean flag.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(data_matrix, num_dimensions=2)
    num_grid_rows = data_matrix.shape[0]
    num_grid_columns = data_matrix.shape[1]

    error_checking.assert_is_valid_lat_numpy_array(
        grid_latitudes_deg_n, allow_nan=False
    )
    error_checking.assert_is_numpy_array(
        grid_latitudes_deg_n,
        exact_dimensions=numpy.array([num_grid_rows], dtype=int)
    )

    grid_longitudes_to_plot_deg_e = lng_conversion.convert_lng_positive_in_west(
        grid_longitudes_deg_e + 0, allow_nan=False
    )
    is_longitude_positive_in_west = True

    if numpy.any(numpy.diff(grid_longitudes_to_plot_deg_e) < 0):
        grid_longitudes_to_plot_deg_e = (
            lng_conversion.convert_lng_negative_in_west(
                grid_longitudes_to_plot_deg_e
            )
        )

        is_longitude_positive_in_west = False

    error_checking.assert_is_numpy_array(
        grid_longitudes_to_plot_deg_e,
        exact_dimensions=numpy.array([num_grid_columns], dtype=int)
    )

    # Do actual stuff.
    (
        grid_latitude_matrix_deg_n, grid_longitude_matrix_deg_e
    ) = grids.latlng_vectors_to_matrices(
        unique_latitudes_deg=grid_latitudes_deg_n,
        unique_longitudes_deg=grid_longitudes_to_plot_deg_e
    )

    data_matrix_to_plot = numpy.ma.masked_where(
        numpy.isnan(data_matrix), data_matrix
    )

    axes_object.pcolor(
        grid_longitude_matrix_deg_e, grid_latitude_matrix_deg_n,
        data_matrix_to_plot,
        cmap=colour_map_object, norm=colour_norm_object,
        edgecolors='None', zorder=-1e11
    )

    if plot_colour_bar:
        gg_plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=data_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            extend_min=True, extend_max=True
        )

    return is_longitude_positive_in_west
