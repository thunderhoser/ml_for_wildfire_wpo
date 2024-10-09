"""Plots Shapley values as spatial maps."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.interpolate import interp1d

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import number_rounding
import longitude_conversion as lng_conversion
import gg_general_utils
import file_system_utils
import error_checking
import gfs_io
import shapley_io
import border_io
import gfs_utils
import misc_utils
import era5_constant_utils
import neural_net
import plotting_utils
import gfs_plotting
import fwi_plotting

TOLERANCE = 1e-6

DATE_FORMAT = '%Y%m%d'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

GFS_SEQUENTIAL_COLOUR_MAP_OBJECT = pyplot.get_cmap('plasma')
GFS_DIVERGING_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')
SHAPLEY_DEFAULT_COLOUR_MAP_OBJECT = pyplot.get_cmap('gist_yarg')
SHAPLEY_FWI_COLOUR_MAP_OBJECT = pyplot.get_cmap('gist_yarg')

COLOUR_BAR_FONT_SIZE = 20
BORDER_COLOUR = numpy.array([139, 69, 19], dtype=float) / 255

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

SHAPLEY_FILE_ARG_NAME = 'input_shapley_file_name'
GFS_DIRECTORY_ARG_NAME = 'input_gfs_directory_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
GFS_FCST_TARGET_DIR_ARG_NAME = 'input_gfs_fcst_target_dir_name'
GFS_NORM_FILE_ARG_NAME = 'input_gfs_norm_file_name'
PREDICTOR_COLOUR_LIMITS_ARG_NAME = 'predictor_colour_limits_prctile'
SHAPLEY_COLOUR_LIMITS_ARG_NAME = 'shapley_colour_limits_prctile'
REGION_ARG_NAME = 'region_name'
SHAPLEY_LINE_WIDTH_ARG_NAME = 'shapley_line_width'
SHAPLEY_LINE_OPACITY_ARG_NAME = 'shapley_line_opacity'
PLOT_LATITUDE_LIMITS_ARG_NAME = 'plot_latitude_limits_deg_n'
PLOT_LONGITUDE_LIMITS_ARG_NAME = 'plot_longitude_limits_deg_e'
SHAPLEY_HALF_NUM_CONTOURS_ARG_NAME = 'shapley_half_num_contours'
SHAPLEY_LOG_SPACE_ARG_NAME = 'plot_shapley_in_log_space'
SHAPLEY_SMOOTHING_RADIUS_ARG_NAME = 'shapley_smoothing_radius_px'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SHAPLEY_FILE_HELP_STRING = (
    'Path to input file.  Shapley values will be read from here by '
    '`shapley_io.read_file`.'
)
GFS_DIRECTORY_HELP_STRING = (
    'Name of directory with *unnormalized* GFS weather forecasts.  Files '
    'therein will be found by `gfs_io.find_file` and read by '
    '`gfs_io.read_file`.'
)
TARGET_DIR_HELP_STRING = (
    'Name of directory with *unnormalized* target fields.  Files therein will '
    'be found by `canadian_fwo_io.find_file` and read by '
    '`canadian_fwo_io.read_file`.'
)
GFS_FCST_TARGET_DIR_HELP_STRING = (
    'Name of directory with *unnormalized* GFS fire-weather forecasts.  Files '
    'therein will be found by `gfs_daily_io.find_file` and read by '
    '`gfs_daily_io.read_file`.'
)
GFS_NORM_FILE_HELP_STRING = (
    'Path to file with GFS-normalization parameters.  If you pass this '
    'normalization file, the percentiles in `{0:s}` will be computed once -- '
    'on values in this normalization file -- and the resulting colour limits '
    'will be used for every GFS plot.  If you leave this argument empty, the '
    'percentiles in `{0:s}` will be computed for every 2-D slice of GFS data.'
).format(
    PREDICTOR_COLOUR_LIMITS_ARG_NAME
)
PREDICTOR_COLOUR_LIMITS_HELP_STRING = (
    'Colour limits for predictor values, in terms of percentile (ranging from '
    '0...100).'
)
SHAPLEY_COLOUR_LIMITS_HELP_STRING = (
    'Same as `{0:s}` but for Shapley values.  Colour limits will be computed '
    'for every 2-D slice of Shapley values.'
).format(
    PREDICTOR_COLOUR_LIMITS_ARG_NAME
)
REGION_HELP_STRING = (
    'Name of region for which Shapley values will be computed.  If you leave '
    'this empty, the region will not be reported in figure titles.'
)
SHAPLEY_LINE_WIDTH_HELP_STRING = 'Line width for Shapley-value contours.'
SHAPLEY_LINE_OPACITY_HELP_STRING = (
    'Line opacity (in range 0...1) for Shapley contours.'
)
PLOT_LATITUDE_LIMITS_HELP_STRING = (
    'Length-2 list with min/max latitudes (deg north) to plot.  If you want to '
    'plot the full domain, just leave this argument empty.'
)
PLOT_LONGITUDE_LIMITS_HELP_STRING = (
    'Length-2 list with min/max longitudes (deg east) to plot.  If you want to '
    'plot the full domain, just leave this argument empty.'
)
SHAPLEY_HALF_NUM_CONTOURS_HELP_STRING = (
    'Half-number of line contours for Shapley values.  This many contours will '
    'be used on either side of zero -- i.e., for both positive and negative '
    'values.'
)
SHAPLEY_LOG_SPACE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot Shapley values in logarithmic (linear) '
    'space.'
)
SHAPLEY_SMOOTHING_RADIUS_HELP_STRING = (
    'Smoothing radius (units = num pixels) for Shapley values.  If you do not '
    'want to smooth Shapley values, make this non-positive.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SHAPLEY_FILE_ARG_NAME, type=str, required=True,
    help=SHAPLEY_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GFS_DIRECTORY_ARG_NAME, type=str, required=True,
    help=GFS_DIRECTORY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GFS_FCST_TARGET_DIR_ARG_NAME, type=str, required=True,
    help=GFS_FCST_TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GFS_NORM_FILE_ARG_NAME, type=str, required=True,
    help=GFS_NORM_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_COLOUR_LIMITS_ARG_NAME, type=float, nargs=2, required=True,
    help=PREDICTOR_COLOUR_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHAPLEY_COLOUR_LIMITS_ARG_NAME, type=float, nargs=2, required=True,
    help=SHAPLEY_COLOUR_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + REGION_ARG_NAME, type=str, required=False, default='',
    help=REGION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHAPLEY_LINE_WIDTH_ARG_NAME, type=float, required=True,
    help=SHAPLEY_LINE_WIDTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHAPLEY_LINE_OPACITY_ARG_NAME, type=float, required=False,
    default=1., help=SHAPLEY_LINE_OPACITY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_LATITUDE_LIMITS_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1.], help=PLOT_LATITUDE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_LONGITUDE_LIMITS_ARG_NAME, type=float, nargs='+',
    required=False, default=[361.], help=PLOT_LONGITUDE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHAPLEY_HALF_NUM_CONTOURS_ARG_NAME, type=int, required=True,
    help=SHAPLEY_HALF_NUM_CONTOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHAPLEY_LOG_SPACE_ARG_NAME, type=int, required=True,
    help=SHAPLEY_LOG_SPACE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHAPLEY_SMOOTHING_RADIUS_ARG_NAME, type=float, required=True,
    help=SHAPLEY_SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_grid_line_spacing(plot_latitude_limits_deg_n,
                           plot_longitude_limits_deg_e):
    """Determines spacing of meridians and parallels in plot.

    :param plot_latitude_limits_deg_n: length-2 numpy array with latitude limits
        of plot.
    :param plot_longitude_limits_deg_e: length-2 numpy array with longitude
        limits of plot.
    :return: parallel_spacing_deg: Spacing between successive parallels.
    :return: meridian_spacing_deg: Spacing between successive meridians.
    """

    meridian_spacing_deg = (
        numpy.max(plot_longitude_limits_deg_e) -
        numpy.min(plot_longitude_limits_deg_e)
    ) / 7
    parallel_spacing_deg = (
        numpy.max(plot_latitude_limits_deg_n) -
        numpy.min(plot_latitude_limits_deg_n)
    ) / 6

    if meridian_spacing_deg > 1:
        meridian_spacing_deg = numpy.floor(meridian_spacing_deg)
    else:
        meridian_spacing_deg = number_rounding.floor_to_nearest(
            meridian_spacing_deg, 0.1
        )

    if parallel_spacing_deg > 1:
        parallel_spacing_deg = numpy.floor(parallel_spacing_deg)
    else:
        parallel_spacing_deg = number_rounding.floor_to_nearest(
            parallel_spacing_deg, 0.1
        )

    return parallel_spacing_deg, meridian_spacing_deg


def _smooth_maps(shapley_table_xarray, smoothing_radius_px):
    """Smooths Shapley maps via Gaussian filter.

    :param shapley_table_xarray: xarray table returned by
        `shapley_io.read_file`.
    :param smoothing_radius_px: e-folding radius (num pixels).
    :return: shapley_table_xarray: Same as input but with smoothed maps.
    """

    shapley_keys = [
        shapley_io.SHAPLEY_FOR_3D_GFS_KEY, shapley_io.SHAPLEY_FOR_2D_GFS_KEY,
        shapley_io.SHAPLEY_FOR_LAGLEAD_TARGETS_KEY,
        shapley_io.SHAPLEY_FOR_ERA5_KEY,
        shapley_io.SHAPLEY_FOR_PREDN_BASELINE_KEY
    ]
    stx = shapley_table_xarray

    print((
        'Smoothing maps with Gaussian filter (e-folding radius of {0:.1f} grid '
        'cells)...'
    ).format(
        smoothing_radius_px
    ))

    for this_key in shapley_keys:
        if this_key not in stx.data_vars:
            continue

        this_shapley_matrix = stx[this_key].values

        if len(this_shapley_matrix.shape) == 3:
            for i in range(this_shapley_matrix.shape[2]):
                this_shapley_matrix[..., i] = (
                    gg_general_utils.apply_gaussian_filter(
                        input_matrix=this_shapley_matrix[..., i],
                        e_folding_radius_grid_cells=smoothing_radius_px
                    )
                )
        elif len(this_shapley_matrix.shape) == 4:
            for i in range(this_shapley_matrix.shape[2]):
                for j in range(this_shapley_matrix.shape[3]):
                    this_shapley_matrix[..., i, j] = (
                        gg_general_utils.apply_gaussian_filter(
                            input_matrix=this_shapley_matrix[..., i, j],
                            e_folding_radius_grid_cells=smoothing_radius_px
                        )
                    )
        elif len(this_shapley_matrix.shape) == 5:
            for i in range(this_shapley_matrix.shape[2]):
                for j in range(this_shapley_matrix.shape[3]):
                    for k in range(this_shapley_matrix.shape[4]):
                        this_shapley_matrix[..., i, j, k] = (
                            gg_general_utils.apply_gaussian_filter(
                                input_matrix=this_shapley_matrix[..., i, j, k],
                                e_folding_radius_grid_cells=smoothing_radius_px
                            )
                        )

        stx = stx.assign({
            this_key: (stx[this_key].dims, this_shapley_matrix)
        })

    shapley_table_xarray = stx
    return shapley_table_xarray


def _plot_one_shapley_field(
        shapley_matrix, axes_object, figure_object, grid_latitudes_deg_n,
        grid_longitudes_deg_e, min_colour_percentile, max_colour_percentile,
        half_num_contours, colour_map_object, line_width, opacity,
        output_file_name, plot_in_log_space=False):
    """Plots 2-D field of Shapley values as a spatial map.

    M = number of rows in grid
    N = number of columns in grid

    :param shapley_matrix: M-by-N numpy array of Shapley values.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param min_colour_percentile: Determines minimum absolute Shapley value to
        plot.
    :param max_colour_percentile: Determines max absolute Shapley value to plot.
    :param half_num_contours: Number of contours on either side of zero.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param line_width: Width of contour lines.
    :param opacity: Opacity of contour lines.
    :param output_file_name: Path to output file.
    :param plot_in_log_space: Boolean flag.  If True (False), colour scale will
        be logarithmic in base 10 (linear).
    :return: min_abs_contour_value: Minimum absolute contour value.
    :return: max_abs_contour_value: Max absolute contour value.
    """

    # Check input args.
    error_checking.assert_is_boolean(plot_in_log_space)
    num_grid_rows = shapley_matrix.shape[0]
    num_grid_columns = shapley_matrix.shape[1]

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

    if numpy.any(numpy.diff(grid_longitudes_to_plot_deg_e) < 0):
        grid_longitudes_to_plot_deg_e = (
            lng_conversion.convert_lng_negative_in_west(
                grid_longitudes_to_plot_deg_e
            )
        )

    error_checking.assert_is_numpy_array(
        grid_longitudes_to_plot_deg_e,
        exact_dimensions=numpy.array([num_grid_columns], dtype=int)
    )

    error_checking.assert_is_integer(half_num_contours)
    error_checking.assert_is_geq(half_num_contours, 5)

    if plot_in_log_space:
        scaled_shapley_matrix = numpy.log10(
            1 + numpy.absolute(shapley_matrix)
        )
    else:
        scaled_shapley_matrix = shapley_matrix

    min_abs_contour_value = numpy.percentile(
        numpy.absolute(scaled_shapley_matrix), min_colour_percentile
    )
    max_abs_contour_value = numpy.percentile(
        numpy.absolute(scaled_shapley_matrix), max_colour_percentile
    )
    max_abs_contour_value = max([
        max_abs_contour_value,
        min_abs_contour_value + 1e-9
    ])

    (
        grid_latitude_matrix_deg_n, grid_longitude_matrix_deg_e
    ) = grids.latlng_vectors_to_matrices(
        unique_latitudes_deg=grid_latitudes_deg_n,
        unique_longitudes_deg=grid_longitudes_to_plot_deg_e
    )

    # Plot positive values.
    contour_levels = numpy.linspace(
        min_abs_contour_value, max_abs_contour_value, num=half_num_contours
    )

    if plot_in_log_space:
        positive_shapley_matrix = numpy.log10(
            1 + numpy.maximum(shapley_matrix, 0.)
        )
    else:
        positive_shapley_matrix = shapley_matrix

    axes_object.contour(
        grid_longitude_matrix_deg_e, grid_latitude_matrix_deg_n,
        positive_shapley_matrix,
        contour_levels, cmap=colour_map_object,
        vmin=numpy.min(contour_levels), vmax=numpy.max(contour_levels),
        linewidths=line_width, linestyles='solid', alpha=opacity,
        zorder=1e6
    )

    if plot_in_log_space:
        negative_shapley_matrix = numpy.log10(
            1 + numpy.absolute(numpy.minimum(shapley_matrix, 0.))
        )
    else:
        negative_shapley_matrix = -shapley_matrix

    # Plot negative values.
    axes_object.contour(
        grid_longitude_matrix_deg_e, grid_latitude_matrix_deg_n,
        negative_shapley_matrix,
        contour_levels, cmap=colour_map_object,
        vmin=numpy.min(contour_levels), vmax=numpy.max(contour_levels),
        linewidths=line_width, linestyles='dotted', alpha=opacity,
        zorder=1e6
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return min_abs_contour_value, max_abs_contour_value


def _plot_one_fwi_field(
        data_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e,
        field_name, title_string):
    """Plots a single 2-D fire-weather field with Shapley values on top.

    M = number of rows in grid
    N = number of columns in grid
    P = number of points in border file

    :param data_matrix: M-by-N numpy array of data values.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg
        north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg
        east).
    :param field_name: Field name.
    :param title_string: Title.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    colour_map_object, colour_norm_object = fwi_plotting.field_to_colour_scheme(
        field_name
    )

    is_longitude_positive_in_west = fwi_plotting.plot_field(
        data_matrix=data_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        axes_object=axes_object, plot_colour_bar=True
    )

    if is_longitude_positive_in_west:
        border_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            border_longitudes_deg_e
        )
        grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            grid_longitudes_deg_e
        )
    else:
        border_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            border_longitudes_deg_e
        )
        grid_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            grid_longitudes_deg_e
        )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )

    parallel_spacing_deg, meridian_spacing_deg = _get_grid_line_spacing(
        plot_latitude_limits_deg_n=grid_latitudes_deg_n,
        plot_longitude_limits_deg_e=grid_longitudes_deg_e
    )
    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=grid_latitudes_deg_n,
        plot_longitudes_deg_e=grid_longitudes_deg_e,
        axes_object=axes_object,
        meridian_spacing_deg=meridian_spacing_deg,
        parallel_spacing_deg=parallel_spacing_deg
    )

    axes_object.set_xlim(
        numpy.min(grid_longitudes_deg_e),
        numpy.max(grid_longitudes_deg_e)
    )
    axes_object.set_ylim(
        numpy.min(grid_latitudes_deg_n),
        numpy.max(grid_latitudes_deg_n)
    )
    axes_object.set_title(title_string)

    return figure_object, axes_object


def _plot_one_gfs_field(
        data_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e,
        field_name, min_colour_percentile, max_colour_percentile,
        title_string, gfs_norm_param_table_xarray=None):
    """Plots a single 2-D GFS field with Shapley values on top.

    M = number of rows in grid
    N = number of columns in grid
    P = number of points in border file

    :param data_matrix: M-by-N numpy array of data values.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg
        north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg
        east).
    :param field_name: Field name.
    :param min_colour_percentile: See documentation at top of script.
    :param max_colour_percentile: Same.
    :param title_string: Title.
    :param gfs_norm_param_table_xarray: xarray table with normalization params
        for GFS data -- in format returned by `gfs_io.read_normalization_file`
        -- but subset to the relevant field, pressure level (if applicable),
        and forecast hour (if applicable).
        If you want colour limits to be based on the current 2-D slice of data
        (`data_matrix`), leave this argument alone.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if gfs_plotting.use_diverging_colour_scheme(field_name):
        colour_map_object = GFS_DIVERGING_COLOUR_MAP_OBJECT

        if gfs_norm_param_table_xarray is None:
            max_colour_value = numpy.nanpercentile(
                numpy.absolute(data_matrix), max_colour_percentile
            )
        else:
            gfs_npt = gfs_norm_param_table_xarray

            if field_name in gfs_utils.ALL_2D_FIELD_NAMES:
                y_values = gfs_npt[gfs_utils.QUANTILE_KEY_2D].values[0, 0, :]
            else:
                y_values = gfs_npt[gfs_utils.QUANTILE_KEY_3D].values[0, 0, :]

            interp_object = interp1d(
                x=gfs_npt.coords[gfs_utils.QUANTILE_LEVEL_DIM].values,
                y=y_values, kind='linear', bounds_error=True, assume_sorted=True
            )

            new_percentile = numpy.mean(
                numpy.array([max_colour_percentile, 100.])
            )
            max_colour_value = interp_object(new_percentile * 0.01)

        if numpy.isnan(max_colour_value):
            max_colour_value = TOLERANCE

        max_colour_value = max([max_colour_value, TOLERANCE])
        min_colour_value = -1 * max_colour_value
    else:
        colour_map_object = GFS_SEQUENTIAL_COLOUR_MAP_OBJECT

        if gfs_norm_param_table_xarray is None:
            min_colour_value = numpy.nanpercentile(
                data_matrix, min_colour_percentile
            )
            max_colour_value = numpy.nanpercentile(
                data_matrix, max_colour_percentile
            )
        else:
            gfs_npt = gfs_norm_param_table_xarray

            if field_name in gfs_utils.ALL_2D_FIELD_NAMES:
                y_values = gfs_npt[gfs_utils.QUANTILE_KEY_2D].values[0, 0, :]
            else:
                y_values = gfs_npt[gfs_utils.QUANTILE_KEY_3D].values[0, 0, :]

            interp_object = interp1d(
                x=gfs_npt.coords[gfs_utils.QUANTILE_LEVEL_DIM].values,
                y=y_values, kind='linear', bounds_error=True, assume_sorted=True
            )
            min_colour_value = interp_object(min_colour_percentile * 0.01)
            max_colour_value = interp_object(max_colour_percentile * 0.01)

        if numpy.isnan(min_colour_value):
            min_colour_value = 0.
            max_colour_value = TOLERANCE

        max_colour_value = max([max_colour_value, min_colour_value + TOLERANCE])

    data_matrix = gfs_plotting.field_to_plotting_units(
        data_matrix_default_units=data_matrix, field_name=field_name
    )[0]
    colour_limits = gfs_plotting.field_to_plotting_units(
        data_matrix_default_units=
        numpy.array([min_colour_value, max_colour_value]),
        field_name=field_name
    )[0]
    min_colour_value = colour_limits[0]
    max_colour_value = colour_limits[1]

    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )
    plot_in_log2_scale = field_name in [
        gfs_utils.PRECIP_NAME, gfs_utils.CONVECTIVE_PRECIP_NAME
    ]

    is_longitude_positive_in_west = gfs_plotting.plot_field(
        data_matrix=data_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        axes_object=axes_object,
        plot_colour_bar=True,
        plot_in_log2_scale=plot_in_log2_scale
    )

    if is_longitude_positive_in_west:
        border_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            border_longitudes_deg_e
        )
        grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            grid_longitudes_deg_e
        )
    else:
        border_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            border_longitudes_deg_e
        )
        grid_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            grid_longitudes_deg_e
        )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    parallel_spacing_deg, meridian_spacing_deg = _get_grid_line_spacing(
        plot_latitude_limits_deg_n=grid_latitudes_deg_n,
        plot_longitude_limits_deg_e=grid_longitudes_deg_e
    )
    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=grid_latitudes_deg_n,
        plot_longitudes_deg_e=grid_longitudes_deg_e,
        axes_object=axes_object,
        meridian_spacing_deg=meridian_spacing_deg,
        parallel_spacing_deg=parallel_spacing_deg
    )

    axes_object.set_xlim(
        numpy.min(grid_longitudes_deg_e),
        numpy.max(grid_longitudes_deg_e)
    )
    axes_object.set_ylim(
        numpy.min(grid_latitudes_deg_n),
        numpy.max(grid_latitudes_deg_n)
    )
    axes_object.set_title(title_string)

    return figure_object, axes_object


def _plot_one_era5_field(
        data_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e,
        min_colour_percentile, max_colour_percentile, title_string):
    """Plots a single ERA5 time-constant field with Shapley values on top.

    :param data_matrix: See documentation for `_plot_one_gfs_field`.
    :param grid_latitudes_deg_n: Same.
    :param grid_longitudes_deg_e: Same.
    :param border_latitudes_deg_n: Same.
    :param border_longitudes_deg_e: Same.
    :param min_colour_percentile: Same.
    :param max_colour_percentile: Same.
    :param title_string: Same.
    :return: figure_object: Same.
    :return: axes_object: Same.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    colour_map_object = GFS_SEQUENTIAL_COLOUR_MAP_OBJECT
    min_colour_value = numpy.nanpercentile(
        data_matrix, min_colour_percentile
    )
    max_colour_value = numpy.nanpercentile(
        data_matrix, max_colour_percentile
    )

    if numpy.isnan(min_colour_value):
        min_colour_value = 0.
        max_colour_value = TOLERANCE

    max_colour_value = max([max_colour_value, min_colour_value + TOLERANCE])

    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )

    is_longitude_positive_in_west = gfs_plotting.plot_field(
        data_matrix=data_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        axes_object=axes_object,
        plot_colour_bar=True,
        plot_in_log2_scale=False
    )

    if is_longitude_positive_in_west:
        border_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            border_longitudes_deg_e
        )
        grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            grid_longitudes_deg_e
        )
    else:
        border_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            border_longitudes_deg_e
        )
        grid_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            grid_longitudes_deg_e
        )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    parallel_spacing_deg, meridian_spacing_deg = _get_grid_line_spacing(
        plot_latitude_limits_deg_n=grid_latitudes_deg_n,
        plot_longitude_limits_deg_e=grid_longitudes_deg_e
    )
    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=grid_latitudes_deg_n,
        plot_longitudes_deg_e=grid_longitudes_deg_e,
        axes_object=axes_object,
        meridian_spacing_deg=meridian_spacing_deg,
        parallel_spacing_deg=parallel_spacing_deg
    )

    axes_object.set_xlim(
        numpy.min(grid_longitudes_deg_e),
        numpy.max(grid_longitudes_deg_e)
    )
    axes_object.set_ylim(
        numpy.min(grid_latitudes_deg_n),
        numpy.max(grid_latitudes_deg_n)
    )
    axes_object.set_title(title_string)

    return figure_object, axes_object


def _run(shapley_file_name, gfs_directory_name, target_dir_name,
         gfs_forecast_target_dir_name, gfs_normalization_file_name,
         predictor_colour_limits_prctile, shapley_colour_limits_prctile,
         region_name, shapley_line_width, shapley_line_opacity,
         plot_latitude_limits_deg_n, plot_longitude_limits_deg_e,
         shapley_half_num_contours, plot_shapley_in_log_space,
         shapley_smoothing_radius_px, output_dir_name):
    """Plots Shapley values as spatial maps.

    This is effectively the main method.

    :param shapley_file_name: See documentation at top of this script.
    :param gfs_directory_name: Same.
    :param target_dir_name: Same.
    :param gfs_forecast_target_dir_name: Same.
    :param gfs_normalization_file_name: Same.
    :param predictor_colour_limits_prctile: Same.
    :param shapley_colour_limits_prctile: Same.
    :param region_name: Same.
    :param shapley_line_width: Same.
    :param shapley_line_opacity: Same.
    :param plot_latitude_limits_deg_n: Same.
    :param plot_longitude_limits_deg_e: Same.
    :param shapley_half_num_contours: Same.
    :param plot_shapley_in_log_space: Same.
    :param shapley_smoothing_radius_px: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    predictor_colour_limits_prctile = numpy.sort(
        predictor_colour_limits_prctile
    )
    error_checking.assert_is_geq(predictor_colour_limits_prctile[0], 0.)
    error_checking.assert_is_leq(predictor_colour_limits_prctile[0], 10.)
    error_checking.assert_is_geq(predictor_colour_limits_prctile[1], 90.)
    error_checking.assert_is_leq(predictor_colour_limits_prctile[1], 100.)

    shapley_colour_limits_prctile = numpy.sort(shapley_colour_limits_prctile)
    error_checking.assert_is_geq(shapley_colour_limits_prctile[0], 0.)
    error_checking.assert_is_leq(shapley_colour_limits_prctile[0], 10.)
    error_checking.assert_is_geq(shapley_colour_limits_prctile[1], 90.)
    error_checking.assert_is_leq(shapley_colour_limits_prctile[1], 100.)

    if region_name == '':
        region_name = None

    error_checking.assert_is_greater(shapley_line_width, 0.)
    if shapley_smoothing_radius_px < TOLERANCE:
        shapley_smoothing_radius_px = None

    error_checking.assert_is_greater(shapley_line_opacity, 0.)
    error_checking.assert_is_leq(shapley_line_opacity, 1.)

    if (
            len(plot_latitude_limits_deg_n) == 1 and
            plot_latitude_limits_deg_n[0] < 0
    ):
        plot_latitude_limits_deg_n = None

    if (
            len(plot_longitude_limits_deg_e) == 1 and
            plot_longitude_limits_deg_e[0] > 360
    ):
        plot_longitude_limits_deg_e = None

    if plot_latitude_limits_deg_n is not None:
        error_checking.assert_is_numpy_array(
            plot_latitude_limits_deg_n,
            exact_dimensions=numpy.array([2], dtype=int)
        )
        error_checking.assert_is_valid_lat_numpy_array(
            plot_latitude_limits_deg_n
        )

        plot_latitude_limits_deg_n = numpy.sort(plot_latitude_limits_deg_n)
        error_checking.assert_is_greater_numpy_array(
            numpy.diff(plot_latitude_limits_deg_n), 0.
        )

    if plot_longitude_limits_deg_e is not None:
        error_checking.assert_is_numpy_array(
            plot_longitude_limits_deg_e,
            exact_dimensions=numpy.array([2], dtype=int)
        )
        error_checking.assert_is_valid_lng_numpy_array(
            plot_longitude_limits_deg_e,
            positive_in_west_flag=False, negative_in_west_flag=False,
            allow_nan=False
        )
        error_checking.assert_is_greater_numpy_array(
            numpy.absolute(numpy.diff(plot_longitude_limits_deg_e)),
            0
        )

    if gfs_normalization_file_name is None:
        gfs_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            gfs_normalization_file_name
        ))
        gfs_norm_param_table_xarray = gfs_io.read_normalization_file(
            gfs_normalization_file_name
        )

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read Shapley values.
    print('Reading Shapley values from: "{0:s}"...'.format(shapley_file_name))
    shapley_table_xarray = shapley_io.read_file(shapley_file_name)
    if shapley_smoothing_radius_px is not None:
        shapley_table_xarray = _smooth_maps(
            shapley_table_xarray=shapley_table_xarray,
            smoothing_radius_px=shapley_smoothing_radius_px
        )

    stx = shapley_table_xarray
    model_lead_time_days = stx.attrs[shapley_io.MODEL_LEAD_TIME_KEY]

    # Read model metadata.
    model_file_name = stx.attrs[shapley_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    vod = model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]

    vod[neural_net.GFS_DIRECTORY_KEY] = gfs_directory_name
    vod[neural_net.TARGET_DIRECTORY_KEY] = target_dir_name
    vod[neural_net.GFS_FORECAST_TARGET_DIR_KEY] = gfs_forecast_target_dir_name
    vod[neural_net.GFS_NORM_FILE_KEY] = None
    vod[neural_net.TARGET_NORM_FILE_KEY] = None
    vod[neural_net.ERA5_NORM_FILE_KEY] = None
    vod[neural_net.SENTINEL_VALUE_KEY] = None
    vod[neural_net.MODEL_LEAD_TO_GFS_PRED_LEADS_KEY] = {
        model_lead_time_days: vod[
            neural_net.MODEL_LEAD_TO_GFS_PRED_LEADS_KEY
        ][model_lead_time_days]
    }
    vod[neural_net.MODEL_LEAD_TO_TARGET_LAGS_KEY] = {
        model_lead_time_days: vod[
            neural_net.MODEL_LEAD_TO_TARGET_LAGS_KEY
        ][model_lead_time_days]
    }
    vod[neural_net.MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY] = {
        model_lead_time_days: vod[
            neural_net.MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY
        ][model_lead_time_days]
    }

    validation_option_dict = vod

    init_date_string = stx.attrs[shapley_io.INIT_DATE_KEY]
    init_date_string_nice = (
        init_date_string[:4] +
        '-' + init_date_string[4:6] +
        '-' + init_date_string[6:]
    )
    target_field_name = stx.attrs[shapley_io.TARGET_FIELD_KEY]

    print(SEPARATOR_STRING)

    try:
        data_dict = neural_net.create_data(
            option_dict=validation_option_dict,
            init_date_string=init_date_string,
            model_lead_time_days=model_lead_time_days
        )
        print(SEPARATOR_STRING)
    except:
        print(SEPARATOR_STRING)
        return

    predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    model_input_layer_names = data_dict[neural_net.INPUT_LAYER_NAMES_KEY]
    del data_dict

    if plot_latitude_limits_deg_n is not None:
        desired_row_indices = misc_utils.desired_latitudes_to_rows(
            grid_latitudes_deg_n=stx[shapley_io.LATITUDE_KEY].values,
            start_latitude_deg_n=plot_latitude_limits_deg_n[0],
            end_latitude_deg_n=plot_latitude_limits_deg_n[1]
        )
        stx = stx.isel({shapley_io.ROW_DIM: desired_row_indices})

        for k in range(len(predictor_matrices)):
            if len(predictor_matrices[k].shape) < 2:
                continue

            predictor_matrices[k] = (
                predictor_matrices[k][:, desired_row_indices, ...]
            )

    if plot_longitude_limits_deg_e is not None:
        desired_column_indices = misc_utils.desired_longitudes_to_columns(
            grid_longitudes_deg_e=stx[shapley_io.LONGITUDE_KEY].values,
            start_longitude_deg_e=plot_longitude_limits_deg_e[0],
            end_longitude_deg_e=plot_longitude_limits_deg_e[1]
        )
        stx = stx.isel({shapley_io.COLUMN_DIM: desired_column_indices})

        for k in range(len(predictor_matrices)):
            if len(predictor_matrices[k].shape) < 2:
                continue

            predictor_matrices[k] = (
                predictor_matrices[k][:, :, desired_column_indices, ...]
            )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    gfs_lead_times_hours = vod[neural_net.MODEL_LEAD_TO_GFS_PRED_LEADS_KEY][
        model_lead_time_days
    ]
    gfs_field_names = vod[neural_net.GFS_PREDICTOR_FIELDS_KEY]
    gfs_field_names_2d = [
        f for f in gfs_field_names
        if f in gfs_utils.ALL_2D_FIELD_NAMES
    ]

    if len(gfs_field_names_2d) > 0:
        lyr_idx = model_input_layer_names.index(neural_net.GFS_2D_LAYER_NAME)
        gfs_2d_predictor_matrix = predictor_matrices[lyr_idx][0, ...]
    else:
        gfs_2d_predictor_matrix = numpy.array([])

    for t in range(len(gfs_lead_times_hours)):
        for f in range(len(gfs_field_names_2d)):
            this_predictor_matrix = gfs_2d_predictor_matrix[..., t, f] + 0.
            unit_string = gfs_plotting.FIELD_TO_PLOTTING_UNIT_STRING[
                gfs_field_names_2d[f]
            ]

            title_string = (
                'GFS {0:s} ({1:s}), {2:s} + {3:d} hours\n'
                'Shapley values for {4:d}-day {5:s}{6:s}'
            ).format(
                gfs_plotting.FIELD_NAME_TO_FANCY[gfs_field_names_2d[f]],
                unit_string,
                init_date_string_nice,
                gfs_lead_times_hours[t],
                model_lead_time_days,
                fwi_plotting.FIELD_NAME_TO_SIMPLE[target_field_name],
                '' if region_name is None else ' over ' + region_name
            )

            output_file_name = (
                '{0:s}/shapley_gfs_{1:s}_{2:03d}hours.jpg'
            ).format(
                output_dir_name,
                gfs_field_names_2d[f].replace('_', '-'),
                gfs_lead_times_hours[t]
            )

            if gfs_norm_param_table_xarray is None:
                this_norm_param_table_xarray = None
            else:
                gfs_npt = gfs_norm_param_table_xarray

                t_idxs = numpy.where(
                    gfs_npt.coords[gfs_utils.FORECAST_HOUR_DIM].values ==
                    gfs_lead_times_hours[t]
                )[0]
                this_npt = gfs_npt.isel({gfs_utils.FORECAST_HOUR_DIM: t_idxs})

                f_idxs = numpy.where(
                    this_npt.coords[gfs_utils.FIELD_DIM_2D].values ==
                    gfs_field_names_2d[f]
                )[0]
                this_npt = this_npt.isel({gfs_utils.FIELD_DIM_2D: f_idxs})

                this_norm_param_table_xarray = this_npt

            figure_object, axes_object = _plot_one_gfs_field(
                data_matrix=this_predictor_matrix,
                grid_latitudes_deg_n=stx[shapley_io.LATITUDE_KEY].values,
                grid_longitudes_deg_e=stx[shapley_io.LONGITUDE_KEY].values,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                field_name=gfs_field_names_2d[f],
                min_colour_percentile=predictor_colour_limits_prctile[0],
                max_colour_percentile=predictor_colour_limits_prctile[1],
                title_string=title_string,
                gfs_norm_param_table_xarray=this_norm_param_table_xarray
            )

            min_colour_value, max_colour_value = _plot_one_shapley_field(
                shapley_matrix=
                stx[shapley_io.SHAPLEY_FOR_2D_GFS_KEY].values[..., t, f],
                axes_object=axes_object,
                figure_object=figure_object,
                grid_latitudes_deg_n=stx[shapley_io.LATITUDE_KEY].values,
                grid_longitudes_deg_e=stx[shapley_io.LONGITUDE_KEY].values,
                min_colour_percentile=shapley_colour_limits_prctile[0],
                max_colour_percentile=shapley_colour_limits_prctile[1],
                half_num_contours=shapley_half_num_contours,
                colour_map_object=SHAPLEY_DEFAULT_COLOUR_MAP_OBJECT,
                line_width=shapley_line_width,
                opacity=shapley_line_opacity,
                plot_in_log_space=plot_shapley_in_log_space,
                output_file_name=output_file_name
            )

            colour_norm_object = pyplot.Normalize(
                vmin=min_colour_value, vmax=max_colour_value
            )
            plotting_utils.add_colour_bar(
                figure_file_name=output_file_name,
                colour_map_object=SHAPLEY_DEFAULT_COLOUR_MAP_OBJECT,
                colour_norm_object=colour_norm_object,
                orientation_string='vertical',
                font_size=COLOUR_BAR_FONT_SIZE,
                cbar_label_string='Absolute Shapley value',
                tick_label_format_string='{0:.2g}',
                log_space=plot_shapley_in_log_space
            )

    gfs_pressure_levels_mb = vod[neural_net.GFS_PRESSURE_LEVELS_KEY]
    gfs_field_names_3d = [
        f for f in gfs_field_names
        if f in gfs_utils.ALL_3D_FIELD_NAMES
    ]

    if len(gfs_field_names_3d) > 0:
        lyr_idx = model_input_layer_names.index(neural_net.GFS_3D_LAYER_NAME)
        gfs_3d_predictor_matrix = predictor_matrices[lyr_idx][0, ...]
    else:
        gfs_3d_predictor_matrix = numpy.array([])

    for p in range(len(gfs_pressure_levels_mb)):
        for t in range(len(gfs_lead_times_hours)):
            for f in range(len(gfs_field_names_3d)):

                this_predictor_matrix = (
                    gfs_3d_predictor_matrix[..., p, t, f] + 0.
                )
                unit_string = gfs_plotting.FIELD_TO_PLOTTING_UNIT_STRING[
                    gfs_field_names_3d[f]
                ]

                title_string = (
                    'GFS {0:s} ({1:s}), {2:.0f} mb, {3:s} + {4:d} hours\n'
                    'Shapley values for {5:d}-day {6:s}{7:s}'
                ).format(
                    gfs_plotting.FIELD_NAME_TO_FANCY[gfs_field_names_3d[f]],
                    unit_string,
                    gfs_pressure_levels_mb[p],
                    init_date_string_nice,
                    gfs_lead_times_hours[t],
                    model_lead_time_days,
                    fwi_plotting.FIELD_NAME_TO_SIMPLE[target_field_name],
                    '' if region_name is None else ' over ' + region_name
                )

                output_file_name = (
                    '{0:s}/shapley_gfs_{1:s}_{2:04d}mb_{3:03d}hours.jpg'
                ).format(
                    output_dir_name,
                    gfs_field_names_3d[f].replace('_', '-'),
                    int(numpy.round(gfs_pressure_levels_mb[p])),
                    gfs_lead_times_hours[t]
                )

                if gfs_norm_param_table_xarray is None:
                    this_norm_param_table_xarray = None
                else:
                    gfs_npt = gfs_norm_param_table_xarray

                    p_idxs = numpy.where(
                        gfs_npt.coords[gfs_utils.PRESSURE_LEVEL_DIM].values ==
                        gfs_pressure_levels_mb[p]
                    )[0]
                    this_npt = gfs_npt.isel(
                        {gfs_utils.PRESSURE_LEVEL_DIM: p_idxs}
                    )

                    f_idxs = numpy.where(
                        this_npt.coords[gfs_utils.FIELD_DIM_3D].values ==
                        gfs_field_names_3d[f]
                    )[0]
                    this_npt = this_npt.isel({gfs_utils.FIELD_DIM_3D: f_idxs})

                    this_norm_param_table_xarray = this_npt

                figure_object, axes_object = _plot_one_gfs_field(
                    data_matrix=this_predictor_matrix,
                    grid_latitudes_deg_n=stx[shapley_io.LATITUDE_KEY].values,
                    grid_longitudes_deg_e=stx[shapley_io.LONGITUDE_KEY].values,
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    field_name=gfs_field_names_3d[f],
                    min_colour_percentile=predictor_colour_limits_prctile[0],
                    max_colour_percentile=predictor_colour_limits_prctile[1],
                    title_string=title_string,
                    gfs_norm_param_table_xarray=this_norm_param_table_xarray
                )

                min_colour_value, max_colour_value = _plot_one_shapley_field(
                    shapley_matrix=
                    stx[shapley_io.SHAPLEY_FOR_3D_GFS_KEY].values[..., p, t, f],
                    axes_object=axes_object,
                    figure_object=figure_object,
                    grid_latitudes_deg_n=stx[shapley_io.LATITUDE_KEY].values,
                    grid_longitudes_deg_e=stx[shapley_io.LONGITUDE_KEY].values,
                    min_colour_percentile=shapley_colour_limits_prctile[0],
                    max_colour_percentile=shapley_colour_limits_prctile[1],
                    half_num_contours=shapley_half_num_contours,
                    colour_map_object=SHAPLEY_DEFAULT_COLOUR_MAP_OBJECT,
                    line_width=shapley_line_width,
                    opacity=shapley_line_opacity,
                    plot_in_log_space=plot_shapley_in_log_space,
                    output_file_name=output_file_name
                )

                colour_norm_object = pyplot.Normalize(
                    vmin=min_colour_value, vmax=max_colour_value
                )
                plotting_utils.add_colour_bar(
                    figure_file_name=output_file_name,
                    colour_map_object=SHAPLEY_DEFAULT_COLOUR_MAP_OBJECT,
                    colour_norm_object=colour_norm_object,
                    orientation_string='vertical',
                    font_size=COLOUR_BAR_FONT_SIZE,
                    cbar_label_string='Absolute Shapley value',
                    tick_label_format_string='{0:.2g}',
                    log_space=plot_shapley_in_log_space
                )

    # TODO(thunderhoser): This code does not yet handle the case where lagged
    # targets or GFS fire-weather forecasts are missing.
    target_lag_times_days = vod[neural_net.MODEL_LEAD_TO_TARGET_LAGS_KEY][
        model_lead_time_days
    ]
    gfs_target_lead_times_days = vod[
        neural_net.MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY
    ][model_lead_time_days]

    target_lag_or_lead_times_days = numpy.concatenate([
        target_lag_times_days, -1 * gfs_target_lead_times_days
    ])
    all_target_field_names = vod[neural_net.TARGET_FIELDS_KEY]

    lyr_idx = model_input_layer_names.index(
        neural_net.LAGLEAD_TARGET_LAYER_NAME
    )
    laglead_target_predictor_matrix = predictor_matrices[lyr_idx][0, ...]

    for t in range(len(target_lag_or_lead_times_days)):
        for f in range(len(all_target_field_names)):
            this_predictor_matrix = laglead_target_predictor_matrix[..., t, f]

            if target_lag_or_lead_times_days[t] < 0:
                title_string = (
                    'GFS-forecast {0:s}, {1:s} + {2:d} days\n'
                    'Shapley values for {3:d}-day {4:s}{5:s}'
                ).format(
                    fwi_plotting.FIELD_NAME_TO_SIMPLE[
                        all_target_field_names[f]
                    ],
                    init_date_string_nice,
                    -1 * target_lag_or_lead_times_days[t],
                    model_lead_time_days,
                    fwi_plotting.FIELD_NAME_TO_SIMPLE[target_field_name],
                    '' if region_name is None else ' over ' + region_name
                )
            else:
                title_string = (
                    'Lagged-truth {0:s}, {1:s} - {2:d} days\n'
                    'Shapley values for {3:d}-day {4:s}{5:s}'
                ).format(
                    fwi_plotting.FIELD_NAME_TO_SIMPLE[
                        all_target_field_names[f]
                    ],
                    init_date_string_nice,
                    target_lag_or_lead_times_days[t],
                    model_lead_time_days,
                    fwi_plotting.FIELD_NAME_TO_SIMPLE[target_field_name],
                    '' if region_name is None else ' over ' + region_name
                )

            output_file_name = (
                '{0:s}/shapley_laglead_target_{1:s}_{2:+03d}days.jpg'
            ).format(
                output_dir_name,
                all_target_field_names[f].replace('_', '-'),
                -1 * target_lag_or_lead_times_days[t]
            )

            figure_object, axes_object = _plot_one_fwi_field(
                data_matrix=this_predictor_matrix,
                grid_latitudes_deg_n=stx[shapley_io.LATITUDE_KEY].values,
                grid_longitudes_deg_e=stx[shapley_io.LONGITUDE_KEY].values,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                field_name=all_target_field_names[f],
                title_string=title_string
            )

            min_colour_value, max_colour_value = _plot_one_shapley_field(
                shapley_matrix=
                stx[shapley_io.SHAPLEY_FOR_LAGLEAD_TARGETS_KEY].values[..., t, f],
                axes_object=axes_object,
                figure_object=figure_object,
                grid_latitudes_deg_n=stx[shapley_io.LATITUDE_KEY].values,
                grid_longitudes_deg_e=stx[shapley_io.LONGITUDE_KEY].values,
                min_colour_percentile=shapley_colour_limits_prctile[0],
                max_colour_percentile=shapley_colour_limits_prctile[1],
                half_num_contours=shapley_half_num_contours,
                colour_map_object=SHAPLEY_FWI_COLOUR_MAP_OBJECT,
                line_width=shapley_line_width,
                opacity=shapley_line_opacity,
                plot_in_log_space=plot_shapley_in_log_space,
                output_file_name=output_file_name
            )

            colour_norm_object = pyplot.Normalize(
                vmin=min_colour_value, vmax=max_colour_value
            )
            plotting_utils.add_colour_bar(
                figure_file_name=output_file_name,
                colour_map_object=SHAPLEY_FWI_COLOUR_MAP_OBJECT,
                colour_norm_object=colour_norm_object,
                orientation_string='vertical',
                font_size=COLOUR_BAR_FONT_SIZE,
                cbar_label_string='Absolute Shapley value',
                tick_label_format_string='{0:.2g}',
                log_space=plot_shapley_in_log_space
            )

    era5_constant_field_names = vod[
        neural_net.ERA5_CONSTANT_PREDICTOR_FIELDS_KEY
    ]

    if era5_constant_field_names is None:
        era5_constant_predictor_matrix = numpy.array([])
    else:
        lyr_idx = model_input_layer_names.index(neural_net.ERA5_LAYER_NAME)
        era5_constant_predictor_matrix = predictor_matrices[lyr_idx][0, ...]

    for f in range(len(era5_constant_field_names)):
        this_predictor_matrix = era5_constant_predictor_matrix[..., f]

        unit_string = era5_constant_utils.FIELD_NAME_TO_UNIT_STRING[
            era5_constant_field_names[f]
        ]
        if unit_string != '':
            unit_string = '({0:s})'.format(unit_string)

        title_string = (
            'ERA5 {0:s}{1:s}\n'
            'Shapley values for {2:d}-day {3:s}{4:s}'
        ).format(
            era5_constant_utils.FIELD_NAME_TO_FANCY[
                era5_constant_field_names[f]
            ],
            unit_string,
            model_lead_time_days,
            fwi_plotting.FIELD_NAME_TO_SIMPLE[target_field_name],
            '' if region_name is None else ' over ' + region_name
        )

        output_file_name = '{0:s}/shapley_era5_{1:s}.jpg'.format(
            output_dir_name,
            era5_constant_field_names[f].replace('_', '-')
        )

        figure_object, axes_object = _plot_one_era5_field(
            data_matrix=this_predictor_matrix,
            grid_latitudes_deg_n=stx[shapley_io.LATITUDE_KEY].values,
            grid_longitudes_deg_e=stx[shapley_io.LONGITUDE_KEY].values,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            min_colour_percentile=predictor_colour_limits_prctile[0],
            max_colour_percentile=predictor_colour_limits_prctile[1],
            title_string=title_string
        )

        min_colour_value, max_colour_value = _plot_one_shapley_field(
            shapley_matrix=stx[shapley_io.SHAPLEY_FOR_ERA5_KEY].values[..., f],
            axes_object=axes_object,
            figure_object=figure_object,
            grid_latitudes_deg_n=stx[shapley_io.LATITUDE_KEY].values,
            grid_longitudes_deg_e=stx[shapley_io.LONGITUDE_KEY].values,
            min_colour_percentile=shapley_colour_limits_prctile[0],
            max_colour_percentile=shapley_colour_limits_prctile[1],
            half_num_contours=shapley_half_num_contours,
            colour_map_object=SHAPLEY_DEFAULT_COLOUR_MAP_OBJECT,
            line_width=shapley_line_width,
            opacity=shapley_line_opacity,
            plot_in_log_space=plot_shapley_in_log_space,
            output_file_name=output_file_name
        )

        colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )
        plotting_utils.add_colour_bar(
            figure_file_name=output_file_name,
            colour_map_object=SHAPLEY_DEFAULT_COLOUR_MAP_OBJECT,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            font_size=COLOUR_BAR_FONT_SIZE,
            cbar_label_string='Absolute Shapley value',
            tick_label_format_string='{0:.2g}',
            log_space=plot_shapley_in_log_space
        )

    do_residual_prediction = vod[neural_net.DO_RESIDUAL_PREDICTION_KEY]

    if do_residual_prediction:
        lyr_idx = model_input_layer_names.index(
            neural_net.PREDN_BASELINE_LAYER_NAME
        )
        resid_baseline_predictor_matrix = predictor_matrices[lyr_idx][0, ...]
    else:
        resid_baseline_predictor_matrix = numpy.array([])

    for f in range(len(all_target_field_names)):
        this_predictor_matrix = resid_baseline_predictor_matrix[..., f]

        title_string = (
            'Baseline {0:s}, {1:s} - {2:d} days\n'
            'Shapley values for {3:d}-day {4:s}{5:s}'
        ).format(
            fwi_plotting.FIELD_NAME_TO_SIMPLE[all_target_field_names[f]],
            init_date_string_nice,
            numpy.min(target_lag_times_days),
            model_lead_time_days,
            fwi_plotting.FIELD_NAME_TO_SIMPLE[target_field_name],
            '' if region_name is None else ' over ' + region_name
        )

        output_file_name = '{0:s}/shapley_residual_baseline_{1:s}.jpg'.format(
            output_dir_name,
            all_target_field_names[f].replace('_', '-')
        )

        figure_object, axes_object = _plot_one_fwi_field(
            data_matrix=this_predictor_matrix,
            grid_latitudes_deg_n=stx[shapley_io.LATITUDE_KEY].values,
            grid_longitudes_deg_e=stx[shapley_io.LONGITUDE_KEY].values,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            field_name=all_target_field_names[f],
            title_string=title_string
        )

        min_colour_value, max_colour_value = _plot_one_shapley_field(
            shapley_matrix=
            stx[shapley_io.SHAPLEY_FOR_PREDN_BASELINE_KEY].values[..., f],
            axes_object=axes_object,
            figure_object=figure_object,
            grid_latitudes_deg_n=stx[shapley_io.LATITUDE_KEY].values,
            grid_longitudes_deg_e=stx[shapley_io.LONGITUDE_KEY].values,
            min_colour_percentile=shapley_colour_limits_prctile[0],
            max_colour_percentile=shapley_colour_limits_prctile[1],
            half_num_contours=shapley_half_num_contours,
            colour_map_object=SHAPLEY_FWI_COLOUR_MAP_OBJECT,
            line_width=shapley_line_width,
            opacity=shapley_line_opacity,
            plot_in_log_space=plot_shapley_in_log_space,
            output_file_name=output_file_name
        )

        colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )
        plotting_utils.add_colour_bar(
            figure_file_name=output_file_name,
            colour_map_object=SHAPLEY_FWI_COLOUR_MAP_OBJECT,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            font_size=COLOUR_BAR_FONT_SIZE,
            cbar_label_string='Absolute Shapley value',
            tick_label_format_string='{0:.2g}',
            log_space=plot_shapley_in_log_space
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        shapley_file_name=getattr(INPUT_ARG_OBJECT, SHAPLEY_FILE_ARG_NAME),
        gfs_directory_name=getattr(INPUT_ARG_OBJECT, GFS_DIRECTORY_ARG_NAME),
        target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        gfs_forecast_target_dir_name=getattr(
            INPUT_ARG_OBJECT, GFS_FCST_TARGET_DIR_ARG_NAME
        ),
        gfs_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, GFS_NORM_FILE_ARG_NAME
        ),
        predictor_colour_limits_prctile=numpy.array(
            getattr(INPUT_ARG_OBJECT, PREDICTOR_COLOUR_LIMITS_ARG_NAME),
            dtype=float
        ),
        shapley_colour_limits_prctile=numpy.array(
            getattr(INPUT_ARG_OBJECT, SHAPLEY_COLOUR_LIMITS_ARG_NAME),
            dtype=float
        ),
        region_name=getattr(INPUT_ARG_OBJECT, REGION_ARG_NAME),
        shapley_line_width=getattr(
            INPUT_ARG_OBJECT, SHAPLEY_LINE_WIDTH_ARG_NAME
        ),
        shapley_line_opacity=getattr(
            INPUT_ARG_OBJECT, SHAPLEY_LINE_OPACITY_ARG_NAME
        ),
        plot_latitude_limits_deg_n=numpy.array(
            getattr(INPUT_ARG_OBJECT, PLOT_LATITUDE_LIMITS_ARG_NAME),
            dtype=float
        ),
        plot_longitude_limits_deg_e=numpy.array(
            getattr(INPUT_ARG_OBJECT, PLOT_LONGITUDE_LIMITS_ARG_NAME),
            dtype=float
        ),
        shapley_half_num_contours=getattr(
            INPUT_ARG_OBJECT, SHAPLEY_HALF_NUM_CONTOURS_ARG_NAME
        ),
        plot_shapley_in_log_space=bool(getattr(
            INPUT_ARG_OBJECT, SHAPLEY_LOG_SPACE_ARG_NAME
        )),
        shapley_smoothing_radius_px=getattr(
            INPUT_ARG_OBJECT, SHAPLEY_SMOOTHING_RADIUS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
