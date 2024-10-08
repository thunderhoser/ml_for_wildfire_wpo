"""Plots Shapley values as spatial maps."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import shapley_io
import border_io
import gfs_utils
import neural_net
import plotting_utils
import gfs_plotting
import fwi_plotting

TOLERANCE = 1e-6

DATE_FORMAT = '%Y%m%d'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

GFS_SEQUENTIAL_COLOUR_MAP_OBJECT = pyplot.get_cmap('viridis')
GFS_DIVERGING_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

COLOUR_BAR_FONT_SIZE = 12
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

SHAPLEY_FILE_ARG_NAME = 'input_shapley_file_name'
GFS_DIRECTORY_ARG_NAME = 'input_gfs_directory_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
GFS_FCST_TARGET_DIR_ARG_NAME = 'input_gfs_fcst_target_dir_name'
PREDICTOR_COLOUR_LIMITS_ARG_NAME = 'predictor_colour_limits_prctile'
SHAPLEY_COLOUR_LIMITS_ARG_NAME = 'shapley_colour_limits_prctile'
REGION_ARG_NAME = 'region_name'
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
PREDICTOR_COLOUR_LIMITS_HELP_STRING = (
    'Colour limits for predictor values, in terms of percentile (ranging from '
    '0...100).  For each 2-D predictor slice, these percentiles will be '
    'recomputed and will determine the limits of the colour bar.'
)
SHAPLEY_COLOUR_LIMITS_HELP_STRING = (
    'Same as `{0:s}` but for Shapley values.'
).format(
    PREDICTOR_COLOUR_LIMITS_ARG_NAME
)
REGION_HELP_STRING = (
    'Name of region for which Shapley values will be computed.  If you leave '
    'this empty, the region will not be reported in figure titles.'
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_shapley_field(
        shapley_matrix, axes_object, figure_object, grid_latitudes_deg_n,
        grid_longitudes_deg_e, min_colour_percentile, max_colour_percentile,
        half_num_contours, colour_map_object, line_width,
        output_file_name, plot_in_log_space=False):
    """Plots 2-D field of Shapley values as a spatial map.

    M = number of rows in grid
    N = number of columns in grid

    :param shapley_matrix: M-by-N numpy array of Shapley values.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :param figure_object: BLAHH.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param min_colour_percentile: Determines minimum absolute Shapley value to
        plot.
    :param max_colour_percentile: Determines max absolute Shapley value to plot.
    :param half_num_contours: Number of contours on either side of zero.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param line_width: Width of contour lines.
    :param output_file_name: Path to output file.
    :param plot_in_log_space: Boolean flag.  If True (False), colour scale will
        be logarithmic in base 10 (linear).
    """

    # TODO: doc for figure_object
    # TODO: need output doc

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

    min_abs_contour_value = numpy.percentile(
        numpy.absolute(shapley_matrix), min_colour_percentile
    )
    max_abs_contour_value = numpy.percentile(
        numpy.absolute(shapley_matrix), max_colour_percentile
    )

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
        linewidths=line_width, linestyles='solid', zorder=1e6
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
        linewidths=line_width, linestyles='dotted', zorder=1e6
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return min_abs_contour_value, max_abs_contour_value


def _plot_one_gfs_field(
        data_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e,
        field_name, min_colour_percentile, max_colour_percentile,
        title_string, output_file_name):
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
    :param output_file_name: Path to output file.
    """

    # TODO: update output documentation.

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if gfs_plotting.use_diverging_colour_scheme(field_name):
        colour_map_object = GFS_DIVERGING_COLOUR_MAP_OBJECT
        max_colour_value = numpy.nanpercentile(
            numpy.absolute(data_matrix), max_colour_percentile
        )
        if numpy.isnan(max_colour_value):
            max_colour_value = TOLERANCE

        max_colour_value = max([max_colour_value, TOLERANCE])
        min_colour_value = -1 * max_colour_value
    else:
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
        line_colour=numpy.full(3, 0.)
    )
    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=grid_latitudes_deg_n,
        plot_longitudes_deg_e=grid_longitudes_deg_e,
        axes_object=axes_object,
        meridian_spacing_deg=20.,
        parallel_spacing_deg=10.
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
         gfs_forecast_target_dir_name, predictor_colour_limits_prctile,
         shapley_colour_limits_prctile, region_name, output_dir_name):
    """Plots Shapley values as spatial maps.

    This is effectively the main method.

    :param shapley_file_name: See documentation at top of this script.
    :param gfs_directory_name: Same.
    :param target_dir_name: Same.
    :param gfs_forecast_target_dir_name: Same.
    :param predictor_colour_limits_prctile: Same.
    :param shapley_colour_limits_prctile: Same.
    :param region_name: Same.
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

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read Shapley values.
    print('Reading Shapley values from: "{0:s}"...'.format(shapley_file_name))
    shapley_table_xarray = shapley_io.read_file(shapley_file_name)
    stx = shapley_table_xarray

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
    validation_option_dict = vod

    init_date_string = stx.attrs[shapley_io.INIT_DATE_KEY]
    init_date_string_nice = (
        init_date_string[:4] +
        '-' + init_date_string[4:6] +
        '-' + init_date_string[6:]
    )
    model_lead_time_days = stx.attrs[shapley_io.MODEL_LEAD_TIME_KEY]
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

    # TODO
    # model_input_layer_names = data_dict[neural_net.INPUT_LAYER_NAMES_KEY]
    model_input_layer_names = [shapley_io.GFS_3D_LAYER_NAME, shapley_io.GFS_2D_LAYER_NAME]
    del data_dict

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    lyr_idx = model_input_layer_names.index(shapley_io.GFS_2D_LAYER_NAME)
    this_predictor_matrix = predictor_matrices[lyr_idx][0, ...]
    gfs_lead_times_hours = vod[neural_net.MODEL_LEAD_TO_GFS_PRED_LEADS_KEY][
        model_lead_time_days
    ]
    gfs_field_names = vod[neural_net.GFS_PREDICTOR_FIELDS_KEY]

    for i in range(len(gfs_lead_times_hours)):
        for j in range(len(gfs_field_names)):
            gfs_matrix_to_plot, unit_string = (
                gfs_plotting.field_to_plotting_units(
                    data_matrix_default_units=this_predictor_matrix[..., i, j],
                    field_name=gfs_field_names[j]
                )
            )

            title_string = (
                'GFS {0:s} ({1:s}), {2:s} + {3:d} hours\n'
                'Shapley values for {4:d}-day {5:s}{6:s}'
            ).format(
                gfs_plotting.FIELD_NAME_TO_FANCY[gfs_field_names[j]],
                unit_string,
                init_date_string_nice,
                gfs_lead_times_hours[i],
                model_lead_time_days,
                fwi_plotting.FIELD_NAME_TO_SIMPLE[target_field_name],
                '' if region_name is None else ' over ' + region_name
            )

            output_file_name = (
                '{0:s}/shapley_gfs_{1:s}_{2:03d}hours.jpg'
            ).format(
                output_dir_name,
                gfs_field_names[j].replace('_', '-'),
                gfs_lead_times_hours[i]
            )

            figure_object, axes_object = _plot_one_gfs_field(
                data_matrix=gfs_matrix_to_plot,
                grid_latitudes_deg_n=stx[shapley_io.LATITUDE_KEY].values,
                grid_longitudes_deg_e=stx[shapley_io.LONGITUDE_KEY].values,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                field_name=gfs_field_names[j],
                min_colour_percentile=predictor_colour_limits_prctile[0],
                max_colour_percentile=predictor_colour_limits_prctile[1],
                title_string=title_string,
                output_file_name=output_file_name
            )

            min_colour_value, max_colour_value = _plot_one_shapley_field(
                shapley_matrix=
                stx[shapley_io.SHAPLEY_FOR_2D_GFS_KEY].values[..., i, j],
                axes_object=axes_object,
                figure_object=figure_object,
                grid_latitudes_deg_n=stx[shapley_io.LATITUDE_KEY].values,
                grid_longitudes_deg_e=stx[shapley_io.LONGITUDE_KEY].values,
                min_colour_percentile=shapley_colour_limits_prctile[0],
                max_colour_percentile=shapley_colour_limits_prctile[1],
                half_num_contours=10,  # TODO: make input arg
                colour_map_object=pyplot.get_cmap('Greys'),  # TODO: make input arg
                line_width=4,  # TODO: make input arg
                plot_in_log_space=False,  # TODO: make input arg
                output_file_name=output_file_name
            )

            colour_norm_object = pyplot.Normalize(
                vmin=min_colour_value, vmax=max_colour_value
            )
            plotting_utils.add_colour_bar(
                figure_file_name=output_file_name,
                colour_map_object=pyplot.get_cmap('Greys'),
                colour_norm_object=colour_norm_object,
                orientation_string='vertical',
                font_size=COLOUR_BAR_FONT_SIZE,
                cbar_label_string='Absolute Shapley value',
                tick_label_format_string='{0:.2g}',
                log_space=False
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
        predictor_colour_limits_prctile=numpy.array(
            getattr(INPUT_ARG_OBJECT, PREDICTOR_COLOUR_LIMITS_ARG_NAME),
            dtype=float
        ),
        shapley_colour_limits_prctile=numpy.array(
            getattr(INPUT_ARG_OBJECT, SHAPLEY_COLOUR_LIMITS_ARG_NAME),
            dtype=float
        ),
        region_name=getattr(INPUT_ARG_OBJECT, REGION_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
