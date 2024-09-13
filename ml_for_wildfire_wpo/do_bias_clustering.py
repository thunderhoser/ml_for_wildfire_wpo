"""Clusters a spatial field of model biases."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import longitude_conversion as lng_conversion
import file_system_utils
import border_io
import regression_evaluation as regression_eval
import bias_clustering
import plotting_utils
import gfs_plotting

# TODO(thunderhoser): Currently, one gridded-evaluation file must correspond to
# one lead time, which means that each bias-clustering must be based on one
# field at one lead time.  However, I might want the option of one field at all
# lead times.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

INPUT_FILE_ARG_NAME = 'input_gridded_eval_file_arg_name'
MIN_CLUSTER_SIZE_ARG_NAME = 'min_cluster_size_px'
TARGET_FIELD_ARG_NAME = 'target_field_name'
BIAS_DISCRETIZATION_INTERVALS_ARG_NAME = 'bias_discretization_intervals'
BUFFER_DISTANCE_ARG_NAME = 'buffer_distance_px'
DO_BACKWARDS_ARG_NAME = 'do_backwards_clustering'
CLUSTER_SSDIFF_ARG_NAME = 'cluster_ssdiff_instead_of_bias'
MAIN_OUTPUT_FILE_ARG_NAME = 'main_output_file_name'
OUTPUT_FIGURE_FILE_ARG_NAME = 'output_figure_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing gridded evaluation for one model.  Will be '
    'read by `regression_evaluation.read_file`.'
)
MIN_CLUSTER_SIZE_HELP_STRING = 'Minimum cluster size (number of pixels).'
TARGET_FIELD_HELP_STRING = (
    'Name of target field (must be accepted by `canadian_fwi_utils.check_field_name`).'
    '  Will cluster biases only for this field.'
)
BIAS_DISCRETIZATION_INTERVALS_HELP_STRING = (
    'List of bias-discretization intervals for the given target field.'
)
BUFFER_DISTANCE_HELP_STRING = (
    'Buffer distance (number of pixels), used to allow for "gaps" within a '
    'cluster.  If you do not want to allow gaps (i.e., if you want every '
    'cluster to be spatially contiguous), make this 0.'
)
DO_BACKWARDS_HELP_STRING = (
    'Boolean flag.  If 1, will run the backwards clustering algorithm, '
    'starting with the largest spatial scale.  If 0, will run the forward '
    'algorithm, starting with the smallest scale.'
)
CLUSTER_SSDIFF_HELP_STRING = (
    'Boolean flag.  If 1, will cluster spread-skill difference instead of bias.'
)
MAIN_OUTPUT_FILE_HELP_STRING = (
    'Path to main output file.  The clustering will be written here by '
    '`bias_clustering.write_file`.'
)
OUTPUT_FIGURE_FILE_HELP_STRING = (
    'Path to figure file.  The clustering will be plotted and then written '
    'here as a JPEG.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_CLUSTER_SIZE_ARG_NAME, type=int, required=True,
    help=MIN_CLUSTER_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELD_ARG_NAME, type=str, required=True,
    help=TARGET_FIELD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BIAS_DISCRETIZATION_INTERVALS_ARG_NAME, type=float, nargs='+',
    required=True, help=BIAS_DISCRETIZATION_INTERVALS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BUFFER_DISTANCE_ARG_NAME, type=float, required=True,
    help=BUFFER_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DO_BACKWARDS_ARG_NAME, type=int, required=True,
    help=DO_BACKWARDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CLUSTER_SSDIFF_ARG_NAME, type=int, required=True,
    help=CLUSTER_SSDIFF_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAIN_OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=MAIN_OUTPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FIGURE_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FIGURE_FILE_HELP_STRING
)


def _run(gridded_eval_file_arg_name, min_cluster_size_px, target_field_name,
         bias_discretization_intervals, buffer_distance_px,
         do_backwards_clustering, cluster_ssdiff_instead_of_bias,
         main_output_file_name, figure_file_name):
    """Clusters a spatial field of model biases.

    This is effectively the main method.

    :param gridded_eval_file_arg_name: See documentation at top of this script.
    :param min_cluster_size_px: Same.
    :param target_field_name: Same.
    :param bias_discretization_intervals: Same.
    :param buffer_distance_px: Same.
    :param do_backwards_clustering: Same.
    :param cluster_ssdiff_instead_of_bias: Same.
    :param main_output_file_name: Same.
    :param figure_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=figure_file_name)

    print('Reading data from: "{0:s}"...'.format(gridded_eval_file_arg_name))
    gridded_eval_table_xarray = regression_eval.read_file(
        gridded_eval_file_arg_name
    )
    getx = gridded_eval_table_xarray

    field_index = numpy.where(
        getx.coords[regression_eval.FIELD_DIM].values == target_field_name
    )[0][0]

    if cluster_ssdiff_instead_of_bias:
        bias_matrix = numpy.nanmean(
            getx[regression_eval.SSDIFF_KEY].values[..., field_index, :],
            axis=-1
        )
    else:
        bias_matrix = numpy.nanmean(
            getx[regression_eval.BIAS_KEY].values[..., field_index, :], axis=-1
        )

    if do_backwards_clustering:
        cluster_id_matrix = bias_clustering.find_clusters_one_field_backwards(
            bias_matrix=bias_matrix,
            min_cluster_size=min_cluster_size_px,
            bias_discretization_intervals=bias_discretization_intervals,
            buffer_distance_px=buffer_distance_px
        )
    else:
        cluster_id_matrix = bias_clustering.find_clusters_one_field(
            bias_matrix=bias_matrix,
            min_cluster_size=min_cluster_size_px,
            bias_discretization_intervals=bias_discretization_intervals,
            buffer_distance_px=buffer_distance_px
        )

    print(SEPARATOR_STRING)
    print('Writing results to file: "{0:s}"...'.format(main_output_file_name))

    grid_latitudes_deg_n = getx.coords[regression_eval.LATITUDE_DIM].values
    grid_longitudes_deg_e = getx.coords[regression_eval.LONGITUDE_DIM].values
    bias_clustering.write_file(
        netcdf_file_name=main_output_file_name,
        cluster_id_matrix=numpy.expand_dims(cluster_id_matrix, axis=-1),
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        field_names=[target_field_name]
    )

    unique_cluster_ids = numpy.unique(cluster_id_matrix)
    random_colours = numpy.random.rand(len(unique_cluster_ids), 3)
    colour_map_object = matplotlib.colors.ListedColormap(random_colours)
    colour_norm_object = matplotlib.colors.Normalize(
        vmin=numpy.min(cluster_id_matrix),
        vmax=numpy.max(cluster_id_matrix)
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    is_longitude_positive_in_west = gfs_plotting.plot_field(
        data_matrix=cluster_id_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        axes_object=axes_object, plot_colour_bar=True
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

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

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        gridded_eval_file_arg_name=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME
        ),
        min_cluster_size_px=getattr(
            INPUT_ARG_OBJECT, MIN_CLUSTER_SIZE_ARG_NAME
        ),
        target_field_name=getattr(INPUT_ARG_OBJECT, TARGET_FIELD_ARG_NAME),
        bias_discretization_intervals=numpy.array(
            getattr(INPUT_ARG_OBJECT, BIAS_DISCRETIZATION_INTERVALS_ARG_NAME),
            dtype=float
        ),
        buffer_distance_px=getattr(INPUT_ARG_OBJECT, BUFFER_DISTANCE_ARG_NAME),
        do_backwards_clustering=bool(
            getattr(INPUT_ARG_OBJECT, DO_BACKWARDS_ARG_NAME)
        ),
        cluster_ssdiff_instead_of_bias=bool(
            getattr(INPUT_ARG_OBJECT, CLUSTER_SSDIFF_ARG_NAME)
        ),
        main_output_file_name=getattr(
            INPUT_ARG_OBJECT, MAIN_OUTPUT_FILE_ARG_NAME
        ),
        figure_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FIGURE_FILE_ARG_NAME)
    )
