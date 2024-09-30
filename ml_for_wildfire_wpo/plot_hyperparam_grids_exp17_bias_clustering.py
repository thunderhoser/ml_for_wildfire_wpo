"""Plots hyperparameter grids for bias-clustering extension to Experiment 17.

One hyperparameter grid = one evaluation metric vs. all hyperparams.
"""

import os
import sys
import argparse
import numpy
from scipy.stats import rankdata
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import imagemagick_utils
import file_system_utils
import gg_plotting_utils
import canadian_fwi_utils
import regression_evaluation as regression_eval

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRIC_NAMES = [
    regression_eval.MAE_KEY,
    regression_eval.MSE_KEY,
    regression_eval.BIAS_KEY,
    regression_eval.DWMSE_KEY,
    regression_eval.KGE_KEY,
    regression_eval.RELIABILITY_KEY
]

METRIC_NAMES_SIMPLE = [
    'MAE', 'MSE', 'bias', 'DWMSE', 'KGE', 'REL'
]

TARGET_FIELD_NAMES = [
    canadian_fwi_utils.FWI_NAME,
    canadian_fwi_utils.DSR_NAME,
    canadian_fwi_utils.FFMC_NAME,
    canadian_fwi_utils.DMC_NAME,
    canadian_fwi_utils.DC_NAME,
    canadian_fwi_utils.ISI_NAME,
    canadian_fwi_utils.BUI_NAME
]

BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1 = numpy.array(['tiny', 'small', 'medium', 'large'])
MIN_CLUSTER_SIZES_PX_AXIS2 = numpy.array([2, 5, 10, 20, 50, 100, 200], dtype=int)
BUFFER_DISTANCES_PX_AXIS3 = numpy.array([0, 1, 2], dtype=int)
DO_BACKWARDS_FLAGS_AXIS4 = numpy.array([0, 1], dtype=int)

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.075
WHITE_COLOUR = numpy.full(3, 1.)
BLACK_COLOUR = numpy.full(3, 0.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.075
SELECTED_MARKER_INDICES = numpy.array([0, 0, 0, 0], dtype=int)

MAIN_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis', lut=20)
BIAS_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='seismic', lut=20)
MAIN_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))
BIAS_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))

FONT_SIZE = 26
AXIS_LABEL_FONT_SIZE = 26
TICK_LABEL_FONT_SIZE = 14

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

EXPERIMENT_DIR_ARG_NAME = 'experiment_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXPERIMENT_DIR_HELP_STRING = 'Name of top-level directory with models.'
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _finite_percentile(input_array, percentile_level):
    """Takes percentile of input array, considering only finite values.

    :param input_array: numpy array.
    :param percentile_level: Percentile level, ranging from 0...100.
    :return: output_percentile: Percentile value.
    """

    return numpy.percentile(
        input_array[numpy.isfinite(input_array)], percentile_level
    )


def _plot_scores_2d(
        score_matrix, min_colour_value, max_colour_value, x_tick_labels,
        y_tick_labels):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param x_tick_labels: length-N list of tick labels.
    :param y_tick_labels: length-M list of tick labels.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if min_colour_value is None:
        colour_map_object = BIAS_COLOUR_MAP_OBJECT
        min_colour_value = -1 * max_colour_value
    else:
        colour_map_object = MAIN_COLOUR_MAP_OBJECT

    axes_object.imshow(
        score_matrix, cmap=colour_map_object, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value
    )

    x_tick_values = numpy.linspace(
        0, score_matrix.shape[1] - 1, num=score_matrix.shape[1], dtype=float
    )
    y_tick_values = numpy.linspace(
        0, score_matrix.shape[0] - 1, num=score_matrix.shape[0], dtype=float
    )

    pyplot.xticks(
        x_tick_values, x_tick_labels,
        rotation=90., fontsize=TICK_LABEL_FONT_SIZE
    )
    pyplot.yticks(y_tick_values, y_tick_labels, fontsize=TICK_LABEL_FONT_SIZE)

    colour_norm_object = matplotlib.colors.Normalize(
        vmin=min_colour_value, vmax=max_colour_value, clip=False
    )

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=score_matrix[numpy.invert(numpy.isnan(score_matrix))],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        font_size=FONT_SIZE, fraction_of_axis_length=0.6
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.

    I = number of bias-discretization interval intervals
    M = number of minimum cluster sizes
    D = number of buffer distances
    B = number of backwards flags

    :param score_matrix: I-by-M-by-D-by-B numpy array of score values.
    :param score_name: Name of score.
    """

    scores_1d = numpy.ravel(score_matrix)
    scores_1d[numpy.isnan(scores_1d)] = numpy.inf
    sort_indices_1d = numpy.argsort(scores_1d)
    i_sort_indices, m_sort_indices, d_sort_indices, b_sort_indices = (
        numpy.unravel_index(sort_indices_1d, score_matrix.shape)
    )

    for k in range(len(i_sort_indices)):
        i = i_sort_indices[k]
        m = m_sort_indices[k]
        d = d_sort_indices[k]
        b = b_sort_indices[k]

        print((
            r'{0:d}th-lowest {1:s} = {2:.4g} ... '
            r'discretization $\Delta\Delta$ = {3:s} ... '
            r'min size = {4:d} ... '
            r'buffer dist = {5:d} px ... '
            r'backwards = {6:d}'
        ).format(
            k + 1, score_name, score_matrix[i, m, d, b],
            BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1[i],
            MIN_CLUSTER_SIZES_PX_AXIS2[m],
            BUFFER_DISTANCES_PX_AXIS3[d],
            DO_BACKWARDS_FLAGS_AXIS4[b]
        ))


def _print_ranking_all_scores(metric_matrix):
    """Prints ranking for all scores.

    I = number of bias-discretization interval intervals
    M = number of minimum cluster sizes
    D = number of buffer distances
    B = number of backwards flags
    F = number of target fields
    X = number of metrics

    :param metric_matrix: I-by-M-by-D-by-B-by-F-by-X numpy array of scores.
    """

    rank_matrix = numpy.full(metric_matrix.shape, numpy.nan)

    for f in range(len(TARGET_FIELD_NAMES)):
        for x in range(len(METRIC_NAMES)):
            if METRIC_NAMES[x] == regression_eval.KGE_KEY:
                these_scores = numpy.ravel(-1 * metric_matrix[..., f, x])
            elif METRIC_NAMES[x] == regression_eval.BIAS_KEY:
                these_scores = numpy.ravel(numpy.absolute(metric_matrix[..., f, x]))
            else:
                these_scores = numpy.ravel(metric_matrix[..., f, x])

            these_scores[numpy.isnan(these_scores)] = numpy.inf
            rank_matrix[..., f, x] = numpy.reshape(
                rankdata(these_scores, method='average'),
                metric_matrix[..., f, x].shape
            )

    overall_rank_matrix = numpy.mean(rank_matrix, axis=(-1, -2))

    sort_indices_1d = numpy.argsort(numpy.ravel(overall_rank_matrix))
    i_sort_indices, m_sort_indices, d_sort_indices, b_sort_indices = (
        numpy.unravel_index(sort_indices_1d, overall_rank_matrix.shape)
    )

    for k in range(len(i_sort_indices)):
        i = i_sort_indices[k]
        m = m_sort_indices[k]
        d = d_sort_indices[k]
        b = b_sort_indices[k]

        print((
            r'\n{0:d}th overall ... discretization $\Delta\Delta$ = {1:s} ... '
            r'min size = {2:d} ... '
            r'buffer dist = {3:d} px ... '
            r'backwards = {4:d}:'
        ).format(
            k + 1,
            BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1[i],
            MIN_CLUSTER_SIZES_PX_AXIS2[m],
            BUFFER_DISTANCES_PX_AXIS3[d],
            DO_BACKWARDS_FLAGS_AXIS4[b]
        ))

        for f in range(len(TARGET_FIELD_NAMES)):
            print((
                'MAE/MSE/bias/DWMSE/KGE/REL ranks for {0:s} = '
                '{1:.1f}, {2:.1f}, {3:.1f}, {4:.1f}, {5:.1f}, {6:.1f}'
            ).format(
                TARGET_FIELD_NAMES[f],
                rank_matrix[..., f, 0],
                rank_matrix[..., f, 1],
                rank_matrix[..., f, 2],
                rank_matrix[..., f, 3],
                rank_matrix[..., f, 4],
                rank_matrix[..., f, 5]
            ))


def _run(experiment_dir_name, output_dir_name):
    """Plots hyperparameter grids for bias-clustering extension to Experiment 17.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of this script.
    :param output_dir_name: Same.
    """

    axis1_length = len(BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1)
    axis2_length = len(MIN_CLUSTER_SIZES_PX_AXIS2)
    axis3_length = len(BUFFER_DISTANCES_PX_AXIS3)
    axis4_length = len(DO_BACKWARDS_FLAGS_AXIS4)

    y_tick_labels = [
        '{0:s}'.format(b)
        for b in BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1
    ]
    y_tick_labels = [l[0].upper() + l[1:] for l in y_tick_labels]
    x_tick_labels = ['{0:d}'.format(m) for m in MIN_CLUSTER_SIZES_PX_AXIS2]

    y_axis_label = r'Bias-discretization $\Delta\Delta$ ($^{\circ}$C)'
    x_axis_label = 'Min cluster size (pixels)'

    num_metrics = len(METRIC_NAMES)
    num_target_fields = len(TARGET_FIELD_NAMES)
    dimensions = (
        axis1_length, axis2_length, axis3_length, axis4_length,
        num_target_fields, num_metrics
    )

    metric_matrix = numpy.full(dimensions, numpy.nan)

    for i in range(axis1_length):
        for j in range(axis2_length):
            for k in range(axis3_length):
                for m in range(axis4_length):
                    this_eval_file_name = (
                        '{0:s}/validation_lead-time-days=02/isotonic_regression/'
                        'bias-discretization-interval-interval={1:s}_'
                        'min-cluster-size-px={2:03d}_buffer-distance-px={3:d}_'
                        'do-backwards-clustering={4:d}/ungridded_evaluation.nc'
                    ).format(
                        experiment_dir_name,
                        BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1[i],
                        MIN_CLUSTER_SIZES_PX_AXIS2[j],
                        BUFFER_DISTANCES_PX_AXIS3[k],
                        DO_BACKWARDS_FLAGS_AXIS4[m]
                    )

                    if not os.path.isfile(this_eval_file_name):
                        continue

                    print('Reading data from: "{0:s}"...'.format(
                        this_eval_file_name
                    ))
                    this_eval_table_xarray = regression_eval.read_file(
                        this_eval_file_name
                    )
                    etx = this_eval_table_xarray

                    for f in range(num_target_fields):
                        for x in range(num_metrics):
                            f_other = numpy.where(
                                etx.coords[regression_eval.FIELD_DIM].values
                                == TARGET_FIELD_NAMES[f]
                            )[0][0]

                            metric_matrix[i, j, k, m, f, x] = numpy.mean(
                                etx[METRIC_NAMES[x]].values[f_other, :]
                            )

    print(SEPARATOR_STRING)

    for f in range(num_target_fields):
        for x in range(num_metrics):
            if METRIC_NAMES[x] == regression_eval.BIAS_KEY:
                _print_ranking_one_score(
                    score_matrix=numpy.absolute(metric_matrix[..., f, x]),
                    score_name='absolute bias for {0:s}'.format(
                        TARGET_FIELD_NAMES[f]
                    )
                )
            elif METRIC_NAMES[x] == regression_eval.KGE_KEY:
                _print_ranking_one_score(
                    score_matrix=-1 * metric_matrix[..., f, x],
                    score_name='negative KGE for {0:s}'.format(
                        TARGET_FIELD_NAMES[f]
                    )
                )
            else:
                _print_ranking_one_score(
                    score_matrix=metric_matrix[..., f, x],
                    score_name='{0:s} for {1:s}'.format(
                        METRIC_NAMES_SIMPLE[x], TARGET_FIELD_NAMES[f]
                    )
                )

            print(SEPARATOR_STRING)

    _print_ranking_all_scores(metric_matrix)
    print(SEPARATOR_STRING)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    dimensions = (axis3_length, axis4_length, num_target_fields, num_metrics)
    panel_file_name_matrix = numpy.full(dimensions, '', dtype=object)

    for k in range(axis3_length):
        for m in range(axis4_length):
            for f in range(num_target_fields):
                for x in range(num_metrics):
                    if METRIC_NAMES[x] == regression_eval.KGE_KEY:
                        min_colour_value = _finite_percentile(
                            metric_matrix[..., f, x], 5
                        )
                        max_colour_value = _finite_percentile(
                            metric_matrix[..., f, x], 100
                        )
                        this_index = numpy.nanargmax(numpy.ravel(
                            metric_matrix[..., f, x]
                        ))
                        marker_colour = BLACK_COLOUR
                    elif METRIC_NAMES[x] == regression_eval.BIAS_KEY:
                        min_colour_value = None
                        max_colour_value = _finite_percentile(
                            numpy.absolute(metric_matrix[..., f, x]), 95
                        )
                        this_index = numpy.nanargmin(numpy.ravel(
                            numpy.absolute(metric_matrix[..., f, x])
                        ))
                        marker_colour = BLACK_COLOUR
                    else:
                        min_colour_value = _finite_percentile(
                            metric_matrix[..., f, x], 0
                        )
                        max_colour_value = _finite_percentile(
                            metric_matrix[..., f, x], 95
                        )
                        this_index = numpy.nanargmin(numpy.ravel(
                            metric_matrix[..., f, x]
                        ))
                        marker_colour = WHITE_COLOUR

                    figure_object, axes_object = _plot_scores_2d(
                        score_matrix=metric_matrix[..., k, m, f, x],
                        min_colour_value=min_colour_value,
                        max_colour_value=max_colour_value,
                        x_tick_labels=x_tick_labels,
                        y_tick_labels=y_tick_labels
                    )

                    best_indices = numpy.unravel_index(
                        this_index, metric_matrix[..., f, x].shape
                    )

                    figure_width_px = (
                        figure_object.get_size_inches()[0] * figure_object.dpi
                    )
                    marker_size_px = figure_width_px * (
                        BEST_MARKER_SIZE_GRID_CELLS / axis2_length
                    )

                    if best_indices[2] == k and best_indices[3] == m:
                        axes_object.plot(
                            best_indices[1], best_indices[0],
                            linestyle='None', marker=BEST_MARKER_TYPE,
                            markersize=marker_size_px, markeredgewidth=0,
                            markerfacecolor=marker_colour,
                            markeredgecolor=marker_colour
                        )

                    if (
                            SELECTED_MARKER_INDICES[2] == k and
                            SELECTED_MARKER_INDICES[3] == m
                    ):
                        axes_object.plot(
                            SELECTED_MARKER_INDICES[1],
                            SELECTED_MARKER_INDICES[0],
                            linestyle='None', marker=SELECTED_MARKER_TYPE,
                            markersize=marker_size_px, markeredgewidth=0,
                            markerfacecolor=marker_colour,
                            markeredgecolor=marker_colour
                        )

                    axes_object.set_xlabel(
                        x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE
                    )
                    axes_object.set_ylabel(
                        y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE
                    )

                    title_string = '{0:s}{1:s} for {2:s}'.format(
                        METRIC_NAMES_SIMPLE[x][0].upper(),
                        METRIC_NAMES_SIMPLE[x][1:],
                        TARGET_FIELD_NAMES[f]
                    )
                    title_string += (
                        '\n{0:s} clustering with buffer dist = {1:d} px'
                    ).format(
                        'BACKWARDS' if DO_BACKWARDS_FLAGS_AXIS4[m]
                        else 'FORWARD',
                        BUFFER_DISTANCES_PX_AXIS3[k]
                    )
                    axes_object.set_title(title_string)

                    panel_file_name_matrix[k, m, f, x] = (
                        '{0:s}/{1:s}_{2:s}_buffer-distance-px={3:d}_'
                        'do-backwards-clustering={4:d}.jpg'
                    ).format(
                        output_dir_name,
                        TARGET_FIELD_NAMES[f].replace('_', '-'),
                        METRIC_NAMES_SIMPLE[x].lower().replace('_', '-'),
                        BUFFER_DISTANCES_PX_AXIS3[k],
                        DO_BACKWARDS_FLAGS_AXIS4[m]
                    )

                    print('Saving figure to: "{0:s}"...'.format(
                        panel_file_name_matrix[k, m, f, x]
                    ))
                    figure_object.savefig(
                        panel_file_name_matrix[k, m, f, x],
                        dpi=FIGURE_RESOLUTION_DPI,
                        pad_inches=0,
                        bbox_inches='tight'
                    )
                    pyplot.close(figure_object)

                    print('Resizing panel: "{0:s}"...'.format(
                        panel_file_name_matrix[k, m, f, x]
                    ))
                    imagemagick_utils.resize_image(
                        input_file_name=panel_file_name_matrix[k, m, f, x],
                        output_file_name=panel_file_name_matrix[k, m, f, x],
                        output_size_pixels=int(2.5e6)
                    )

    for f in range(num_target_fields):
        for x in range(num_metrics):
            concat_figure_file_name = '{0:s}/{1:s}_{2:s}.jpg'.format(
                output_dir_name,
                TARGET_FIELD_NAMES[f].replace('_', '-'),
                METRIC_NAMES_SIMPLE[x].lower().replace('_', '-')
            )

            print('Concatenating panels to: "{0:s}"...'.format(
                concat_figure_file_name
            ))
            imagemagick_utils.concatenate_images(
                input_file_names=
                numpy.ravel(panel_file_name_matrix[..., f, x]).tolist(),
                output_file_name=concat_figure_file_name,
                num_panel_rows=axis3_length,
                num_panel_columns=axis4_length
            )
            imagemagick_utils.resize_image(
                input_file_name=concat_figure_file_name,
                output_file_name=concat_figure_file_name,
                output_size_pixels=int(1e7)
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
