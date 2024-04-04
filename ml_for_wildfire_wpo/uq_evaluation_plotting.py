"""Plotting methods for evaluation of uncertainty quantification (UQ)."""

import os
import sys
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import pit_histogram_utils as pith_utils
import spread_skill_utils as ss_utils
import discard_test_utils as dt_utils
import canadian_fwi_utils

FIELD_NAME_TO_TITLE = {
    canadian_fwi_utils.FFMC_NAME: 'FFMC',
    canadian_fwi_utils.DSR_NAME: 'DSR',
    canadian_fwi_utils.DC_NAME: 'DC',
    canadian_fwi_utils.DMC_NAME: 'DMC',
    canadian_fwi_utils.BUI_NAME: 'BUI',
    canadian_fwi_utils.FWI_NAME: 'FWI',
    canadian_fwi_utils.ISI_NAME: 'ISI'
}

REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
REFERENCE_LINE_WIDTH = 2.

DEFAULT_LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
DEFAULT_LINE_WIDTH = 3.

INSET_HISTO_FACE_COLOUR = numpy.full(3, 152. / 255)
INSET_HISTO_EDGE_COLOUR = numpy.full(3, 0.)
INSET_HISTO_EDGE_WIDTH = 1.

DEFAULT_HISTOGRAM_FACE_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
DEFAULT_HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)
DEFAULT_HISTOGRAM_EDGE_WIDTH = 2.

MEAN_PREDICTION_LINE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
MEAN_PREDICTION_COLOUR_STRING = 'purple'
MEAN_TARGET_LINE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
MEAN_TARGET_COLOUR_STRING = 'green'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

FONT_SIZE = 40
INSET_FONT_SIZE = 20

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _plot_means_as_inset(
        figure_object, bin_centers, bin_mean_predictions,
        bin_mean_target_values, plotting_corner_string, for_spread_skill_plot):
    """Plots means (mean prediction and target by bin) on inset axes.

    B = number of bins

    :param figure_object: Will plot as inset in this figure (instance of
        `matplotlib.figure.Figure`).
    :param bin_centers: length-B numpy array with value at center of each bin.
        These values will be plotted on the x-axis.
    :param bin_mean_predictions: length-B numpy array with mean prediction in
        each bin.  These values will be plotted on the y-axis.
    :param bin_mean_target_values: length-B numpy array with mean target value
        (event frequency) in each bin.  These values will be plotted on the
        y-axis.
    :param plotting_corner_string: String in
        ['top_right', 'top_left', 'bottom_right', 'bottom_left'].
    :param for_spread_skill_plot: Boolean flag.
    :return: inset_axes_object: Axes handle for histogram (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    if plotting_corner_string == 'top_right':
        inset_axes_object = figure_object.add_axes([0.625, 0.55, 0.25, 0.25])
    elif plotting_corner_string == 'bottom_right':
        inset_axes_object = figure_object.add_axes([0.625, 0.3, 0.25, 0.25])
    elif plotting_corner_string == 'bottom_left':
        inset_axes_object = figure_object.add_axes([0.2, 0.3, 0.25, 0.25])
    elif plotting_corner_string == 'top_left':
        inset_axes_object = figure_object.add_axes([0.2, 0.55, 0.25, 0.25])

    nan_flags = numpy.logical_or(
        numpy.isnan(bin_mean_target_values),
        numpy.isnan(bin_mean_predictions)
    )
    assert not numpy.all(nan_flags)
    real_indices = numpy.where(numpy.invert(nan_flags))[0]

    target_handle = inset_axes_object.plot(
        bin_centers[real_indices], bin_mean_target_values[real_indices],
        color=MEAN_TARGET_LINE_COLOUR, linestyle='solid', linewidth=2
    )[0]

    prediction_handle = inset_axes_object.plot(
        bin_centers[real_indices], bin_mean_predictions[real_indices],
        color=MEAN_PREDICTION_LINE_COLOUR, linestyle='dashed', linewidth=2
    )[0]

    y_max = max([
        numpy.nanmax(bin_mean_predictions),
        numpy.nanmax(bin_mean_target_values)
    ])
    y_min = min([
        numpy.nanmin(bin_mean_predictions),
        numpy.nanmin(bin_mean_target_values)
    ])
    inset_axes_object.set_ylim(y_min, y_max)
    inset_axes_object.set_xlim(left=0.)

    inset_axes_object.tick_params(
        axis='x', labelsize=INSET_FONT_SIZE, rotation='vertical'
    )
    inset_axes_object.tick_params(axis='y', labelsize=INSET_FONT_SIZE)

    if for_spread_skill_plot:
        anchor_arg = (0.5, -0.25)
    else:
        anchor_arg = (0.5, -0.2)

    inset_axes_object.legend(
        [target_handle, prediction_handle],
        ['Mean target', 'Mean prediction'],
        loc='upper center', bbox_to_anchor=anchor_arg,
        fancybox=True, shadow=True, ncol=1, fontsize=INSET_FONT_SIZE
    )

    return inset_axes_object


def _plot_histogram(axes_object, bin_edges, bin_frequencies):
    """Plots histogram on existing axes.

    B = number of bins

    :param axes_object: Will plot histogram on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param bin_edges: length-(B + 1) numpy array with values at edges of each
        bin. These values will be plotted on the x-axis.
    :param bin_frequencies: length-B numpy array with fraction of examples in
        each bin. These values will be plotted on the y-axis.
    :return: histogram_axes_object: Axes handle for histogram only (also
        instance of `matplotlib.axes._subplots.AxesSubplot`).
    """

    histogram_axes_object = axes_object.twinx()
    axes_object.set_zorder(histogram_axes_object.get_zorder() + 1)
    axes_object.patch.set_visible(False)

    histogram_axes_object.bar(
        x=bin_edges[:-1], height=bin_frequencies, width=numpy.diff(bin_edges),
        color=INSET_HISTO_FACE_COLOUR, edgecolor=INSET_HISTO_EDGE_COLOUR,
        linewidth=INSET_HISTO_EDGE_WIDTH, align='edge'
    )

    return histogram_axes_object


def plot_spread_vs_skill(
        result_table_xarray, target_field_name,
        line_colour=DEFAULT_LINE_COLOUR, line_style='solid',
        line_width=DEFAULT_LINE_WIDTH):
    """Creates spread-skill plot for one target variable.

    :param result_table_xarray: xarray table in format returned by
        `spread_skill_utils.get_spread_vs_skill`.
    :param target_field_name: Will create plot for this target field only.
    :param line_colour: Line colour (in any format accepted by matplotlib).
    :param line_style: Line style (in any format accepted by matplotlib).
    :param line_width: Line width (in any format accepted by matplotlib).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Find the desired target field.
    rtx = result_table_xarray
    j = numpy.where(
        rtx.coords[ss_utils.FIELD_DIM].values == target_field_name
    )[0][0]

    spread_skill_reliability = rtx[ss_utils.SSREL_KEY].values[j]
    spread_skill_ratio = rtx[ss_utils.SSRAT_KEY].values[j]
    mean_prediction_stdevs = (
        rtx[ss_utils.MEAN_PREDICTION_STDEV_KEY].values[j, :]
    )
    rmse_values = rtx[ss_utils.RMSE_KEY].values[j, :]
    bin_edges = rtx[ss_utils.BIN_EDGE_PREDICTION_STDEV_KEY].values[j, :]
    example_counts = rtx[ss_utils.EXAMPLE_COUNT_KEY].values[j, :]
    mean_deterministic_preds = (
        rtx[ss_utils.MEAN_DETERMINISTIC_PRED_KEY].values[j, :]
    )
    mean_target_values = rtx[ss_utils.MEAN_TARGET_KEY].values[j, :]

    # Do actual stuff.
    nan_flags = numpy.logical_or(
        numpy.isnan(mean_prediction_stdevs),
        numpy.isnan(rmse_values)
    )
    assert not numpy.all(nan_flags)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    max_value_to_plot = 1.01 * max([
        numpy.nanmax(mean_prediction_stdevs),
        numpy.nanmax(rmse_values)
    ])
    perfect_x_coords = numpy.array([0, max_value_to_plot])
    perfect_y_coords = numpy.array([0, max_value_to_plot])
    axes_object.plot(
        perfect_x_coords,
        perfect_y_coords,
        color=numpy.full(3, 0.),
        linestyle='dashed',
        linewidth=REFERENCE_LINE_WIDTH
    )

    real_indices = numpy.where(numpy.invert(nan_flags))[0]
    axes_object.plot(
        mean_prediction_stdevs[real_indices],
        rmse_values[real_indices],
        color=line_colour, linestyle=line_style, linewidth=line_width,
        marker='o', markersize=12, markeredgewidth=0,
        markerfacecolor=line_colour, markeredgecolor=line_colour
    )

    axes_object.set_xlabel('Spread (standard deviation of ensemble)')
    axes_object.set_ylabel('Skill (RMSE of deterministic prediction)')

    bin_frequencies = example_counts.astype(float) / numpy.sum(example_counts)

    if numpy.isnan(mean_prediction_stdevs[-1]):
        bin_edges[-1] = bin_edges[-2] + (bin_edges[-2] - bin_edges[-3])
    else:
        bin_edges[-1] = (
            bin_edges[-2] + 2 * (mean_prediction_stdevs[-1] - bin_edges[-2])
        )

    histogram_axes_object = _plot_histogram(
        axes_object=axes_object,
        bin_edges=bin_edges,
        bin_frequencies=bin_frequencies * 100
    )
    histogram_axes_object.set_ylabel('% data samples per bin')

    # axes_object.set_xlim(
    #     min([bin_edges[0], 0]),
    #     bin_edges[-1]
    # )
    # axes_object.set_ylim(0, 1.01 * numpy.nanmax(rmse_values))

    axes_object.set_xlim(
        min([bin_edges[0], 0]),
        max_value_to_plot
    )
    axes_object.set_ylim(0, max_value_to_plot)

    inset_axes_object = _plot_means_as_inset(
        figure_object=figure_object,
        bin_centers=mean_prediction_stdevs,
        bin_mean_predictions=mean_deterministic_preds,
        bin_mean_target_values=mean_target_values,
        plotting_corner_string='bottom_right',
        for_spread_skill_plot=True
    )
    inset_axes_object.set_zorder(axes_object.get_zorder() + 1)

    inset_axes_object.set_xticks(axes_object.get_xticks())
    inset_axes_object.set_xlim(axes_object.get_xlim())
    inset_axes_object.set_xlabel('Spread', fontsize=INSET_FONT_SIZE)
    inset_axes_object.set_ylabel(
        'Mean target or pred', fontsize=INSET_FONT_SIZE
    )
    inset_axes_object.set_title(
        'Means by model spread', fontsize=INSET_FONT_SIZE
    )

    title_string = (
        'Spread vs. skill for {0:s}\n'
        'SSREL = {1:.1f}; SSRAT = {2:.2f}'
    ).format(
        FIELD_NAME_TO_TITLE[target_field_name],
        spread_skill_reliability,
        spread_skill_ratio
    )

    print(title_string)
    axes_object.set_title(title_string)

    return figure_object, axes_object


def plot_discard_test(
        result_table_xarray, target_field_name,
        line_colour=DEFAULT_LINE_COLOUR, line_style='solid',
        line_width=DEFAULT_LINE_WIDTH):
    """Plots results of discard test.

    :param result_table_xarray: xarray table in format returned by
        `discard_test_utils.run_discard_test`.
    :param target_field_name: Will create plot for this target field only.
    :param line_colour: See doc for `plot_spread_vs_skill`.
    :param line_style: Same.
    :param line_width: Same.
    :return: figure_object: Same.
    :return: axes_object: Same.
    """

    # Find the desired target field.
    rtx = result_table_xarray
    j = numpy.where(
        rtx.coords[ss_utils.FIELD_DIM].values == target_field_name
    )[0][0]

    deterministic_errors = rtx[dt_utils.DETERMINISTIC_ERROR_KEY].values[j, :]
    mean_deterministic_preds = (
        rtx[dt_utils.MEAN_DETERMINISTIC_PRED_KEY].values[j, :]
    )
    mean_target_values = rtx[dt_utils.MEAN_TARGET_KEY].values[j, :]
    discard_improvement = rtx[dt_utils.DISCARD_IMPROVEMENT_KEY].values[j]
    mono_fraction = rtx[dt_utils.MONO_FRACTION_KEY].values[j]
    discard_fractions = rtx.coords[dt_utils.DISCARD_FRACTION_DIM].values

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.plot(
        discard_fractions, deterministic_errors,
        color=line_colour, linestyle=line_style, linewidth=line_width,
        marker='o', markersize=12, markeredgewidth=0,
        markerfacecolor=line_colour, markeredgecolor=line_colour
    )

    axes_object.set_xlabel('Discard fraction')
    axes_object.set_xlim(left=0.)
    axes_object.set_ylabel('Mean deterministic error')

    inset_axes_object = _plot_means_as_inset(
        figure_object=figure_object,
        bin_centers=discard_fractions,
        bin_mean_predictions=mean_deterministic_preds,
        bin_mean_target_values=mean_target_values,
        plotting_corner_string='top_right',
        for_spread_skill_plot=False
    )
    inset_axes_object.set_zorder(axes_object.get_zorder() + 1)

    inset_axes_object.set_xticks(axes_object.get_xticks())
    inset_axes_object.set_xlim(axes_object.get_xlim())
    inset_axes_object.set_xlabel(
        'Discard fraction',
        fontsize=INSET_FONT_SIZE
    )
    inset_axes_object.set_ylabel(
        'Mean target or pred', fontsize=INSET_FONT_SIZE
    )
    inset_axes_object.set_title(
        'Means by model spread', fontsize=INSET_FONT_SIZE
    )

    title_string = (
        'Discard test for {0:s}\n'
        'MF = {1:.1f}%; DI = {2:.2f} per %'
    ).format(
        FIELD_NAME_TO_TITLE[target_field_name],
        100 * mono_fraction,
        100 * discard_improvement
    )

    print(title_string)
    axes_object.set_title(title_string)

    return figure_object, axes_object


def plot_pit_histogram(
        result_table_xarray, target_field_name,
        face_colour=DEFAULT_HISTOGRAM_FACE_COLOUR,
        edge_colour=DEFAULT_HISTOGRAM_EDGE_COLOUR,
        edge_width=DEFAULT_HISTOGRAM_EDGE_WIDTH):
    """Plots PIT (prob integral transform) histogram for one target variable.

    :param result_table_xarray: xarray table in format returned by
        `pit_utils.get_histogram_all_vars`.
    :param target_field_name: Will create plot for this target field only.
    :param face_colour: Face colour (in any format accepted by matplotlib).
    :param edge_colour: Edge colour (in any format accepted by matplotlib).
    :param edge_width: Edge width (in any format accepted by matplotlib).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Find the desired target field.
    rtx = result_table_xarray
    j = numpy.where(
        rtx.coords[ss_utils.FIELD_DIM].values == target_field_name
    )[0][0]

    pit_deviation = rtx[pith_utils.PIT_DEVIATION_KEY].values[j]
    bin_counts = rtx[pith_utils.EXAMPLE_COUNT_KEY].values[j, :]

    bin_edges = rtx.coords[pith_utils.BIN_EDGE_DIM].values
    bin_frequencies = bin_counts.astype(float) / numpy.sum(bin_counts)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.bar(
        x=bin_edges[:-1], height=bin_frequencies, width=numpy.diff(bin_edges),
        color=face_colour, edgecolor=edge_colour, linewidth=edge_width,
        align='edge'
    )

    num_bins = len(bin_edges) - 1
    perfect_x_coords = numpy.array([0, 1], dtype=float)
    perfect_y_coords = numpy.array([1. / num_bins, 1. / num_bins])
    axes_object.plot(
        perfect_x_coords,
        perfect_y_coords,
        color=REFERENCE_LINE_COLOUR,
        linestyle='dashed',
        linewidth=REFERENCE_LINE_WIDTH
    )

    axes_object.set_xlabel('PIT value')
    axes_object.set_ylabel('Frequency')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(bottom=0.)

    title_string = (
        'PIT histogram for {0:s}\n'
        'PITD = {1:.3f}'
    ).format(
        FIELD_NAME_TO_TITLE[target_field_name],
        pit_deviation
    )

    print(title_string)
    axes_object.set_title(title_string)

    return figure_object, axes_object
