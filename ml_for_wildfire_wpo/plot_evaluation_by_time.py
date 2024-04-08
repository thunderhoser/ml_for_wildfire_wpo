"""Plots evaluation metrics as a function of time period."""

import os
import sys
import copy
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.patches
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import canadian_fwi_utils
import regression_evaluation as regression_eval

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

# TODO(thunderhoser): Add UQ-evaluation files to the mix.

FIELD_NAME_TO_FANCY = {
    canadian_fwi_utils.FFMC_NAME: 'FFMC',
    canadian_fwi_utils.DMC_NAME: 'DMC',
    canadian_fwi_utils.DC_NAME: 'DC',
    canadian_fwi_utils.BUI_NAME: 'BUI',
    canadian_fwi_utils.ISI_NAME: 'ISI',
    canadian_fwi_utils.FWI_NAME: 'FWI',
    canadian_fwi_utils.DSR_NAME: 'DSR'
}

MARKER_TYPE = 'o'
MARKER_SIZE = 16
LINE_WIDTH = 4
HISTOGRAM_EDGE_WIDTH = 1.5

POLYGON_OPACITY = 0.5

HISTOGRAM_FACE_COLOUR = numpy.full(3, 152. / 255)
HISTOGRAM_FACE_COLOUR = matplotlib.colors.to_rgba(HISTOGRAM_FACE_COLOUR, 0.5)
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

LINE_COLOURS_MEAN_STDEV = [
    numpy.array([31, 120, 180], dtype=float) / 255,
    # numpy.array([166, 206, 227], dtype=float) / 255,
    numpy.array([76, 157, 199], dtype=float) / 255,
    numpy.array([51, 160, 44], dtype=float) / 255,
    # numpy.array([178, 223, 138], dtype=float) / 255
    numpy.array([122, 198, 54], dtype=float) / 255
]
LINE_STYLES_MEAN_STDEV = ['solid', 'dashed', 'solid', 'dashed']

LINE_COLOURS_DWMSE_REL = [
    numpy.array([27, 158, 119], dtype=float) / 255,
    numpy.array([217, 95, 2], dtype=float) / 255
]
LINE_STYLES_DWMSE_REL = ['solid', 'solid']

LINE_COLOURS_MAE_BIAS_RMSE = [
    numpy.array([27, 158, 119], dtype=float) / 255,
    numpy.array([217, 95, 2], dtype=float) / 255,
    numpy.array([117, 112, 179], dtype=float) / 255
]
LINE_STYLES_MAE_BIAS_RMSE = ['solid', 'solid', 'solid']

LINE_COLOURS_SKILL_SCORES = [
    numpy.array([27, 158, 119], dtype=float) / 255,
    numpy.array([217, 95, 2], dtype=float) / 255,
    numpy.array([117, 112, 179], dtype=float) / 255,
    numpy.array([231, 41, 138], dtype=float) / 255,
    numpy.full(3, 0.)
]
LINE_STYLES_SKILL_SCORES = ['solid', 'solid', 'solid', 'solid', 'solid']

INPUT_FILES_ARG_NAME = 'input_file_names_by_period'
PERIOD_DESCRIPTIONS_ARG_NAME = 'period_description_strings'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to evaluation files, one per time period.  These files will '
    'be read by `regression_evaluation.read_file`.'
)
PERIOD_DESCRIPTIONS_HELP_STRING = (
    'List of descriptions one per time period.  For example, if {0:s} contains '
    '12 monthly files, you probably want this list to '
    '"Jan" "Feb" "Mar" ... "Dec".'
).format(
    INPUT_FILES_ARG_NAME
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level for bootstrapped confidence intervals.  If the '
    'evaluation files do not contain bootstrapped results, this argument will '
    'not be used.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PERIOD_DESCRIPTIONS_ARG_NAME, type=str, nargs='+', required=True,
    help=PERIOD_DESCRIPTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_set_of_metrics_1field(
        metric_matrix, metric_names,
        num_examples_by_period, description_string_by_period,
        line_colour_by_metric, line_style_by_metric, main_axis_flag_by_metric,
        main_y_label_string, aux_y_label_string, title_string,
        confidence_level, output_file_name):
    """Plots one set of metrics.

    P = number of time periods
    M = number of metrics
    B = number of bootstrap replicates

    :param metric_matrix: M-by-P-by-B numpy array of metric values.
    :param metric_names: length-M list of metric names.
    :param num_examples_by_period: length-P numpy array of example counts
        (sample size).
    :param description_string_by_period: length-P list of descriptions.
    :param line_colour_by_metric: length-M list of line colours.
    :param line_style_by_metric: length-M list of line styles.
    :param main_axis_flag_by_metric: length-M numpy array of Boolean flags,
        where True (False) means that the metric will be plotted on the left
        (right) y-axis.
    :param main_y_label_string: y-axis label for main (left) axis.
    :param aux_y_label_string: y-axis label for auxiliary (right) axis.
    :param title_string: Figure title.
    :param confidence_level: Confidence level for bootstrapping.
    :param output_file_name: Path to output file.
    """

    num_metrics = metric_matrix.shape[0]
    num_periods = metric_matrix.shape[1]
    num_bootstrap_reps = metric_matrix.shape[2]

    figure_object, main_axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    plotting_bias_on_main_axes = any([
        'bias' in mn.lower()
        for mn in numpy.array(metric_names)[main_axis_flag_by_metric]
    ])
    plotting_bias_on_aux_axes = any([
        'bias' in mn.lower()
        for mn in numpy.array(metric_names)[main_axis_flag_by_metric == False]
    ])
    x_values = numpy.linspace(0, num_periods - 1, num=num_periods, dtype=float)

    if numpy.any(main_axis_flag_by_metric == False):
        aux_axes_object = main_axes_object.twinx()
        main_axes_object.set_zorder(aux_axes_object.get_zorder() + 1)
        main_axes_object.patch.set_visible(False)
    else:
        aux_axes_object = None

    legend_handles = [None] * num_metrics
    legend_strings = copy.deepcopy(metric_names)

    for i in range(num_metrics):
        if main_axis_flag_by_metric[i]:
            this_handle = main_axes_object.plot(
                x_values, numpy.mean(metric_matrix[i, ...], axis=1),
                color=line_colour_by_metric[i],
                linestyle=line_style_by_metric[i],
                linewidth=LINE_WIDTH,
                marker=MARKER_TYPE,
                markersize=MARKER_SIZE,
                markerfacecolor=line_colour_by_metric[i],
                markeredgecolor=line_colour_by_metric[i],
                markeredgewidth=0
            )[0]
        else:
            this_handle = aux_axes_object.plot(
                x_values, numpy.mean(metric_matrix[i, ...], axis=1),
                color=line_colour_by_metric[i],
                linestyle=line_style_by_metric[i],
                linewidth=LINE_WIDTH,
                marker=MARKER_TYPE,
                markersize=MARKER_SIZE,
                markerfacecolor=line_colour_by_metric[i],
                markeredgecolor=line_colour_by_metric[i],
                markeredgewidth=0
            )[0]

        legend_handles[i] = this_handle

        if num_bootstrap_reps > 1:
            x_matrix = numpy.expand_dims(x_values, axis=1)
            x_matrix = numpy.repeat(
                x_matrix, axis=1, repeats=num_bootstrap_reps
            )

            polygon_coord_matrix = regression_eval.confidence_interval_to_polygon(
                x_value_matrix=x_matrix,
                y_value_matrix=metric_matrix[i, ...],
                confidence_level=confidence_level,
                same_order=False
            )

            polygon_colour = matplotlib.colors.to_rgba(
                line_colour_by_metric[i], POLYGON_OPACITY
            )
            patch_object = matplotlib.patches.Polygon(
                polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
            )
            main_axes_object.add_patch(patch_object)

    main_axes_object.set_ylabel(main_y_label_string)
    main_axes_object.set_xlim([
        numpy.min(x_values) - 0.5,
        numpy.max(x_values) + 0.5
    ])

    if aux_axes_object is not None:
        aux_axes_object.set_ylabel(aux_y_label_string)
        aux_axes_object.set_xlim([
            numpy.min(x_values) - 0.5,
            numpy.max(x_values) + 0.5
        ])

    histogram_axes_object = main_axes_object.twinx()
    if aux_axes_object is not None:
        aux_axes_object.set_zorder(histogram_axes_object.get_zorder() + 1)
        aux_axes_object.patch.set_visible(False)

    main_axes_object.set_zorder(histogram_axes_object.get_zorder() + 2)
    main_axes_object.patch.set_visible(False)

    this_handle = histogram_axes_object.bar(
        x=x_values, height=num_examples_by_period,
        width=1.,
        color=HISTOGRAM_FACE_COLOUR,
        edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Sample size')

    histogram_axes_object.set_ylabel('')
    histogram_axes_object.set_yticks([], [])

    print('Sample size by period: {0:s}'.format(
        str(num_examples_by_period)
    ))

    num_legend_columns = int(numpy.ceil(
        numpy.sqrt(len(legend_handles))
    ))
    main_axes_object.legend(
        legend_handles, legend_strings,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
        shadow=True,
        ncol=num_legend_columns
    )

    main_axes_object.set_xticks(x_values)
    main_axes_object.set_xticklabels(description_string_by_period, rotation=90.)
    main_axes_object.set_title(title_string)

    if plotting_bias_on_main_axes:
        main_axes_object.plot(
            main_axes_object.get_xlim(), numpy.full(2, 0.),
            linestyle='solid', linewidth=4, color=numpy.full(3, 0.)
        )
    if plotting_bias_on_aux_axes:
        aux_axes_object.plot(
            aux_axes_object.get_xlim(), numpy.full(2, 0.),
            linestyle='solid', linewidth=4, color=numpy.full(3, 0.)
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name,
        dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(eval_file_name_by_period, description_string_by_period,
         confidence_level, output_dir_name):
    """Plots evaluation metrics as a function of time period.

    This is effectively the main method.

    :param eval_file_name_by_period: See documentation at top of this script.
    :param description_string_by_period: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    num_periods = len(eval_file_name_by_period)
    error_checking.assert_is_numpy_array(
        numpy.array(description_string_by_period),
        exact_dimensions=numpy.array([num_periods], dtype=int)
    )

    error_checking.assert_is_geq(confidence_level, 0.8)
    error_checking.assert_is_less_than(confidence_level, 1.)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read data.
    evaluation_tables_xarray = [xarray.Dataset()] * num_periods

    for i in range(num_periods):
        print('Reading data from: "{0:s}"...'.format(
            eval_file_name_by_period[i]
        ))
        evaluation_tables_xarray[i] = regression_eval.read_file(
            eval_file_name_by_period[i]
        )

    first_etx = evaluation_tables_xarray[0]
    target_field_names = first_etx.coords[regression_eval.FIELD_DIM].values
    num_target_fields = len(target_field_names)

    # Plot stuff.
    num_examples_by_period = numpy.array([
        numpy.nansum(
            etx[regression_eval.RELIABILITY_COUNT_KEY].values, axis=1
        )[0]
        for etx in evaluation_tables_xarray
    ])

    target_mean_matrix = numpy.stack([
        etx[regression_eval.TARGET_MEAN_KEY].values
        for etx in evaluation_tables_xarray
    ], axis=0)

    target_stdev_matrix = numpy.stack([
        etx[regression_eval.TARGET_STDEV_KEY].values
        for etx in evaluation_tables_xarray
    ], axis=0)

    prediction_mean_matrix = numpy.stack([
        etx[regression_eval.PREDICTION_MEAN_KEY].values
        for etx in evaluation_tables_xarray
    ], axis=0)

    prediction_stdev_matrix = numpy.stack([
        etx[regression_eval.PREDICTION_STDEV_KEY].values
        for etx in evaluation_tables_xarray
    ], axis=0)

    metric_matrix = numpy.stack([
        target_mean_matrix, prediction_mean_matrix,
        target_stdev_matrix, prediction_stdev_matrix
    ], axis=0)

    metric_names = [
        'Target mean', 'Prediction mean', 'Target stdev', 'Prediction stdev'
    ]

    for k in range(num_target_fields):
        _plot_one_set_of_metrics_1field(
            metric_matrix=metric_matrix[..., k, :],
            metric_names=metric_names,
            num_examples_by_period=num_examples_by_period,
            description_string_by_period=description_string_by_period,
            line_colour_by_metric=LINE_COLOURS_MEAN_STDEV,
            line_style_by_metric=LINE_STYLES_MEAN_STDEV,
            main_axis_flag_by_metric=numpy.array([1, 1, 0, 0], dtype=bool),
            main_y_label_string='Mean',
            aux_y_label_string='Stdev',
            confidence_level=confidence_level,
            title_string='Actual and predicted climo for {0:s}'.format(
                FIELD_NAME_TO_FANCY[target_field_names[k]]
            ),
            output_file_name='{0:s}/means_and_stdevs_{1:s}.jpg'.format(
                output_dir_name,
                target_field_names[k].replace('_', '-')
            )
        )

    dwmse_matrix = numpy.stack([
        etx[regression_eval.DWMSE_KEY].values
        for etx in evaluation_tables_xarray
    ], axis=0)

    reliability_matrix = numpy.stack([
        etx[regression_eval.RELIABILITY_KEY].values
        for etx in evaluation_tables_xarray
    ], axis=0)

    metric_matrix = numpy.stack([dwmse_matrix, reliability_matrix], axis=0)
    metric_names = ['DWMSE', 'Reliability']

    for k in range(num_target_fields):
        _plot_one_set_of_metrics_1field(
            metric_matrix=metric_matrix[..., k, :],
            metric_names=metric_names,
            num_examples_by_period=num_examples_by_period,
            description_string_by_period=description_string_by_period,
            line_colour_by_metric=LINE_COLOURS_DWMSE_REL,
            line_style_by_metric=LINE_STYLES_DWMSE_REL,
            main_axis_flag_by_metric=numpy.array([1, 0], dtype=bool),
            main_y_label_string='Dual-weighted MSE',
            aux_y_label_string='Reliability',
            confidence_level=confidence_level,
            title_string='DWMSE and reliability for {0:s}'.format(
                FIELD_NAME_TO_FANCY[target_field_names[k]]
            ),
            output_file_name='{0:s}/dwmse_reliability_{1:s}.jpg'.format(
                output_dir_name,
                target_field_names[k].replace('_', '-')
            )
        )

    mae_matrix = numpy.stack([
        etx[regression_eval.MAE_KEY].values
        for etx in evaluation_tables_xarray
    ], axis=0)

    bias_matrix = numpy.stack([
        etx[regression_eval.BIAS_KEY].values
        for etx in evaluation_tables_xarray
    ], axis=0)

    mse_matrix = numpy.stack([
        etx[regression_eval.MSE_KEY].values
        for etx in evaluation_tables_xarray
    ], axis=0)

    rmse_matrix = numpy.sqrt(mse_matrix)

    metric_matrix = numpy.stack(
        [mae_matrix, bias_matrix, rmse_matrix], axis=0
    )
    metric_names = ['MAE', 'Bias', 'RMSE']

    for k in range(num_target_fields):
        _plot_one_set_of_metrics_1field(
            metric_matrix=metric_matrix[..., k, :],
            metric_names=metric_names,
            num_examples_by_period=num_examples_by_period,
            description_string_by_period=description_string_by_period,
            line_colour_by_metric=LINE_COLOURS_MAE_BIAS_RMSE,
            line_style_by_metric=LINE_STYLES_MAE_BIAS_RMSE,
            main_axis_flag_by_metric=numpy.array([1, 0, 1], dtype=bool),
            main_y_label_string='MAE or RMSE',
            aux_y_label_string='Bias',
            confidence_level=confidence_level,
            title_string='Basic metrics for {0:s}'.format(
                FIELD_NAME_TO_FANCY[target_field_names[k]]
            ),
            output_file_name='{0:s}/mae_bias_rmse_{1:s}.jpg'.format(
                output_dir_name,
                target_field_names[k].replace('_', '-')
            )
        )

    maess_matrix = numpy.stack([
        etx[regression_eval.MAE_SKILL_SCORE_KEY].values
        for etx in evaluation_tables_xarray
    ], axis=0)

    msess_matrix = numpy.stack([
        etx[regression_eval.MSE_SKILL_SCORE_KEY].values
        for etx in evaluation_tables_xarray
    ], axis=0)

    dwmsess_matrix = numpy.stack([
        etx[regression_eval.DWMSE_SKILL_SCORE_KEY].values
        for etx in evaluation_tables_xarray
    ], axis=0)

    correlation_matrix = numpy.stack([
        etx[regression_eval.CORRELATION_KEY].values
        for etx in evaluation_tables_xarray
    ], axis=0)

    kge_matrix = numpy.stack([
        etx[regression_eval.KGE_KEY].values
        for etx in evaluation_tables_xarray
    ], axis=0)

    metric_matrix = numpy.stack([
        maess_matrix, msess_matrix, dwmsess_matrix,
        correlation_matrix, kge_matrix
    ], axis=0)

    metric_names = [
        'MAE skill score', 'MSE skill score', 'DWMSE skill score',
        'Correlation', 'KGE'
    ]

    for k in range(num_target_fields):
        _plot_one_set_of_metrics_1field(
            metric_matrix=metric_matrix[..., k, :],
            metric_names=metric_names,
            num_examples_by_period=num_examples_by_period,
            description_string_by_period=description_string_by_period,
            line_colour_by_metric=LINE_COLOURS_SKILL_SCORES,
            line_style_by_metric=LINE_STYLES_SKILL_SCORES,
            main_axis_flag_by_metric=numpy.full(5, 1, dtype=bool),
            main_y_label_string='Metric',
            aux_y_label_string='',
            confidence_level=confidence_level,
            title_string='Skill scores for {0:s}'.format(
                FIELD_NAME_TO_FANCY[target_field_names[k]]
            ),
            output_file_name='{0:s}/skill_scores_{1:s}.jpg'.format(
                output_dir_name,
                target_field_names[k].replace('_', '-')
            )
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        eval_file_name_by_period=getattr(
            INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME
        ),
        description_string_by_period=getattr(
            INPUT_ARG_OBJECT, PERIOD_DESCRIPTIONS_ARG_NAME
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
