"""Plots evaluation metrics as a function of lead time."""

import os
import sys
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
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

MODEL_MARKER_TYPE = 'o'
BASELINE_MARKER_TYPE = 's'
MARKER_SIZE = 16
MODEL_LINE_WIDTH = 5
BASELINE_LINE_WIDTH = 2.5

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

LINE_COLOURS = [
    numpy.array([27, 158, 119], dtype=float) / 255,
    numpy.array([217, 95, 2], dtype=float) / 255,
    numpy.array([117, 112, 179], dtype=float) / 255
]

FIELD_NAME_TO_FANCY = {
    canadian_fwi_utils.FFMC_NAME: 'FFMC',
    canadian_fwi_utils.DMC_NAME: 'DMC',
    canadian_fwi_utils.DC_NAME: 'DC',
    canadian_fwi_utils.BUI_NAME: 'BUI',
    canadian_fwi_utils.ISI_NAME: 'ISI',
    canadian_fwi_utils.FWI_NAME: 'FWI',
    canadian_fwi_utils.DSR_NAME: 'DSR'
}

FIRST_TARGET_FIELD_NAMES = [
    canadian_fwi_utils.DMC_NAME,
    canadian_fwi_utils.DC_NAME,
    canadian_fwi_utils.BUI_NAME
]
SECOND_TARGET_FIELD_NAMES = [
    canadian_fwi_utils.FFMC_NAME,
    canadian_fwi_utils.ISI_NAME,
    canadian_fwi_utils.FWI_NAME
]

METRIC_NAME_TO_FANCY_VERBOSE = {
    regression_eval.DWMSE_KEY: 'Dual-weighted MSE',
    regression_eval.RELIABILITY_KEY: 'Reliability',
    regression_eval.MAE_KEY: 'Mean absolute error',
    regression_eval.BIAS_KEY: 'Bias',
    regression_eval.MSE_KEY: 'Root mean squared error',
    regression_eval.CORRELATION_KEY: 'Correlation',
    regression_eval.MAE_SKILL_SCORE_KEY: 'MAE skill score',
    regression_eval.MSE_SKILL_SCORE_KEY: 'MSE skill score',
    regression_eval.DWMSE_SKILL_SCORE_KEY: 'DWMSE skill score',
    regression_eval.KGE_KEY: 'Kling-Gupta efficiency'
}

METRIC_NAME_TO_FANCY_ABBREV = {
    regression_eval.DWMSE_KEY: 'DWMSE',
    regression_eval.RELIABILITY_KEY: 'Reliability',
    regression_eval.MAE_KEY: 'MAE',
    regression_eval.BIAS_KEY: 'Bias',
    regression_eval.MSE_KEY: 'RMSE',
    regression_eval.CORRELATION_KEY: 'Correlation',
    regression_eval.MAE_SKILL_SCORE_KEY: 'MAE skill score',
    regression_eval.MSE_SKILL_SCORE_KEY: 'MSE skill score',
    regression_eval.DWMSE_SKILL_SCORE_KEY: 'DWMSE skill score',
    regression_eval.KGE_KEY: 'KGE'
}

# TODO(thunderhoser): Allow baseline to be unspecified.
# TODO(thunderhoser): Add confidence_level as input arg for bootstrapping.

MODEL_EVAL_FILES_ARG_NAME = 'model_eval_file_name_by_lead'
BASELINE_EVAL_FILES_ARG_NAME = 'baseline_eval_file_name_by_lead'
LEAD_TIMES_ARG_NAME = 'lead_times_days'
MODEL_DESCRIPTION_ARG_NAME = 'model_description_string'
BASELINE_DESCRIPTION_ARG_NAME = 'baseline_description_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_EVAL_FILES_HELP_STRING = (
    '1-D list of paths to evaluation files for model.  This list must have '
    'the same length as {0:s}.  Each file will be read by '
    '`regression_evaluation.read_file`.'
).format(
    LEAD_TIMES_ARG_NAME
)
BASELINE_EVAL_FILES_HELP_STRING = 'Same as {0:s} but for baseline.'.format(
    MODEL_EVAL_FILES_ARG_NAME
)
LEAD_TIMES_HELP_STRING = '1-D list of lead times.'
MODEL_DESCRIPTION_HELP_STRING = (
    'Description of model (will be used in figure legends).'
)
BASELINE_DESCRIPTION_HELP_STRING = (
    'Description of baseline (will be used in figure legends).'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_EVAL_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=MODEL_EVAL_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BASELINE_EVAL_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=BASELINE_EVAL_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=LEAD_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_DESCRIPTION_ARG_NAME, type=str, required=True,
    help=MODEL_DESCRIPTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BASELINE_DESCRIPTION_ARG_NAME, type=str, required=True,
    help=BASELINE_DESCRIPTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_metric(
        model_metric_matrix, baseline_metric_matrix, target_field_names,
        metric_name, model_description_string, baseline_description_string,
        lead_times_days, title_string, output_file_name_prefix):
    """Plots one metric, for both models and all target fields, vs. lead time.

    L = number of lead times
    T = number of target fields
    B = number of bootstrap replicates

    :param model_metric_matrix: L-by-T-by-B numpy array of metric values for
        model.
    :param baseline_metric_matrix: Same but for baseline.
    :param target_field_names: length-T list of target fields.
    :param metric_name: Name of metric.
    :param model_description_string: String description of model.
    :param baseline_description_string: String description of baseline.
    :param lead_times_days: length-L numpy array of lead times.
    :param title_string: Figure title.
    :param output_file_name_prefix: Beginning of path to output files.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    legend_handles = []
    legend_strings = []

    for i in range(len(FIRST_TARGET_FIELD_NAMES)):
        j = numpy.where(
            numpy.array(target_field_names) == FIRST_TARGET_FIELD_NAMES[i]
        )[0][0]

        this_handle = axes_object.plot(
            lead_times_days,
            numpy.mean(model_metric_matrix[:, j, ...], axis=-1),
            color=LINE_COLOURS[i],
            linestyle='solid',
            linewidth=MODEL_LINE_WIDTH,
            marker=MODEL_MARKER_TYPE,
            markersize=MARKER_SIZE,
            markerfacecolor=LINE_COLOURS[i],
            markeredgecolor=LINE_COLOURS[i],
            markeredgewidth=0
        )[0]

        legend_handles.append(this_handle)
        legend_strings.append('{0:s} {1:s}'.format(
            model_description_string,
            FIELD_NAME_TO_FANCY[FIRST_TARGET_FIELD_NAMES[i]]
        ))

    for i in range(len(FIRST_TARGET_FIELD_NAMES)):
        j = numpy.where(
            numpy.array(target_field_names) == FIRST_TARGET_FIELD_NAMES[i]
        )[0][0]

        this_handle = axes_object.plot(
            lead_times_days,
            numpy.mean(baseline_metric_matrix[:, j, ...], axis=-1),
            color=LINE_COLOURS[i],
            linestyle='dashed',
            linewidth=BASELINE_LINE_WIDTH,
            marker=BASELINE_MARKER_TYPE,
            markersize=MARKER_SIZE,
            markerfacecolor=LINE_COLOURS[i],
            markeredgecolor=LINE_COLOURS[i],
            markeredgewidth=0
        )[0]

        legend_handles.append(this_handle)
        legend_strings.append('{0:s} {1:s}'.format(
            baseline_description_string,
            FIELD_NAME_TO_FANCY[FIRST_TARGET_FIELD_NAMES[i]]
        ))

    axes_object.set_ylabel(metric_name)
    axes_object.set_xlim([
        numpy.min(lead_times_days) - 0.5,
        numpy.max(lead_times_days) + 0.5
    ])

    axes_object.legend(
        legend_handles, legend_strings,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
        shadow=True,
        ncol=2
    )

    axes_object.set_xticks(lead_times_days)
    axes_object.set_title(title_string)

    output_file_name = '{0:s}_slow.jpg'.format(output_file_name_prefix)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name,
        dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    legend_handles = []
    legend_strings = []

    for i in range(len(SECOND_TARGET_FIELD_NAMES)):
        j = numpy.where(
            numpy.array(target_field_names) == SECOND_TARGET_FIELD_NAMES[i]
        )[0][0]

        this_handle = axes_object.plot(
            lead_times_days,
            numpy.mean(model_metric_matrix[:, j, ...], axis=-1),
            color=LINE_COLOURS[i],
            linestyle='solid',
            linewidth=MODEL_LINE_WIDTH,
            marker=MODEL_MARKER_TYPE,
            markersize=MARKER_SIZE,
            markerfacecolor=LINE_COLOURS[i],
            markeredgecolor=LINE_COLOURS[i],
            markeredgewidth=0
        )[0]

        legend_handles.append(this_handle)
        legend_strings.append('{0:s} {1:s}'.format(
            model_description_string,
            FIELD_NAME_TO_FANCY[SECOND_TARGET_FIELD_NAMES[i]]
        ))

    for i in range(len(SECOND_TARGET_FIELD_NAMES)):
        j = numpy.where(
            numpy.array(target_field_names) == SECOND_TARGET_FIELD_NAMES[i]
        )[0][0]

        this_handle = axes_object.plot(
            lead_times_days,
            numpy.mean(baseline_metric_matrix[:, j, ...], axis=-1),
            color=LINE_COLOURS[i],
            linestyle='dashed',
            linewidth=BASELINE_LINE_WIDTH,
            marker=BASELINE_MARKER_TYPE,
            markersize=MARKER_SIZE,
            markerfacecolor=LINE_COLOURS[i],
            markeredgecolor=LINE_COLOURS[i],
            markeredgewidth=0
        )[0]

        legend_handles.append(this_handle)
        legend_strings.append('{0:s} {1:s}'.format(
            baseline_description_string,
            FIELD_NAME_TO_FANCY[SECOND_TARGET_FIELD_NAMES[i]]
        ))

    axes_object.set_ylabel(metric_name)
    axes_object.set_xlabel('Lead time (days)')
    axes_object.set_xlim([
        numpy.min(lead_times_days) - 0.5,
        numpy.max(lead_times_days) + 0.5
    ])

    axes_object.legend(
        legend_handles, legend_strings,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
        shadow=True,
        ncol=2
    )

    axes_object.set_xticks(lead_times_days)
    axes_object.set_title(title_string)

    output_file_name = '{0:s}_fast.jpg'.format(output_file_name_prefix)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name,
        dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(model_eval_file_name_by_lead, baseline_eval_file_name_by_lead,
         lead_times_days, model_description_string, baseline_description_string,
         output_dir_name):
    """Plots evaluation metrics as a function of lead time.

    This is effectively the main method.

    :param model_eval_file_name_by_lead: See documentation at top of this
        script.
    :param baseline_eval_file_name_by_lead: Same.
    :param lead_times_days: Same.
    :param model_description_string: Same.
    :param baseline_description_string: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    error_checking.assert_is_greater_numpy_array(lead_times_days, 0)
    num_lead_times = len(lead_times_days)

    error_checking.assert_is_numpy_array(
        numpy.array(model_eval_file_name_by_lead),
        exact_dimensions=numpy.array([num_lead_times], dtype=int)
    )
    error_checking.assert_is_numpy_array(
        numpy.array(baseline_eval_file_name_by_lead),
        exact_dimensions=numpy.array([num_lead_times], dtype=int)
    )

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read data.
    model_eval_tables_xarray = [xarray.Dataset()] * num_lead_times
    baseline_eval_tables_xarray = [xarray.Dataset()] * num_lead_times
    target_field_names = None

    for i in range(num_lead_times):
        print('Reading data from: "{0:s}"...'.format(
            model_eval_file_name_by_lead[i]
        ))
        model_eval_tables_xarray[i] = regression_eval.read_file(
            model_eval_file_name_by_lead[i]
        )

        print('Reading data from: "{0:s}"...'.format(
            baseline_eval_file_name_by_lead[i]
        ))
        baseline_eval_tables_xarray[i] = regression_eval.read_file(
            baseline_eval_file_name_by_lead[i]
        )

        if target_field_names is None:
            etx = model_eval_tables_xarray[i]
            target_field_names = etx.coords[regression_eval.FIELD_DIM].values
        else:
            model_eval_tables_xarray[i] = model_eval_tables_xarray[i].sel({
                regression_eval.FIELD_DIM: target_field_names
            })
            baseline_eval_tables_xarray[i] = (
                baseline_eval_tables_xarray[i].sel({
                    regression_eval.FIELD_DIM: target_field_names
                })
            )

    model_mean_ratio_matrix = numpy.stack([
        etx[regression_eval.PREDICTION_MEAN_KEY].values /
        etx[regression_eval.TARGET_MEAN_KEY].values
        for etx in model_eval_tables_xarray
    ], axis=0)

    baseline_mean_ratio_matrix = numpy.stack([
        etx[regression_eval.PREDICTION_MEAN_KEY].values /
        etx[regression_eval.TARGET_MEAN_KEY].values
        for etx in baseline_eval_tables_xarray
    ], axis=0)

    _plot_one_metric(
        model_metric_matrix=model_mean_ratio_matrix,
        baseline_metric_matrix=baseline_mean_ratio_matrix,
        target_field_names=target_field_names,
        metric_name='Climatology bias',
        model_description_string=model_description_string,
        baseline_description_string=baseline_description_string,
        lead_times_days=lead_times_days,
        title_string='Climatology bias (mean prediction over mean target)',
        output_file_name_prefix='{0:s}/climo_bias'.format(output_dir_name)
    )

    model_stdev_ratio_matrix = numpy.stack([
        etx[regression_eval.PREDICTION_STDEV_KEY].values /
        etx[regression_eval.TARGET_STDEV_KEY].values
        for etx in model_eval_tables_xarray
    ], axis=0)

    baseline_stdev_ratio_matrix = numpy.stack([
        etx[regression_eval.PREDICTION_STDEV_KEY].values /
        etx[regression_eval.TARGET_STDEV_KEY].values
        for etx in baseline_eval_tables_xarray
    ], axis=0)

    _plot_one_metric(
        model_metric_matrix=model_stdev_ratio_matrix,
        baseline_metric_matrix=baseline_stdev_ratio_matrix,
        target_field_names=target_field_names,
        metric_name='Variability bias',
        model_description_string=model_description_string,
        baseline_description_string=baseline_description_string,
        lead_times_days=lead_times_days,
        title_string='Variability bias (prediction stdev over target stdev)',
        output_file_name_prefix='{0:s}/variability_bias'.format(output_dir_name)
    )

    other_metric_names = [
        regression_eval.DWMSE_KEY, regression_eval.RELIABILITY_KEY,
        regression_eval.MAE_KEY, regression_eval.BIAS_KEY,
        regression_eval.MSE_KEY, regression_eval.CORRELATION_KEY,
        regression_eval.MAE_SKILL_SCORE_KEY, regression_eval.MSE_SKILL_SCORE_KEY,
        regression_eval.DWMSE_SKILL_SCORE_KEY, regression_eval.KGE_KEY
    ]

    for this_metric_name in other_metric_names:
        model_metric_matrix = numpy.stack([
            etx[this_metric_name].values
            for etx in model_eval_tables_xarray
        ], axis=0)

        baseline_metric_matrix = numpy.stack([
            etx[this_metric_name].values
            for etx in baseline_eval_tables_xarray
        ], axis=0)

        _plot_one_metric(
            model_metric_matrix=model_metric_matrix,
            baseline_metric_matrix=baseline_metric_matrix,
            target_field_names=target_field_names,
            metric_name=METRIC_NAME_TO_FANCY_ABBREV[this_metric_name],
            model_description_string=model_description_string,
            baseline_description_string=baseline_description_string,
            lead_times_days=lead_times_days,
            title_string=METRIC_NAME_TO_FANCY_VERBOSE[this_metric_name],
            output_file_name_prefix='{0:s}/{1:s}'.format(
                output_dir_name,
                METRIC_NAME_TO_FANCY_ABBREV[this_metric_name].lower().replace(
                    ' ', '_'
                )
            )
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_eval_file_name_by_lead=getattr(
            INPUT_ARG_OBJECT, MODEL_EVAL_FILES_ARG_NAME
        ),
        baseline_eval_file_name_by_lead=getattr(
            INPUT_ARG_OBJECT, BASELINE_EVAL_FILES_ARG_NAME
        ),
        lead_times_days=numpy.array(
            getattr(INPUT_ARG_OBJECT, LEAD_TIMES_ARG_NAME), dtype=int
        ),
        model_description_string=getattr(
            INPUT_ARG_OBJECT, MODEL_DESCRIPTION_ARG_NAME
        ),
        baseline_description_string=getattr(
            INPUT_ARG_OBJECT, BASELINE_DESCRIPTION_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
