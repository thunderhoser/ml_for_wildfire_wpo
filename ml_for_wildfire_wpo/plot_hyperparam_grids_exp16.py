"""For every error metric and lead time, plots values AAFO Exp 16 HPs.

AAFO = as a function of
HP = hyperparameter
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

import file_system_utils
import gg_plotting_utils
import imagemagick_utils
import canadian_fwi_utils
import regression_evaluation as regression_eval
import spread_skill_utils as ss_utils
import discard_test_utils as dt_utils
import pit_histogram_utils as pith_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRIC_NAMES = [
    regression_eval.MSE_KEY, regression_eval.DWMSE_KEY,
    regression_eval.MAE_KEY, regression_eval.BIAS_KEY,
    regression_eval.CORRELATION_KEY, regression_eval.KGE_KEY,
    regression_eval.RELIABILITY_KEY,
    ss_utils.SSREL_KEY, ss_utils.SSRAT_KEY,
    dt_utils.MONO_FRACTION_KEY, pith_utils.PIT_DEVIATION_KEY
]

METRIC_NAMES_FANCY = [
    'Mean squared error',
    'Dual-weighted MSE',
    'Mean absolute error',
    'Bias',
    'Correlation',
    'Kling-Gupta efficiency',
    'Reliability',
    'Spread-skill reliability',
    'Spread-skill ratio',
    'Monotonicity fraction',
    'PIT deviation'
]

TARGET_FIELD_TO_ABBREV = {
    canadian_fwi_utils.FFMC_NAME: 'FFMC',
    canadian_fwi_utils.DMC_NAME: 'DMC',
    canadian_fwi_utils.DC_NAME: 'DC',
    canadian_fwi_utils.ISI_NAME: 'ISI',
    canadian_fwi_utils.BUI_NAME: 'BUI',
    canadian_fwi_utils.FWI_NAME: 'FWI',
    canadian_fwi_utils.DSR_NAME: 'DSR'
}

FINE_TUNING_START_EPOCHS_AXIS1 = numpy.array([20, 40, 60, 80, 100], dtype=int)
RAMPUP_EPOCH_COUNTS_AXIS2 = numpy.array([10, 20, 30, 40, 50], dtype=int)
PREDICTOR_TIME_STRATEGIES_AXIS3 = [
    'all-daily-valid-times-up-to-target-time',
    'different-valid-times-up-to-target-time',
    'different-valid-times-up-to-target-time-plus2days',
    'same-valid-times-for-every-model-lead'
]
PREDICTOR_TIME_STRATEGIES_FANCY_AXIS3 = [
    'Daily intervals',
    'Diff intervals per model LT',
    'Diff intervals per model LT + 2 days',
    'Same predictor times for every model LT'
]

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.175
WHITE_COLOUR = numpy.full(3, 1.)
BLACK_COLOUR = numpy.full(3, 0.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.175
SELECTED_MARKER_INDICES = numpy.array([0, 0, 0], dtype=int)

MAIN_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis', lut=20)
MONO_FRACTION_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='cividis', lut=20)
SSRAT_COLOUR_MAP_NAME = 'seismic'
BIAS_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='seismic', lut=20)

NAN_COLOUR = numpy.full(3, 152. / 255)
MAIN_COLOUR_MAP_OBJECT.set_bad(NAN_COLOUR)
MONO_FRACTION_COLOUR_MAP_OBJECT.set_bad(NAN_COLOUR)

FONT_SIZE = 26
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
MODEL_LEAD_TIME_ARG_NAME = 'model_lead_time_days'
TARGET_FIELDS_ARG_NAME = 'target_field_names'

EXPERIMENT_DIR_HELP_STRING = 'Name of top-level directory with models.'
MODEL_LEAD_TIME_HELP_STRING = 'Model lead time.'
TARGET_FIELDS_HELP_STRING = (
    'List of target fields.  Each must be accepted by '
    '`canadian_fwi_utils.check_field_name`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_LEAD_TIME_ARG_NAME, type=int, required=True,
    help=MODEL_LEAD_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=TARGET_FIELDS_HELP_STRING
)


def __rank_one_metric(metric_matrix_3d, metric_name):
    """Ranks values of one metric across all models.

    A = length of first hyperparam axis
    B = length of second hyperparam axis
    C = length of third hyperparam axis
    N = total number of models = A x B x C

    :param metric_matrix_3d: A-by-B-by-C numpy array of values.
    :param metric_name: Name of metric.
    :return: i_sort_indices: length-N numpy array of sort indices along the
        first hyperparameter axis.  The best model has value
        first_hyperparams[i_sort_indices[0]] for the first hyperparam;
        the second-best model has value first_hyperparams[i_sort_indices[1]];
        ...;
        etc.
    :return: j_sort_indices: Same but for second hyperparam axis.
    :return: k_sort_indices: Same but for third hyperparam axis.
    """

    values_linear = numpy.ravel(metric_matrix_3d) + 0.

    if metric_name in [
            dt_utils.MONO_FRACTION_KEY,
            regression_eval.CORRELATION_KEY,
            regression_eval.KGE_KEY
    ]:
        values_linear[numpy.isnan(values_linear)] = -numpy.inf
        sort_indices_linear = numpy.argsort(-values_linear)
    elif metric_name == regression_eval.BIAS_KEY:
        values_linear[numpy.isnan(values_linear)] = numpy.inf
        sort_indices_linear = numpy.argsort(numpy.absolute(values_linear))
    elif metric_name == ss_utils.SSRAT_KEY:
        values_linear[numpy.isnan(values_linear)] = numpy.inf
        sort_indices_linear = numpy.argsort(numpy.absolute(1. - values_linear))
    else:
        values_linear[numpy.isnan(values_linear)] = numpy.inf
        sort_indices_linear = numpy.argsort(values_linear)

    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_linear, metric_matrix_3d.shape
    )

    return i_sort_indices, j_sort_indices, k_sort_indices


def _finite_percentile(input_array, percentile_level):
    """Takes percentile of input array, considering only finite values.

    :param input_array: numpy array.
    :param percentile_level: Percentile level, ranging from 0...100.
    :return: output_percentile: Percentile value.
    """

    return numpy.percentile(
        input_array[numpy.isfinite(input_array)], percentile_level
    )


def _get_ssrat_colour_scheme(max_colour_value):
    """Returns colour scheme for spread-skill ratio (SSRAT).

    :param max_colour_value: Max value in colour scheme.
    :return: colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :return: colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    """

    orig_colour_map_object = pyplot.get_cmap(SSRAT_COLOUR_MAP_NAME)

    negative_values = numpy.linspace(0, 1, num=1001, dtype=float)
    positive_values = numpy.linspace(1, max_colour_value, num=1001, dtype=float)
    bias_values = numpy.concatenate((negative_values, positive_values))

    normalized_values = numpy.linspace(0, 1, num=len(bias_values), dtype=float)
    rgb_matrix = orig_colour_map_object(normalized_values)[:, :-1]

    colour_map_object = matplotlib.colors.ListedColormap(rgb_matrix)
    colour_map_object.set_bad(NAN_COLOUR)
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        bias_values, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def _plot_scores_2d(
        score_matrix, colour_map_object, colour_norm_object, x_tick_labels,
        y_tick_labels):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    :param x_tick_labels: length-N list of tick labels.
    :param y_tick_labels: length-M list of tick labels.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    score_matrix_to_plot = numpy.ma.masked_where(
        numpy.isnan(score_matrix), score_matrix
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.imshow(
        score_matrix_to_plot, cmap=colour_map_object, norm=colour_norm_object,
        origin='lower'
    )

    x_tick_values = numpy.linspace(
        0, score_matrix.shape[1] - 1, num=score_matrix.shape[1], dtype=float
    )
    y_tick_values = numpy.linspace(
        0, score_matrix.shape[0] - 1, num=score_matrix.shape[0], dtype=float
    )
    pyplot.xticks(x_tick_values, x_tick_labels)
    pyplot.yticks(y_tick_values, y_tick_labels)

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=score_matrix[numpy.invert(numpy.isnan(score_matrix))],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        font_size=FONT_SIZE, fraction_of_axis_length=1.
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _read_metrics_one_model(model_dir_name, model_lead_time_days,
                            target_field_names):
    """Reads metrics for one model.
    
    F = number of target fields

    :param model_dir_name: Name of directory with trained model and validation
        data.
    :param model_lead_time_days: Model lead time.
    :param target_field_names: length-F list with names of target fields.
    :return: metric_dict: Dictionary, where each key is a string from the list
        `METRIC_NAMES` and each value is a length-F numpy array.
    """

    metric_dict = {}
    for this_metric_name in METRIC_NAMES:
        metric_dict[this_metric_name] = numpy.nan

    validation_dir_name = '{0:s}/validation_lead-time-days={1:02d}'.format(
        model_dir_name, model_lead_time_days
    )
    if not os.path.isdir(validation_dir_name):
        return metric_dict

    this_file_name = '{0:s}/ungridded_evaluation.nc'.format(validation_dir_name)
    if not os.path.isfile(this_file_name):
        return metric_dict

    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_eval_table_xarray = regression_eval.read_file(this_file_name)
    etx = this_eval_table_xarray
    
    field_indices = numpy.array([
        numpy.where(etx.coords[regression_eval.FIELD_DIM].values == f)[0][0]
        for f in target_field_names
    ], dtype=int)

    metric_dict[regression_eval.MSE_KEY] = numpy.mean(
        etx[regression_eval.MSE_KEY].values[field_indices, :], axis=1
    )
    metric_dict[regression_eval.DWMSE_KEY] = numpy.mean(
        etx[regression_eval.DWMSE_KEY].values[field_indices, :], axis=1
    )
    metric_dict[regression_eval.MAE_KEY] = numpy.mean(
        etx[regression_eval.MAE_KEY].values[field_indices, :], axis=1
    )
    metric_dict[regression_eval.BIAS_KEY] = numpy.mean(
        etx[regression_eval.BIAS_KEY].values[field_indices, :], axis=1
    )
    metric_dict[regression_eval.CORRELATION_KEY] = numpy.mean(
        etx[regression_eval.CORRELATION_KEY].values[field_indices, :], axis=1
    )
    metric_dict[regression_eval.KGE_KEY] = numpy.mean(
        etx[regression_eval.KGE_KEY].values[field_indices, :], axis=1
    )
    metric_dict[regression_eval.RELIABILITY_KEY] = numpy.mean(
        etx[regression_eval.RELIABILITY_KEY].values[field_indices, :], axis=1
    )

    this_file_name = '{0:s}/spread_vs_skill.nc'.format(validation_dir_name)

    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_ss_table_xarray = ss_utils.read_results(this_file_name)
    sstx = this_ss_table_xarray

    metric_dict[ss_utils.SSREL_KEY] = (
        sstx[ss_utils.SSREL_KEY].values[field_indices]
    )
    metric_dict[ss_utils.SSRAT_KEY] = (
        sstx[ss_utils.SSRAT_KEY].values[field_indices]
    )

    this_file_name = '{0:s}/pit_histograms.nc'.format(validation_dir_name)

    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_pit_table_xarray = pith_utils.read_results(this_file_name)
    pitx = this_pit_table_xarray

    metric_dict[pith_utils.PIT_DEVIATION_KEY] = (
        pitx[pith_utils.PIT_DEVIATION_KEY].values[field_indices]
    )

    this_file_name = '{0:s}/discard_test.nc'.format(validation_dir_name)

    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_dt_table_xarray = dt_utils.read_results(this_file_name)
    dtx = this_dt_table_xarray

    metric_dict[dt_utils.MONO_FRACTION_KEY] = (
        dtx[dt_utils.MONO_FRACTION_KEY].values[field_indices]
    )

    return metric_dict


def _print_ranking_1field_1metric(metric_matrix_5d, field_index, field_name,
                                  metric_index):
    """Print ranking for one field and one metric.

    A = length of first hyperparam axis
    B = length of second hyperparam axis
    C = length of third hyperparam axis
    F = number of fields
    M = number of metrics

    :param metric_matrix_5d: A-by-B-by-C-by-F-by-M numpy array of metric values.
    :param field_index: Index for field.
    :param field_name: Name of field.
    :param metric_index: Index for metric.
    """

    i_sort_indices, j_sort_indices, k_sort_indices = __rank_one_metric(
        metric_matrix_3d=metric_matrix_5d[..., field_index, metric_index],
        metric_name=METRIC_NAMES[metric_index]
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:d}th-best {1:s} for {2:s} = {3:.3g} ... '
            'fine-tuning start epoch = {4:d} ... '
            'num rampup epochs = {5:d} ... '
            'predictor time strategy = {6:s}'
        ).format(
            m + 1,
            METRIC_NAMES_FANCY[metric_index],
            TARGET_FIELD_TO_ABBREV[field_name],
            metric_matrix_5d[i, j, k, field_index, metric_index],
            FINE_TUNING_START_EPOCHS_AXIS1[i],
            RAMPUP_EPOCH_COUNTS_AXIS2[j],
            PREDICTOR_TIME_STRATEGIES_FANCY_AXIS3[k].lower()[0] +
            PREDICTOR_TIME_STRATEGIES_FANCY_AXIS3[k][1:]
        ))


def _print_ranking_all_metrics(metric_matrix_5d, target_field_names,
                               main_field_index, main_metric_index):
    """Prints ranking for all metrics.

    :param metric_matrix_5d: See doc for `_print_ranking_1field_1metric`.
    :param target_field_names: 1-D list with names of target fields.
    :param main_field_index: Index for main field.  This can be None.
    :param main_metric_index: Index of main metric.  This can be None.
    """

    metric_rank_matrix_5d = numpy.full(metric_matrix_5d.shape, numpy.nan)
    num_fields = metric_matrix_5d.shape[-2]
    num_metrics = metric_matrix_5d.shape[-1]

    for f in range(num_fields):
        for m in range(num_metrics):
            values_linear = numpy.ravel(metric_matrix_5d[..., f, m]) + 0.

            if METRIC_NAMES[m] in [
                    dt_utils.MONO_FRACTION_KEY,
                    regression_eval.CORRELATION_KEY,
                    regression_eval.KGE_KEY
            ]:
                values_linear = -1 * values_linear
            elif METRIC_NAMES[m] == regression_eval.BIAS_KEY:
                values_linear = numpy.absolute(values_linear)
            elif METRIC_NAMES[m] == ss_utils.SSRAT_KEY:
                values_linear = numpy.absolute(1. - values_linear)

            values_linear[numpy.isnan(values_linear)] = numpy.inf
            metric_rank_matrix_5d[..., f, m] = numpy.reshape(
                rankdata(values_linear, method='average'),
                metric_rank_matrix_5d.shape[:-2]
            )

    if main_field_index is not None and main_metric_index is not None:
        values_linear = numpy.ravel(
            metric_rank_matrix_5d[..., main_field_index, main_metric_index]
        )
    elif main_metric_index is not None:
        values_linear = numpy.ravel(
            numpy.mean(metric_rank_matrix_5d[..., main_metric_index], axis=-1)
        )
    elif main_field_index is not None:
        values_linear = numpy.ravel(
            numpy.mean(metric_rank_matrix_5d[..., main_field_index, :], axis=-1)
        )
    else:
        values_linear = numpy.ravel(
            numpy.mean(metric_rank_matrix_5d, axis=(-1, -2))
        )

    sort_indices_linear = numpy.argsort(values_linear)
    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_linear, metric_rank_matrix_5d.shape[:-2]
    )

    names = METRIC_NAMES
    mrm = metric_rank_matrix_5d

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        for f in range(len(target_field_names)):
            print((
                'Fine-tuning start epoch = {0:d} ... '
                'num rampup epochs = {1:d} ... '
                'predictor time strategy = {2:s} ... '
                'target field = {3:s} ...\n'
                'MAE/MSE/DWMSE ranks = {4:.1f}, {5:.1f}, {6:.1f} ... '
                'bias/REL ranks = {7:.1f}, {8:.1f} ... '
                'correlation/KGE ranks = {9:.1f}, {10:.1f} ... '
                'UQ ranks (SSREL/SSRAT/PITD/MF) = '
                '{11:.1f}, {12:.1f}, {13:.1f}, {14:.1f}'
            ).format(
                FINE_TUNING_START_EPOCHS_AXIS1[i],
                RAMPUP_EPOCH_COUNTS_AXIS2[j],
                PREDICTOR_TIME_STRATEGIES_FANCY_AXIS3[k].lower()[0] +
                PREDICTOR_TIME_STRATEGIES_FANCY_AXIS3[k][1:],
                TARGET_FIELD_TO_ABBREV[target_field_names[f]],
                mrm[i, j, k, f, names.index(regression_eval.MAE_KEY)],
                mrm[i, j, k, f, names.index(regression_eval.MSE_KEY)],
                mrm[i, j, k, f, names.index(regression_eval.DWMSE_KEY)],
                mrm[i, j, k, f, names.index(regression_eval.BIAS_KEY)],
                mrm[i, j, k, f, names.index(regression_eval.RELIABILITY_KEY)],
                mrm[i, j, k, f, names.index(regression_eval.CORRELATION_KEY)],
                mrm[i, j, k, f, names.index(regression_eval.KGE_KEY)],
                mrm[i, j, k, f, names.index(ss_utils.SSREL_KEY)],
                mrm[i, j, k, f, names.index(ss_utils.SSRAT_KEY)],
                mrm[i, j, k, f, names.index(pith_utils.PIT_DEVIATION_KEY)],
                mrm[i, j, k, f, names.index(dt_utils.MONO_FRACTION_KEY)]
            ))

        print('\n')


def _run(experiment_dir_name, model_lead_time_days, target_field_names):
    """For every error metric and lead time, plots values AAFO Exp 16 HPs.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of this script.
    :param model_lead_time_days: Same.
    :param target_field_names: Same.
    """

    length_axis1 = len(FINE_TUNING_START_EPOCHS_AXIS1)
    length_axis2 = len(RAMPUP_EPOCH_COUNTS_AXIS2)
    length_axis3 = len(PREDICTOR_TIME_STRATEGIES_AXIS3)
    num_fields = len(target_field_names)
    num_metrics = len(METRIC_NAMES)

    y_tick_labels = ['{0:d}'.format(c) for c in FINE_TUNING_START_EPOCHS_AXIS1]
    x_tick_labels = ['{0:d}'.format(d) for d in RAMPUP_EPOCH_COUNTS_AXIS2]

    y_axis_label = 'Fine-tuning start epoch'
    x_axis_label = 'Number of rampup epochs'

    metric_matrix_5d = numpy.full(
        (length_axis1, length_axis2, length_axis3, num_fields, num_metrics),
        numpy.nan
    )

    for i in range(length_axis1):
        for j in range(length_axis2):
            for k in range(length_axis3):
                this_model_dir_name = (
                    '{0:s}/second-lead-time-start-epoch={1:03d}_'
                    'num-rampup-epochs={2:03d}_predictor-time-strategy={3:s}'
                ).format(
                    experiment_dir_name,
                    FINE_TUNING_START_EPOCHS_AXIS1[i],
                    RAMPUP_EPOCH_COUNTS_AXIS2[j],
                    PREDICTOR_TIME_STRATEGIES_AXIS3[k]
                )

                this_metric_dict = _read_metrics_one_model(
                    model_dir_name=this_model_dir_name,
                    model_lead_time_days=model_lead_time_days,
                    target_field_names=target_field_names
                )
                for m in range(num_metrics):
                    metric_matrix_5d[i, j, k, :, m] = this_metric_dict[
                        METRIC_NAMES[m]
                    ]

    print(SEPARATOR_STRING)

    for f in range(num_fields):
        for m in range(num_metrics):
            _print_ranking_1field_1metric(
                metric_matrix_5d=metric_matrix_5d,
                field_index=f,
                field_name=target_field_names[f],
                metric_index=m
            )
            print(SEPARATOR_STRING)

    print(
        'List below is based on metric-averaged and field-averaged rankings!'
    )

    _print_ranking_all_metrics(
        metric_matrix_5d=metric_matrix_5d,
        target_field_names=target_field_names,
        main_field_index=None,
        main_metric_index=None
    )
    print(SEPARATOR_STRING)

    # for f in range(num_fields):
    #     print((
    #         'List below is based on metric-averaged rankings for {0:s}!'
    #     ).format(
    #         TARGET_FIELD_TO_ABBREV[target_field_names[f]]
    #     ))
    #
    #     _print_ranking_all_metrics(
    #         metric_matrix_5d=metric_matrix_5d,
    #         target_field_names=target_field_names,
    #         main_field_index=f,
    #         main_metric_index=None
    #     )
    #     print(SEPARATOR_STRING)
    #
    # for m in range(num_metrics):
    #     print((
    #         'List below is based on field-averaged rankings for {0:s}!'
    #     ).format(
    #         METRIC_NAMES[m]
    #     ))
    #
    #     _print_ranking_all_metrics(
    #         metric_matrix_5d=metric_matrix_5d,
    #         target_field_names=target_field_names,
    #         main_field_index=None,
    #         main_metric_index=m
    #     )
    #     print(SEPARATOR_STRING)

    output_dir_name = '{0:s}/hyperparam_grids/lead_time_{1:02d}days'.format(
        experiment_dir_name, model_lead_time_days
    )
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    for f in range(num_fields):
        for m in range(num_metrics):
            panel_file_names = [''] * length_axis3

            for k in range(length_axis3):
                if METRIC_NAMES[m] in [
                        dt_utils.MONO_FRACTION_KEY,
                        regression_eval.CORRELATION_KEY,
                        regression_eval.KGE_KEY
                ]:
                    max_colour_value = _finite_percentile(
                        metric_matrix_5d[..., f, m], 100
                    )
                    min_colour_value = _finite_percentile(
                        metric_matrix_5d[..., f, m], 5
                    )
                    colour_norm_object = matplotlib.colors.Normalize(
                        vmin=min_colour_value, vmax=max_colour_value, clip=False
                    )

                    if METRIC_NAMES[m] == dt_utils.MONO_FRACTION_KEY:
                        colour_map_object = MONO_FRACTION_COLOUR_MAP_OBJECT
                    else:
                        colour_map_object = MAIN_COLOUR_MAP_OBJECT

                    best_linear_index = numpy.nanargmax(
                        numpy.ravel(metric_matrix_5d[..., f, m])
                    )
                    marker_colour = BLACK_COLOUR

                elif METRIC_NAMES[m] == regression_eval.BIAS_KEY:
                    max_colour_value = _finite_percentile(
                        numpy.absolute(metric_matrix_5d[..., f, m]), 97.5
                    )
                    colour_norm_object = matplotlib.colors.Normalize(
                        vmin=-1 * max_colour_value, vmax=max_colour_value,
                        clip=False
                    )
                    colour_map_object = BIAS_COLOUR_MAP_OBJECT

                    best_linear_index = numpy.nanargmin(
                        numpy.ravel(numpy.absolute(metric_matrix_5d[..., f, m]))
                    )
                    marker_colour = BLACK_COLOUR

                elif METRIC_NAMES[m] == ss_utils.SSRAT_KEY:
                    this_offset = _finite_percentile(
                        numpy.absolute(metric_matrix_5d[..., f, m] - 1.), 97.5
                    )
                    colour_map_object, colour_norm_object = (
                        _get_ssrat_colour_scheme(
                            max_colour_value=1. + this_offset
                        )
                    )

                    best_linear_index = numpy.nanargmin(numpy.absolute(
                        numpy.ravel(metric_matrix_5d[..., f, m]) - 1.
                    ))
                    marker_colour = BLACK_COLOUR

                else:
                    max_colour_value = _finite_percentile(
                        metric_matrix_5d[..., f, m], 95
                    )
                    min_colour_value = _finite_percentile(
                        metric_matrix_5d[..., f, m], 0
                    )
                    colour_norm_object = matplotlib.colors.Normalize(
                        vmin=min_colour_value, vmax=max_colour_value, clip=False
                    )
                    colour_map_object = MAIN_COLOUR_MAP_OBJECT

                    best_linear_index = numpy.nanargmin(
                        numpy.ravel(metric_matrix_5d[..., f, m])
                    )
                    marker_colour = WHITE_COLOUR

                figure_object, axes_object = _plot_scores_2d(
                    score_matrix=metric_matrix_5d[..., k, f, m],
                    colour_map_object=colour_map_object,
                    colour_norm_object=colour_norm_object,
                    x_tick_labels=x_tick_labels,
                    y_tick_labels=y_tick_labels
                )

                best_indices = numpy.unravel_index(
                    best_linear_index, metric_matrix_5d[..., f, m].shape
                )

                figure_width_px = (
                    figure_object.get_size_inches()[0] * figure_object.dpi
                )
                marker_size_px = figure_width_px * (
                    BEST_MARKER_SIZE_GRID_CELLS / metric_matrix_5d.shape[1]
                )

                if best_indices[2] == k:
                    axes_object.plot(
                        best_indices[1], best_indices[0],
                        linestyle='None', marker=BEST_MARKER_TYPE,
                        markersize=marker_size_px, markeredgewidth=0,
                        markerfacecolor=marker_colour,
                        markeredgecolor=marker_colour
                    )

                if SELECTED_MARKER_INDICES[2] == k:
                    axes_object.plot(
                        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                        linestyle='None', marker=SELECTED_MARKER_TYPE,
                        markersize=marker_size_px, markeredgewidth=0,
                        markerfacecolor=marker_colour,
                        markeredgecolor=marker_colour
                    )

                axes_object.set_xlabel(x_axis_label)
                axes_object.set_ylabel(y_axis_label)

                title_string = (
                    '{0:s} for {1:s}\nPredictor times = {2:s}'
                ).format(
                    METRIC_NAMES_FANCY[m],
                    TARGET_FIELD_TO_ABBREV[target_field_names[f]],
                    PREDICTOR_TIME_STRATEGIES_FANCY_AXIS3[k].lower()[0] +
                    PREDICTOR_TIME_STRATEGIES_FANCY_AXIS3[k][1:]
                )
                axes_object.set_title(title_string)

                panel_file_names[k] = '{0:s}/{1:s}_{2:s}_{3:s}.jpg'.format(
                    output_dir_name,
                    METRIC_NAMES[m].replace('_', '-'),
                    target_field_names[f].replace('_', '-'),
                    PREDICTOR_TIME_STRATEGIES_AXIS3[k].replace('_', '-')
                )

                print('Saving figure to: "{0:s}"...'.format(
                    panel_file_names[k]
                ))
                figure_object.savefig(
                    panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
                    pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(figure_object)

            num_panel_rows = int(numpy.floor(
                numpy.sqrt(length_axis3)
            ))
            num_panel_columns = int(numpy.ceil(
                float(length_axis3) / num_panel_rows
            ))
            concat_figure_file_name = '{0:s}/{1:s}_{2:s}.jpg'.format(
                output_dir_name,
                METRIC_NAMES[m].replace('_', '-'),
                target_field_names[f].replace('_', '-')
            )

            print('Concatenating panels to: "{0:s}"...'.format(
                concat_figure_file_name
            ))
            imagemagick_utils.concatenate_images(
                input_file_names=panel_file_names,
                output_file_name=concat_figure_file_name,
                num_panel_rows=num_panel_rows,
                num_panel_columns=num_panel_columns
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
        model_lead_time_days=getattr(
            INPUT_ARG_OBJECT, MODEL_LEAD_TIME_ARG_NAME
        ),
        target_field_names=getattr(INPUT_ARG_OBJECT, TARGET_FIELDS_ARG_NAME)
    )
