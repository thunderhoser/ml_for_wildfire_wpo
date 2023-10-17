"""Plots model evaluation."""

import os
import sys
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
import prediction_io
import canadian_fwi_io
import canadian_fwi_utils
import regression_evaluation as regression_eval
import neural_net
import evaluation_plotting as eval_plotting

# TODO(thunderhoser): This script currently handles only regression, not
# classification.

TARGET_FIELD_NAME_TO_VERBOSE = {
    canadian_fwi_utils.DC_NAME: 'drought code (DC)',
    canadian_fwi_utils.DMC_NAME: 'duff moisture code (DMC)',
    canadian_fwi_utils.FFMC_NAME: 'fine-fuel moisture code (FFMC)',
    canadian_fwi_utils.ISI_NAME: 'initial-spread index (ISI)',
    canadian_fwi_utils.BUI_NAME: 'build-up index (BUI)',
    canadian_fwi_utils.FWI_NAME: 'fire-weather index (FWI)',
    canadian_fwi_utils.DSR_NAME: 'daily severity rating (DSR)'
}

POLYGON_OPACITY = 0.5
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILES_ARG_NAME = 'input_eval_file_names'
LINE_STYLES_ARG_NAME = 'line_styles'
LINE_COLOURS_ARG_NAME = 'line_colours'
SET_DESCRIPTIONS_ARG_NAME = 'set_descriptions'
PLOT_FULL_DISTS_ARG_NAME = 'plot_full_error_distributions'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
METRICS_IN_TITLES_ARG_NAME = 'report_metrics_in_titles'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'Space-separated list of paths to input files (each will be read by '
    '`regression_evaluation.read_file`).'
)
LINE_STYLES_HELP_STRING = (
    'Space-separated list of line styles (in any format accepted by '
    'matplotlib).  Must have same length as `{0:s}`.'
).format(INPUT_FILES_ARG_NAME)

LINE_COLOURS_HELP_STRING = (
    'Space-separated list of line colours.  Each colour must be a length-3 '
    'array of (R, G, B) intensities ranging from 0...255.  Colours in each '
    'array should be underscore-separated, so the list should look like '
    '"0_0_0 217_95_2", for examples.  List must have same length as `{0:s}`.'
).format(INPUT_FILES_ARG_NAME)

SET_DESCRIPTIONS_HELP_STRING = (
    'Space-separated list of set descriptions, to be used in legends.  Must '
    'have same length as `{0:s}`.'
).format(INPUT_FILES_ARG_NAME)

PLOT_FULL_DISTS_HELP_STRING = (
    'Boolean flag.  If 1, for each evaluation set, will plot full error '
    'distribution with boxplot.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level (from 0...1).  If you do not want to plot confidence '
    'intervals, leave this alone.'
)
METRICS_IN_TITLES_HELP_STRING = (
    'Boolean flag.  If 1 (0), will (not) report overall metrics in panel '
    'titles.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LINE_STYLES_ARG_NAME, type=str, nargs='+', required=True,
    help=LINE_STYLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LINE_COLOURS_ARG_NAME, type=str, nargs='+', required=True,
    help=LINE_COLOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SET_DESCRIPTIONS_ARG_NAME, type=str, nargs='+', required=True,
    help=SET_DESCRIPTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_FULL_DISTS_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_FULL_DISTS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=-1,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + METRICS_IN_TITLES_ARG_NAME, type=int, required=False, default=1,
    help=METRICS_IN_TITLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_attributes_diagram(
        evaluation_tables_xarray, line_styles, line_colours,
        set_descriptions_abbrev, set_descriptions_verbose, confidence_level,
        climo_mean_target_value, target_field_name, report_reliability_in_title,
        output_dir_name, force_plot_legend=False):
    """Plots attributes diagram for each set and each target variable.

    S = number of evaluation sets
    T_v = number of vector target variables
    T_s = number of scalar target variables
    H = number of heights

    :param evaluation_tables_xarray: length-S list of xarray tables in format
        returned by `evaluation.read_file`.
    :param line_styles: length-S list of line styles.
    :param line_colours: length-S list of line colours.
    :param set_descriptions_abbrev: length-S list of abbreviated descriptions
        for evaluation sets.
    :param set_descriptions_verbose: length-S list of verbose descriptions for
        evaluation sets.
    :param confidence_level: See documentation at top of file.
    :param climo_mean_target_value: Mean target value in training data, i.e.,
        "climatology".
    :param target_field_name: Name of target variable.
    :param report_reliability_in_title: Boolean flag.  If True, will report
        overall reliability in title.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    :param force_plot_legend: Boolean flag.
    """

    mean_predictions_by_set = [
        t[regression_eval.RELIABILITY_X_KEY].values
        for t in evaluation_tables_xarray
    ]
    mean_observations_by_set = [
        t[regression_eval.RELIABILITY_Y_KEY].values
        for t in evaluation_tables_xarray
    ]
    bin_centers_by_set = [
        t[regression_eval.RELIABILITY_BIN_CENTER_KEY].values
        for t in evaluation_tables_xarray
    ]
    example_counts_by_set = [
        t[regression_eval.RELIABILITY_COUNT_KEY].values
        for t in evaluation_tables_xarray
    ]
    inverted_example_counts_by_set = [
        t[regression_eval.INV_RELIABILITY_COUNT_KEY].values
        for t in evaluation_tables_xarray
    ]
    reliabilities_by_set = [
        t[regression_eval.RELIABILITY_KEY].values
        for t in evaluation_tables_xarray
    ]
    mse_skill_scores_by_set = [
        t[regression_eval.MSE_SKILL_SCORE_KEY].values
        for t in evaluation_tables_xarray
    ]

    concat_values = numpy.concatenate([
        numpy.nanmean(a, axis=-1)
        for a in mean_predictions_by_set + mean_observations_by_set
        if a is not None
    ])

    if numpy.all(numpy.isnan(concat_values)):
        return

    max_value_to_plot = numpy.nanpercentile(concat_values, 100.)
    min_value_to_plot = numpy.nanpercentile(concat_values, 0.)
    num_evaluation_sets = len(evaluation_tables_xarray)

    for main_index in range(num_evaluation_sets):
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        legend_handles = []
        legend_strings = []

        this_handle = eval_plotting.plot_attributes_diagram(
            figure_object=figure_object,
            axes_object=axes_object,
            mean_predictions=
            numpy.nanmean(mean_predictions_by_set[main_index], axis=-1),
            mean_observations=
            numpy.nanmean(mean_observations_by_set[main_index], axis=-1),
            mean_value_in_training=climo_mean_target_value,
            min_value_to_plot=min_value_to_plot,
            max_value_to_plot=max_value_to_plot,
            line_colour=line_colours[main_index],
            line_style=line_styles[main_index],
            line_width=4
        )

        if this_handle is not None:
            legend_handles.append(this_handle)
            legend_strings.append(set_descriptions_verbose[main_index])

        num_bootstrap_reps = mean_predictions_by_set[main_index].shape[1]

        if num_bootstrap_reps > 1 and confidence_level is not None:
            polygon_coord_matrix = (
                regression_eval.confidence_interval_to_polygon(
                    x_value_matrix=mean_predictions_by_set[main_index],
                    y_value_matrix=mean_observations_by_set[main_index],
                    confidence_level=confidence_level,
                    same_order=False
                )
            )

            polygon_colour = matplotlib.colors.to_rgba(
                line_colours[main_index], POLYGON_OPACITY
            )
            patch_object = matplotlib.patches.Polygon(
                polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
            )
            axes_object.add_patch(patch_object)

        eval_plotting.plot_inset_histogram(
            figure_object=figure_object,
            bin_centers=bin_centers_by_set[main_index],
            bin_counts=example_counts_by_set[main_index],
            has_predictions=True,
            bar_colour=line_colours[main_index]
        )

        # eval_plotting.plot_inset_histogram(
        #     figure_object=figure_object,
        #     bin_centers=inverted_bin_centers_by_set[main_index],
        #     bin_counts=inverted_example_counts_by_set[main_index],
        #     has_predictions=False,
        #     bar_colour=line_colours[main_index]
        # )

        eval_plotting.plot_inset_histogram(
            figure_object=figure_object,
            bin_centers=bin_centers_by_set[main_index],
            bin_counts=inverted_example_counts_by_set[main_index],
            has_predictions=False,
            bar_colour=line_colours[main_index]
        )

        axes_object.set_xlabel('Prediction')
        axes_object.set_ylabel('Conditional mean observation')
        title_string = 'Attributes diagram for {0:s}'.format(
            TARGET_FIELD_NAME_TO_VERBOSE[target_field_name]
        )
        if report_reliability_in_title:
            title_string += '\nREL = {0:.2f}; MSESS = {1:.2f}'.format(
                numpy.mean(reliabilities_by_set[main_index]),
                numpy.mean(mse_skill_scores_by_set[main_index])
            )

        axes_object.set_title(title_string)

        for i in range(num_evaluation_sets):
            if i == main_index:
                continue

            this_handle = eval_plotting._plot_reliability_curve(
                axes_object=axes_object,
                mean_predictions=
                numpy.nanmean(mean_predictions_by_set[i], axis=-1),
                mean_observations=
                numpy.nanmean(mean_observations_by_set[i], axis=-1),
                min_value_to_plot=min_value_to_plot,
                max_value_to_plot=max_value_to_plot,
                line_colour=line_colours[i],
                line_style=line_styles[i],
                line_width=4
            )

            if this_handle is not None:
                legend_handles.append(this_handle)
                legend_strings.append(set_descriptions_verbose[i])

            num_bootstrap_reps = mean_predictions_by_set[i].shape[1]

            if num_bootstrap_reps > 1 and confidence_level is not None:
                polygon_coord_matrix = (
                    regression_eval.confidence_interval_to_polygon(
                        x_value_matrix=mean_predictions_by_set[i],
                        y_value_matrix=mean_observations_by_set[i],
                        confidence_level=confidence_level,
                        same_order=False
                    )
                )

                polygon_colour = matplotlib.colors.to_rgba(
                    line_colours[i], POLYGON_OPACITY
                )
                patch_object = matplotlib.patches.Polygon(
                    polygon_coord_matrix, lw=0,
                    ec=polygon_colour, fc=polygon_colour
                )
                axes_object.add_patch(patch_object)

        if len(legend_handles) > 1 or force_plot_legend:
            axes_object.legend(
                legend_handles, legend_strings, loc='center left',
                bbox_to_anchor=(0, 0.35), fancybox=True, shadow=False,
                facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
            )

        figure_file_name = '{0:s}/{1:s}_attributes_{2:s}.jpg'.format(
            output_dir_name,
            target_field_name.replace('_', '-'),
            set_descriptions_abbrev[main_index]
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


def _run(evaluation_file_names, line_styles, line_colour_strings,
         set_descriptions_verbose, plot_full_error_distributions,
         confidence_level, report_metrics_in_titles, output_dir_name):
    """Plots model evaluation.

    This is effectively the main method.

    :param evaluation_file_names: See documentation at top of file.
    :param line_styles: Same.
    :param line_colour_strings: Same.
    :param set_descriptions_verbose: Same.
    :param plot_full_error_distributions: Same.
    :param confidence_level: Same.
    :param report_metrics_in_titles: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if confidence_level < 0:
        confidence_level = None

    if confidence_level is not None:
        error_checking.assert_is_geq(confidence_level, 0.9)
        error_checking.assert_is_less_than(confidence_level, 1.)

    num_evaluation_sets = len(evaluation_file_names)
    expected_dim = numpy.array([num_evaluation_sets], dtype=int)

    error_checking.assert_is_string_list(line_styles)
    error_checking.assert_is_numpy_array(
        numpy.array(line_styles), exact_dimensions=expected_dim
    )

    error_checking.assert_is_string_list(set_descriptions_verbose)
    error_checking.assert_is_numpy_array(
        numpy.array(set_descriptions_verbose), exact_dimensions=expected_dim
    )

    set_descriptions_verbose = [
        s.replace('_', ' ') for s in set_descriptions_verbose
    ]
    set_descriptions_abbrev = [
        s.lower().replace(' ', '-') for s in set_descriptions_verbose
    ]

    error_checking.assert_is_string_list(line_colour_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(line_colour_strings), exact_dimensions=expected_dim
    )
    line_colours = [
        numpy.fromstring(s, dtype=float, sep='_') / 255
        for s in line_colour_strings
    ]

    for i in range(num_evaluation_sets):
        error_checking.assert_is_numpy_array(
            line_colours[i], exact_dimensions=numpy.array([3], dtype=int)
        )
        error_checking.assert_is_geq_numpy_array(line_colours[i], 0.)
        error_checking.assert_is_leq_numpy_array(line_colours[i], 1.)

    # Do actual stuff.
    evaluation_tables_xarray = [xarray.Dataset()] * num_evaluation_sets
    target_field_name = None
    target_norm_file_name = None

    for i in range(num_evaluation_sets):
        print('Reading data from: "{0:s}"...'.format(evaluation_file_names[i]))
        evaluation_tables_xarray[i] = regression_eval.read_file(
            evaluation_file_names[i]
        )

        model_file_name = (
            evaluation_tables_xarray[i].attrs[regression_eval.MODEL_FILE_KEY]
        )
        model_metafile_name = neural_net.find_metafile(
            model_file_name=model_file_name, raise_error_if_missing=True
        )

        print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
        model_metadata_dict = neural_net.read_metafile(model_metafile_name)
        generator_option_dict = model_metadata_dict[
            neural_net.TRAINING_OPTIONS_KEY
        ]
        goptd = generator_option_dict

        if i == 0:
            target_field_name = goptd[neural_net.TARGET_FIELD_KEY]
            target_norm_file_name = goptd[neural_net.TARGET_NORM_FILE_KEY]

        assert target_field_name == goptd[neural_net.TARGET_FIELD_KEY]
        assert target_norm_file_name == goptd[neural_net.TARGET_NORM_FILE_KEY]

    print('Reading normalization params from: "{0:s}"...'.format(
        target_norm_file_name
    ))
    target_norm_param_table_xarray = canadian_fwi_io.read_normalization_file(
        target_norm_file_name
    )
    tnpt = target_norm_param_table_xarray

    k = numpy.where(
        tnpt.coords[canadian_fwi_utils.FIELD_DIM].values == target_field_name
    )[0][0]
    climo_mean_target_value = tnpt[canadian_fwi_utils.MEAN_VALUE_KEY].values[k]

    _plot_attributes_diagram(
        evaluation_tables_xarray=evaluation_tables_xarray,
        line_styles=line_styles,
        line_colours=line_colours,
        set_descriptions_abbrev=set_descriptions_abbrev,
        set_descriptions_verbose=set_descriptions_verbose,
        confidence_level=confidence_level,
        climo_mean_target_value=climo_mean_target_value,
        target_field_name=target_field_name,
        report_reliability_in_title=report_metrics_in_titles,
        output_dir_name=output_dir_name
    )

    for i in range(num_evaluation_sets):
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        etx = evaluation_tables_xarray[i]

        eval_plotting.plot_taylor_diagram(
            target_stdev=numpy.nanmean(
                etx[regression_eval.TARGET_STDEV_KEY].values
            ),
            prediction_stdev=numpy.nanmean(
                etx[regression_eval.PREDICTION_STDEV_KEY].values
            ),
            correlation=numpy.nanmean(
                etx[regression_eval.CORRELATION_KEY].values
            ),
            marker_colour=line_colours[i],
            figure_object=figure_object
        )

        title_string = 'Taylor diagram for {0:s}'.format(
            TARGET_FIELD_NAME_TO_VERBOSE[target_field_name]
        )
        if report_metrics_in_titles:
            title_string += (
                '\nActual/pred stdevs = {0:.2f}, {1:.2f}; corr = {2:.2f}'
            ).format(
                numpy.nanmean(etx[regression_eval.TARGET_STDEV_KEY].values),
                numpy.nanmean(etx[regression_eval.PREDICTION_STDEV_KEY].values),
                numpy.nanmean(etx[regression_eval.CORRELATION_KEY].values)
            )

        axes_object.set_title(title_string)

        figure_file_name = '{0:s}/{1:s}_taylor_{2:s}.jpg'.format(
            output_dir_name,
            target_field_name.replace('_', '-'),
            set_descriptions_abbrev[i]
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    if not plot_full_error_distributions:
        return

    for i in range(num_evaluation_sets):
        prediction_file_names = evaluation_tables_xarray[i].attrs[
            regression_eval.PREDICTION_FILES_KEY
        ]
        num_times = len(prediction_file_names)
        error_matrix = numpy.array([], dtype=float)

        for j in range(num_times):
            print('Reading data from: "{0:s}"...'.format(
                prediction_file_names[j]
            ))
            this_prediction_table_xarray = prediction_io.read_file(
                prediction_file_names[j]
            )
            tpt = this_prediction_table_xarray

            if error_matrix.size == 0:
                these_dim = (
                    (num_times,) + tpt[prediction_io.TARGET_KEY].values.shape
                )
                error_matrix = numpy.full(these_dim, numpy.nan)

            error_matrix[j, ...] = (
                tpt[prediction_io.PREDICTION_KEY].values -
                tpt[prediction_io.TARGET_KEY].values
            )
            error_matrix[j, ...][
                tpt[prediction_io.WEIGHT_KEY].values < 0.05
            ] = numpy.nan

        error_values = numpy.ravel(error_matrix)
        error_values = error_values[numpy.invert(numpy.isnan(error_values))]

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        eval_plotting.plot_error_distribution(
            error_values=error_values,
            min_error_to_plot=numpy.percentile(error_values, 0.),
            max_error_to_plot=numpy.percentile(error_values, 100.),
            axes_object=axes_object
        )

        title_string = 'Error distribution for {0:s}'.format(
            TARGET_FIELD_NAME_TO_VERBOSE[target_field_name]
        )
        axes_object.set_title(title_string)

        figure_file_name = '{0:s}/{1:s}_error-distribution_{2:s}.jpg'.format(
            output_dir_name,
            target_field_name.replace('_', '-'),
            set_descriptions_abbrev[i]
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
        evaluation_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        line_styles=getattr(INPUT_ARG_OBJECT, LINE_STYLES_ARG_NAME),
        line_colour_strings=getattr(INPUT_ARG_OBJECT, LINE_COLOURS_ARG_NAME),
        set_descriptions_verbose=getattr(
            INPUT_ARG_OBJECT, SET_DESCRIPTIONS_ARG_NAME
        ),
        plot_full_error_distributions=bool(
            getattr(INPUT_ARG_OBJECT, PLOT_FULL_DISTS_ARG_NAME)
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        report_metrics_in_titles=bool(
            getattr(INPUT_ARG_OBJECT, METRICS_IN_TITLES_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
