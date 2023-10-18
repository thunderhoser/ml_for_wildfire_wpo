"""Plots gridded model evaluation."""

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

import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import border_io
import canadian_fwi_utils
import regression_evaluation as regression_eval
import neural_net
import plotting_utils
import gfs_plotting

# TODO(thunderhoser): This script currently handles only regression, not
# classification.

TOLERANCE = 1e-6

TARGET_FIELD_NAME_TO_VERBOSE = {
    canadian_fwi_utils.DC_NAME: 'drought code (DC)',
    canadian_fwi_utils.DMC_NAME: 'duff moisture code (DMC)',
    canadian_fwi_utils.FFMC_NAME: 'fine-fuel moisture code (FFMC)',
    canadian_fwi_utils.ISI_NAME: 'initial-spread index (ISI)',
    canadian_fwi_utils.BUI_NAME: 'build-up index (BUI)',
    canadian_fwi_utils.FWI_NAME: 'fire-weather index (FWI)',
    canadian_fwi_utils.DSR_NAME: 'daily severity rating (DSR)'
}

RMSE_KEY = 'root_mean_squared_error'

METRIC_NAME_TO_VERBOSE = {
    regression_eval.TARGET_STDEV_KEY: 'stdev of actual value',
    regression_eval.PREDICTION_STDEV_KEY: 'stdev of prediction',
    regression_eval.TARGET_MEAN_KEY: 'mean actual value',
    regression_eval.PREDICTION_MEAN_KEY: 'mean prediction',
    RMSE_KEY: 'RMSE',
    regression_eval.MSE_BIAS_KEY: 'bias part of MSE',
    regression_eval.MSE_VARIANCE_KEY: 'variance part of MSE',
    regression_eval.MSE_SKILL_SCORE_KEY: 'MSE skill score',
    regression_eval.DWMSE_KEY: 'DWMSE',
    regression_eval.DWMSE_SKILL_SCORE_KEY: 'DWMSE skill score',
    regression_eval.KS_STATISTIC_KEY: 'K-S statistic',
    regression_eval.KS_P_VALUE_KEY: 'K-S p-value',
    regression_eval.MAE_KEY: 'MAE',
    regression_eval.MAE_SKILL_SCORE_KEY: 'MAE skill score',
    regression_eval.BIAS_KEY: 'bias',
    regression_eval.CORRELATION_KEY: 'correlation',
    regression_eval.KGE_KEY: 'Kling-Gupta efficiency',
    regression_eval.RELIABILITY_KEY: 'reliability'
}

METRIC_NAME_TO_COLOUR_MAP_OBJECT = {
    regression_eval.TARGET_STDEV_KEY: pyplot.get_cmap('cividis'),
    regression_eval.PREDICTION_STDEV_KEY: pyplot.get_cmap('cividis'),
    regression_eval.TARGET_MEAN_KEY: pyplot.get_cmap('cividis'),
    regression_eval.PREDICTION_MEAN_KEY: pyplot.get_cmap('cividis'),
    RMSE_KEY: pyplot.get_cmap('viridis'),
    regression_eval.MSE_BIAS_KEY: pyplot.get_cmap('viridis'),
    regression_eval.MSE_VARIANCE_KEY: pyplot.get_cmap('viridis'),
    regression_eval.MSE_SKILL_SCORE_KEY: pyplot.get_cmap('seismic'),
    regression_eval.DWMSE_KEY: pyplot.get_cmap('viridis'),
    regression_eval.DWMSE_SKILL_SCORE_KEY: pyplot.get_cmap('seismic'),
    regression_eval.KS_STATISTIC_KEY: pyplot.get_cmap('viridis'),
    regression_eval.KS_P_VALUE_KEY: pyplot.get_cmap('viridis'),
    regression_eval.MAE_KEY: pyplot.get_cmap('viridis'),
    regression_eval.MAE_SKILL_SCORE_KEY: pyplot.get_cmap('seismic'),
    regression_eval.BIAS_KEY: pyplot.get_cmap('seismic'),
    regression_eval.CORRELATION_KEY: pyplot.get_cmap('seismic'),
    regression_eval.KGE_KEY: pyplot.get_cmap('seismic'),
    regression_eval.RELIABILITY_KEY: pyplot.get_cmap('viridis')
}

METRIC_NAME_TO_COLOUR_NORM_TYPE_STRING = {
    regression_eval.TARGET_STDEV_KEY: 'sequential',
    regression_eval.PREDICTION_STDEV_KEY: 'sequential',
    regression_eval.TARGET_MEAN_KEY: 'sequential',
    regression_eval.PREDICTION_MEAN_KEY: 'sequential',
    RMSE_KEY: 'sequential',
    regression_eval.MSE_BIAS_KEY: 'sequential',
    regression_eval.MSE_VARIANCE_KEY: 'sequential',
    regression_eval.MSE_SKILL_SCORE_KEY: 'skill_score',
    regression_eval.DWMSE_KEY: 'sequential',
    regression_eval.DWMSE_SKILL_SCORE_KEY: 'skill_score',
    regression_eval.KS_STATISTIC_KEY: 'sequential',
    regression_eval.KS_P_VALUE_KEY: 'sequential',
    regression_eval.MAE_KEY: 'sequential',
    regression_eval.MAE_SKILL_SCORE_KEY: 'skill_score',
    regression_eval.BIAS_KEY: 'diverging',
    regression_eval.CORRELATION_KEY: 'diverging',
    regression_eval.KGE_KEY: 'skill_score',
    regression_eval.RELIABILITY_KEY: 'sequential'
}

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_evaluation_file_name'
MIN_COLOUR_PERCENTILE_ARG_NAME = 'min_colour_percentile'
MAX_COLOUR_PERCENTILE_ARG_NAME = 'max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `regression_evaluation.read_file`).'
)
MIN_COLOUR_PERCENTILE_HELP_STRING = (
    'Percentile (in range 0...100) used to determine minimum value in colour '
    'bar.'
)
MAX_COLOUR_PERCENTILE_HELP_STRING = (
    'Percentile (in range 0...100) used to determine max value in colour bar.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_COLOUR_PERCENTILE_ARG_NAME, type=float, required=True,
    help=MIN_COLOUR_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_COLOUR_PERCENTILE_ARG_NAME, type=float, required=True,
    help=MAX_COLOUR_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_score(
        score_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e,
        colour_map_object, colour_norm_object, title_string, output_file_name):
    """Plots one score.

    M = number of rows in grid
    N = number of columns in grid
    P = number of points in border file

    :param score_matrix: M-by-N numpy array of data values.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg
        north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg
        east).
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap` or similar).
    :param colour_norm_object: Colour-normalizer, used to map from physical
        values to colours (instance of `matplotlib.colors.BoundaryNorm` or
        similar).
    :param title_string: Title.
    :param output_file_name: Path to output file.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    is_longitude_positive_in_west = gfs_plotting.plot_field(
        data_matrix=score_matrix,
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

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(input_file_name, min_colour_percentile, max_colour_percentile,
         output_dir_name):
    """Plots gridded model evaluation.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param min_colour_percentile: Same.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if the input file contains ungridded, rather than
        gridded, evaluation.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_geq(min_colour_percentile, 0.)
    error_checking.assert_is_leq(min_colour_percentile, 10.)
    error_checking.assert_is_geq(max_colour_percentile, 90.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)

    # Do actual stuff.
    print('Reading data from: "{0:s}"...'.format(input_file_name))
    evaluation_table_xarray = regression_eval.read_file(input_file_name)

    if regression_eval.LATITUDE_DIM not in evaluation_table_xarray.coords:
        error_string = (
            'File "{0:s}" contains ungridded evaluation.  This script '
            'handles only gridded evaluation.'
        ).format(
            input_file_name
        )

        raise ValueError(error_string)

    model_file_name = (
        evaluation_table_xarray.attrs[regression_eval.MODEL_FILE_KEY]
    )
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[
        neural_net.TRAINING_OPTIONS_KEY
    ]
    target_field_name = generator_option_dict[neural_net.TARGET_FIELD_KEY]

    etx = evaluation_table_xarray
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    for this_metric_name in list(METRIC_NAME_TO_VERBOSE.keys()):
        this_score_matrix = numpy.nanmean(etx[this_metric_name].values, axis=-1)
        colour_norm_type_string = METRIC_NAME_TO_COLOUR_NORM_TYPE_STRING[
            this_metric_name
        ]

        if colour_norm_type_string == 'sequential':
            min_colour_value = numpy.nanpercentile(
                this_score_matrix, min_colour_percentile
            )
            max_colour_value = numpy.nanpercentile(
                this_score_matrix, max_colour_percentile
            )
        elif colour_norm_type_string == 'diverging':
            max_colour_value = numpy.nanpercentile(
                numpy.absolute(this_score_matrix), max_colour_percentile
            )
            min_colour_value = -1 * max_colour_value
        else:
            max_colour_value = numpy.nanpercentile(
                this_score_matrix, max_colour_percentile
            )
            min_colour_value = -1 * max_colour_value

        if numpy.isnan(max_colour_value):
            min_colour_value = 0.
            max_colour_value = 1.

        max_colour_value = max([
            max_colour_value,
            min_colour_value + TOLERANCE
        ])
        colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )

        title_string = (
            '{0:s}{1:s} for {2:s}\nMin/avg/max = {3:f}/{4:f}/{5:f}'
        ).format(
            METRIC_NAME_TO_VERBOSE[this_metric_name][0].upper(),
            METRIC_NAME_TO_VERBOSE[this_metric_name][1:],
            TARGET_FIELD_NAME_TO_VERBOSE[target_field_name],
            numpy.nanmin(this_score_matrix),
            numpy.nanmean(this_score_matrix),
            numpy.nanmax(this_score_matrix),
        )

        output_file_name = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, this_metric_name
        )

        _plot_one_score(
            score_matrix=this_score_matrix,
            grid_latitudes_deg_n=
            etx.coords[regression_eval.LATITUDE_DIM].values,
            grid_longitudes_deg_e=
            etx.coords[regression_eval.LONGITUDE_DIM].values,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            colour_map_object=
            METRIC_NAME_TO_COLOUR_MAP_OBJECT[this_metric_name],
            colour_norm_object=colour_norm_object,
            title_string=title_string,
            output_file_name=output_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        min_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MIN_COLOUR_PERCENTILE_ARG_NAME
        ),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_COLOUR_PERCENTILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
