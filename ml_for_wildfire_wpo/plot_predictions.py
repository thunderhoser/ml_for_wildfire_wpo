"""Plots NN-based predictions and targets (actual values)."""

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
import time_conversion
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import border_io
import prediction_io
import canadian_fwi_utils
import neural_net
import bias_correction
import fwi_plotting
import plotting_utils

# TODO(thunderhoser): The ensemble_percentile stuff is HACKY.  I need to make
# this code nicer.

TOLERANCE = 1e-6
SENTINEL_VALUE = -1e6

DAYS_TO_SECONDS = 86400
DATE_FORMAT = '%Y%m%d'
DATE_FORMAT_FOR_TITLE = '%Y-%m-%d'

FIELD_NAME_TO_FANCY = {
    canadian_fwi_utils.FFMC_NAME: 'FFMC',
    canadian_fwi_utils.DMC_NAME: 'DMC',
    canadian_fwi_utils.DC_NAME: 'DC',
    canadian_fwi_utils.BUI_NAME: 'BUI',
    canadian_fwi_utils.ISI_NAME: 'ISI',
    canadian_fwi_utils.FWI_NAME: 'FWI',
    canadian_fwi_utils.DSR_NAME: 'DSR'
}

MASK_PIXEL_IF_WEIGHT_BELOW = 0.05

DIFF_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIRS_ARG_NAME = 'input_prediction_dir_name_by_model'
MODEL_DESCRIPTIONS_ARG_NAME = 'description_string_by_model'
ENSEMBLE_PERCENTILES_ARG_NAME = 'ensemble_percentiles'
FIELDS_ARG_NAME = 'field_names'
INIT_DATE_ARG_NAME = 'init_date_string'
MAX_VALUES_ARG_NAME = 'max_diff_value_by_field'
MAX_PERCENTILES_ARG_NAME = 'max_diff_prctile_by_field'
ISOTONIC_MODEL_FILE_ARG_NAME = 'isotonic_model_file_name'
UNCERTAINTY_CALIB_MODEL_FILE_ARG_NAME = 'uncertainty_calib_model_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIRS_HELP_STRING = (
    'List of paths to input directories, one per model.  Files in each '
    'directory will be found by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.'
)
MODEL_DESCRIPTIONS_HELP_STRING = (
    'List of model descriptions, to be used in figure titles.  The list must '
    'be space-separated.  Within each list item, underscores will be replaced '
    'by spaces.  This list must have the same length as {0:s}.'
).format(
    INPUT_DIRS_ARG_NAME
)
ENSEMBLE_PERCENTILES_HELP_STRING = (
    '1-D list of percentiles (ranging from 0...100).  For every model that '
    'outputs an ensemble, this script will plot the given percentiles of the '
    'ensemble distribution.  If you just want to plot the ensemble mean, leave '
    'this argument alone.'
)
FIELDS_HELP_STRING = 'List of fields to plot.'
INIT_DATE_HELP_STRING = (
    'Initialization date (i.e., model-run date, where the model is always run '
    'at 00Z).  Format should be "yyyymmdd".'
)
MAX_VALUES_HELP_STRING = (
    'List of max absolute diff values in colour bar.  This list must have the '
    'same length as {0:s}.  If you instead want max absolute diffs to be '
    'percentiles over the data, leave this argument alone and use {1:s}.'
).format(
    FIELDS_ARG_NAME, MAX_PERCENTILES_ARG_NAME
)
MAX_PERCENTILES_HELP_STRING = (
    'List of percentile levels (from 0...100) used to determine max absolute '
    'diff in colour bar.  This list must have the same length as {0:s}.  If '
    'you instead want to specify raw values, leave this argument alone and use '
    '{1:s}.'
).format(
    FIELDS_ARG_NAME, MAX_VALUES_ARG_NAME
)
ISOTONIC_MODEL_FILE_HELP_STRING = (
    'Path to file with isotonic-regression model, which will be used to bias-'
    'correct ensemble means before plotting.  If you do not want IR, leave '
    'this argument alone.'
)
UNCERTAINTY_CALIB_MODEL_FILE_HELP_STRING = (
    'Path to file with uncertainty-calibration model, which will be used to '
    'bias-correct ensemble spreads before plotting.  If you do not want UC, '
    'leave this argument alone.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIRS_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_DIRS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_DESCRIPTIONS_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_DIRS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ENSEMBLE_PERCENTILES_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1], help=ENSEMBLE_PERCENTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_DATE_ARG_NAME, type=str, required=True,
    help=INIT_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_VALUES_ARG_NAME, type=float, nargs='+', required=False,
    default=[SENTINEL_VALUE - 1], help=MAX_VALUES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILES_ARG_NAME, type=float, nargs='+',
    required=False, default=[SENTINEL_VALUE - 1],
    help=MAX_PERCENTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ISOTONIC_MODEL_FILE_ARG_NAME, type=str, required=False, default='',
    help=ISOTONIC_MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + UNCERTAINTY_CALIB_MODEL_FILE_ARG_NAME, type=str, required=False,
    default='', help=UNCERTAINTY_CALIB_MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _find_lead_time(prediction_file_name_by_model):
    """Finds lead time (must be the same for all models).

    :param prediction_file_name_by_model: See documentation at top of this
        script.
    :return: lead_time_days: Lead time.
    """

    num_models = len(prediction_file_name_by_model)
    lead_time_days = None

    for i in range(num_models):
        prediction_table_xarray = prediction_io.read_file(
            prediction_file_name_by_model[i]
        )
        model_file_name = (
            prediction_table_xarray.attrs[prediction_io.MODEL_FILE_KEY]
        )
        model_metafile_name = neural_net.find_metafile(
            model_file_name=model_file_name, raise_error_if_missing=True
        )

        print('Reading model metadata from: "{0:s}"...'.format(
            model_metafile_name
        ))
        mmd = neural_net.read_metafile(model_metafile_name)
        training_option_dict = mmd[neural_net.TRAINING_OPTIONS_KEY]

        try:
            this_lead_time_days = training_option_dict['target_lead_time_days']
        except KeyError:
            this_lead_time_days = -1

            for this_subdir_name in prediction_file_name_by_model[i].split('/'):
                if 'lead-time-days' in this_subdir_name:
                    this_lead_time_days = int(
                        this_subdir_name.split('lead-time-days=')[-1]
                    )
                    break

                if 'day_lead' in this_subdir_name:
                    this_lead_time_days = int(
                        this_subdir_name.split('day_lead')[0]
                    )
                    break

        if lead_time_days is None:
            lead_time_days = this_lead_time_days + 0
            continue

        assert lead_time_days == this_lead_time_days

    return lead_time_days


def _plot_predictions_one_model(
        prediction_file_name, field_names, ensemble_percentiles,
        init_date_string, lead_time_days, max_diff_value_by_field,
        border_latitudes_deg_n, border_longitudes_deg_e,
        model_description_string, fancy_model_description_string,
        plot_actual, output_dir_name, isotonic_model_dict=None,
        uncertainty_calib_model_dict=None):
    """Plots all predictions (i.e., all predicted fields) for one model.

    P = number of points in borders

    :param prediction_file_name: Path to input file (will be read by
        `prediction_io.read_file`).
    :param field_names: 1-D list of target fields.
    :param ensemble_percentiles: See documentation at top of this script.
    :param init_date_string: Initialization date (format "yyyymmdd").
    :param lead_time_days: Lead time.
    :param max_diff_value_by_field: See documentation at top of this script.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg
        north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg
        east).
    :param model_description_string: Model description for use in file name.
    :param fancy_model_description_string: Model description for use in figure
        title.
    :param plot_actual: Boolean flag.  If True, will plot actual (target)
        fields.
    :param output_dir_name: Path to output directory.  Figures will be saved
        here.
    :param isotonic_model_dict: Dictionary returned by
        `bias_corection.read_file`.  If you do not want to bias-correct ensemble
        means, make this None.
    :param uncertainty_calib_model_dict: Dictionary returned by
        `bias_corection.read_file`.  If you do not want to bias-correct ensemble
        spreads, make this None.
    """

    init_date_unix_sec = time_conversion.string_to_unix_sec(
        init_date_string, DATE_FORMAT
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_table_xarray = prediction_io.read_file(prediction_file_name)

    if isotonic_model_dict is not None:
        prediction_table_xarray = bias_correction.apply_model_suite(
            prediction_table_xarray=prediction_table_xarray,
            model_dict=isotonic_model_dict
        )
    if uncertainty_calib_model_dict is not None:
        prediction_table_xarray = bias_correction.apply_model_suite(
            prediction_table_xarray=prediction_table_xarray,
            model_dict=uncertainty_calib_model_dict
        )

    ptx = prediction_table_xarray

    for j in range(len(field_names)):
        this_field_name_fancy = FIELD_NAME_TO_FANCY[field_names[j]]
        field_index = numpy.where(
            ptx[prediction_io.FIELD_NAME_KEY].values == field_names[j]
        )[0][0]

        full_prediction_matrix = (
            ptx[prediction_io.PREDICTION_KEY].values[..., field_index, :]
        )
        mean_prediction_matrix = numpy.mean(full_prediction_matrix, axis=-1)
        target_matrix = (
            ptx[prediction_io.TARGET_KEY].values[..., field_index]
        )
        weight_matrix = ptx[prediction_io.WEIGHT_KEY].values

        # TODO(thunderhoser): Hack to get rid of Canada.
        (
            latitude_matrix_deg_n, longitude_matrix_deg_e
        ) = grids.latlng_vectors_to_matrices(
            unique_latitudes_deg=ptx[prediction_io.LATITUDE_KEY].values,
            unique_longitudes_deg=lng_conversion.convert_lng_positive_in_west(
                ptx[prediction_io.LONGITUDE_KEY].values
            )
        )

        bad_flag_matrix = numpy.logical_and(
            latitude_matrix_deg_n > 50.,
            longitude_matrix_deg_e > 221.
        )
        weight_matrix[bad_flag_matrix] = -1.

        bad_flag_matrix = numpy.logical_and(
            latitude_matrix_deg_n > 59.5,
            longitude_matrix_deg_e < 192.
        )
        weight_matrix[bad_flag_matrix] = -1.

        bad_flag_matrix = latitude_matrix_deg_n < 23.5
        weight_matrix[bad_flag_matrix] = -1.

        # full_prediction_matrix[weight_matrix < MASK_PIXEL_IF_WEIGHT_BELOW] = (
        #     numpy.nan
        # )
        mean_prediction_matrix[weight_matrix < MASK_PIXEL_IF_WEIGHT_BELOW] = (
            numpy.nan
        )
        target_matrix[weight_matrix < MASK_PIXEL_IF_WEIGHT_BELOW] = numpy.nan

        title_string = (
            'Predicted {0:s} from {1:s}\n'
            'Init 00Z {2:s}, valid local noon {3:s}'
        ).format(
            this_field_name_fancy,
            fancy_model_description_string,
            time_conversion.unix_sec_to_string(
                init_date_unix_sec, DATE_FORMAT_FOR_TITLE
            ),
            time_conversion.unix_sec_to_string(
                init_date_unix_sec + lead_time_days * DAYS_TO_SECONDS,
                DATE_FORMAT_FOR_TITLE
            )
        )

        valid_date_string = time_conversion.unix_sec_to_string(
            init_date_unix_sec + lead_time_days * DAYS_TO_SECONDS,
            DATE_FORMAT
        )

        output_file_name = (
            '{0:s}/valid={1:s}_{2:s}_init={3:s}_{4:s}.jpg'
        ).format(
            output_dir_name,
            valid_date_string,
            field_names[j].replace('_', '-'),
            init_date_string,
            model_description_string
        )

        _plot_one_field(
            data_matrix=mean_prediction_matrix,
            grid_latitudes_deg_n=ptx[prediction_io.LATITUDE_KEY].values,
            grid_longitudes_deg_e=ptx[prediction_io.LONGITUDE_KEY].values,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            field_name=field_names[j],
            title_string=title_string,
            output_file_name=output_file_name
        )

        for this_percentile in ensemble_percentiles:
            title_string = (
                '{0:.1f}th-percentile {1:s} from {2:s}\n'
                'Init 00Z {3:s}, valid local noon {4:s}'
            ).format(
                this_percentile,
                this_field_name_fancy,
                fancy_model_description_string,
                time_conversion.unix_sec_to_string(
                    init_date_unix_sec, DATE_FORMAT_FOR_TITLE
                ),
                time_conversion.unix_sec_to_string(
                    init_date_unix_sec + lead_time_days * DAYS_TO_SECONDS,
                    DATE_FORMAT_FOR_TITLE
                )
            )

            output_file_name = (
                '{0:s}/valid={1:s}_{2:s}_init={3:s}_{4:s}_percentile{5:05.1f}.jpg'
            ).format(
                output_dir_name,
                valid_date_string,
                field_names[j].replace('_', '-'),
                init_date_string,
                model_description_string,
                this_percentile
            )

            this_prediction_matrix = numpy.percentile(
                full_prediction_matrix, this_percentile, axis=-1
            )
            this_prediction_matrix[weight_matrix < MASK_PIXEL_IF_WEIGHT_BELOW] = (
                numpy.nan
            )

            _plot_one_field(
                data_matrix=this_prediction_matrix,
                grid_latitudes_deg_n=ptx[prediction_io.LATITUDE_KEY].values,
                grid_longitudes_deg_e=ptx[prediction_io.LONGITUDE_KEY].values,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                field_name=field_names[j],
                title_string=title_string,
                output_file_name=output_file_name
            )

        title_string = (
            '{0:s} errors from {1:s}\n'
            'Init 00Z {2:s}, valid local noon {3:s}'
        ).format(
            this_field_name_fancy,
            fancy_model_description_string,
            time_conversion.unix_sec_to_string(
                init_date_unix_sec, DATE_FORMAT_FOR_TITLE
            ),
            time_conversion.unix_sec_to_string(
                init_date_unix_sec + lead_time_days * DAYS_TO_SECONDS,
                DATE_FORMAT_FOR_TITLE
            )
        )

        output_file_name = (
            '{0:s}/valid={1:s}_{2:s}_init={3:s}_{4:s}_diffs.jpg'
        ).format(
            output_dir_name,
            valid_date_string,
            field_names[j].replace('_', '-'),
            init_date_string,
            model_description_string
        )

        colour_norm_object = pyplot.Normalize(
            vmin=-1 * max_diff_value_by_field[j],
            vmax=max_diff_value_by_field[j]
        )

        _plot_one_diff_field(
            diff_matrix=mean_prediction_matrix - target_matrix,
            grid_latitudes_deg_n=ptx[prediction_io.LATITUDE_KEY].values,
            grid_longitudes_deg_e=ptx[prediction_io.LONGITUDE_KEY].values,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            colour_map_object=DIFF_COLOUR_MAP_OBJECT,
            colour_norm_object=colour_norm_object,
            title_string=title_string,
            output_file_name=output_file_name
        )

        if not plot_actual:
            continue

        title_string = (
            'Actual {0:s}\n'
            'Valid local noon {1:s}'
        ).format(
            this_field_name_fancy,
            time_conversion.unix_sec_to_string(
                init_date_unix_sec + lead_time_days * DAYS_TO_SECONDS,
                DATE_FORMAT_FOR_TITLE
            )
        )

        output_file_name = (
            '{0:s}/valid={1:s}_{2:s}_actual.jpg'
        ).format(
            output_dir_name,
            valid_date_string,
            field_names[j].replace('_', '-')
        )

        _plot_one_field(
            data_matrix=target_matrix,
            grid_latitudes_deg_n=ptx[prediction_io.LATITUDE_KEY].values,
            grid_longitudes_deg_e=ptx[prediction_io.LONGITUDE_KEY].values,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            field_name=field_names[j],
            title_string=title_string,
            output_file_name=output_file_name
        )


def _plot_one_diff_field(
        diff_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e,
        colour_map_object, colour_norm_object, title_string, output_file_name):
    """Plots one diff field.

    M = number of rows in grid
    N = number of columns in grid

    :param diff_matrix: M-by-N numpy array of differences (predicted minus
        actual).
    :param grid_latitudes_deg_n: See documentation for `_plot_one_field`.
    :param grid_longitudes_deg_e: Same.
    :param border_latitudes_deg_n: Same.
    :param border_longitudes_deg_e: Same.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap` or similar).
    :param colour_norm_object: Colour-normalizer, used to map from physical
        values to colours (instance of `matplotlib.colors.BoundaryNorm` or
        similar).
    :param title_string: See documentation for `_plot_one_field`.
    :param output_file_name: Same.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    is_longitude_positive_in_west = fwi_plotting.plot_field(
        data_matrix=diff_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        axes_object=axes_object,
        plot_colour_bar=True
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


def _plot_one_field(
        data_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e,
        field_name, title_string, output_file_name):
    """Plots one field.

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
    :param output_file_name: Path to output file.
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


def _run(prediction_dir_name_by_model, description_string_by_model,
         ensemble_percentiles, field_names, init_date_string,
         max_diff_value_by_field, max_diff_percentile_by_field,
         isotonic_model_file_name, uncertainty_calib_model_file_name,
         output_dir_name):
    """Plots NN-based predictions and targets (actual values).

    This is effectively the main method.

    :param prediction_dir_name_by_model: See documentation at top of file.
    :param description_string_by_model: Same.
    :param ensemble_percentiles: Same.
    :param field_names: Same.
    :param init_date_string: Same.
    :param max_diff_value_by_field: Same.
    :param max_diff_percentile_by_field: Same.
    :param isotonic_model_file_name: Same.
    :param uncertainty_calib_model_file_name: Same.
    :param output_dir_name: Same.
    """

    # Check input args for colour scheme.
    if (
            (len(max_diff_value_by_field) == 1 and
             max_diff_value_by_field[0] <= SENTINEL_VALUE)
    ):
        max_diff_value_by_field = None

    if (
            (len(max_diff_percentile_by_field) == 1 and
             max_diff_percentile_by_field[0] <= SENTINEL_VALUE)
    ):
        max_diff_percentile_by_field = None

    num_fields = len(field_names)
    expected_dim = numpy.array([num_fields], dtype=int)

    if max_diff_value_by_field is None:
        error_checking.assert_is_numpy_array(
            max_diff_percentile_by_field, exact_dimensions=expected_dim
        )
        error_checking.assert_is_geq_numpy_array(
            max_diff_percentile_by_field, 90.
        )
        error_checking.assert_is_leq_numpy_array(
            max_diff_percentile_by_field, 100.
        )
    else:
        error_checking.assert_is_numpy_array(
            max_diff_value_by_field, exact_dimensions=expected_dim
        )
        error_checking.assert_is_greater_numpy_array(
            max_diff_value_by_field, 0.
        )

    # Check other input args.
    if isotonic_model_file_name == '':
        isotonic_model_file_name = None
    if uncertainty_calib_model_file_name == '':
        uncertainty_calib_model_file_name = None

    if len(ensemble_percentiles) == 1 and ensemble_percentiles[0] < 0:
        ensemble_percentiles = numpy.array([])

    if len(ensemble_percentiles) > 0:
        error_checking.assert_is_geq_numpy_array(ensemble_percentiles, 0.)
        error_checking.assert_is_leq_numpy_array(ensemble_percentiles, 100.)

    num_models = len(prediction_dir_name_by_model)
    error_checking.assert_is_numpy_array(
        numpy.array(description_string_by_model),
        exact_dimensions=numpy.array([num_models], dtype=int)
    )

    fancy_description_string_by_model = [
        d.replace('_', ' ') for d in description_string_by_model
    ]
    description_string_by_model = [
        d.lower().replace('_', '-') for d in description_string_by_model
    ]

    assert all([f in canadian_fwi_utils.ALL_FIELD_NAMES for f in field_names])

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    prediction_file_name_by_model = [
        prediction_io.find_file(
            directory_name=d,
            init_date_string=init_date_string,
            raise_error_if_missing=True
        )
        for d in prediction_dir_name_by_model
    ]
    lead_time_days = _find_lead_time(prediction_file_name_by_model)

    if isotonic_model_file_name is None:
        isotonic_model_dict = None
    else:
        print('Reading isotonic-regression model from: "{0:s}"...'.format(
            isotonic_model_file_name
        ))
        isotonic_model_dict = bias_correction.read_file(
            isotonic_model_file_name
        )
        assert not isotonic_model_dict[bias_correction.DO_UNCERTAINTY_CALIB_KEY]

    if uncertainty_calib_model_file_name is None:
        uncertainty_calib_model_dict = None
    else:
        print('Reading uncertainty-calibration model from: "{0:s}"...'.format(
            uncertainty_calib_model_file_name
        ))
        uncertainty_calib_model_dict = bias_correction.read_file(
            uncertainty_calib_model_file_name
        )
        ucmd = uncertainty_calib_model_dict

        assert ucmd[bias_correction.DO_UNCERTAINTY_CALIB_KEY]
        assert (
            ucmd[bias_correction.DO_IR_BEFORE_UC_KEY] ==
            (isotonic_model_file_name is not None)
        )

    if max_diff_value_by_field is None:
        max_diff_value_by_field = numpy.full(num_fields, numpy.nan)

        for j in range(num_fields):
            this_field_diff_values = [None] * num_models

            for i in range(num_models):
                ptx = prediction_io.read_file(prediction_file_name_by_model[i])
                if isotonic_model_dict is not None:
                    ptx = bias_correction.apply_model_suite(
                        prediction_table_xarray=ptx,
                        model_dict=isotonic_model_dict
                    )
                if uncertainty_calib_model_dict is not None:
                    ptx = bias_correction.apply_model_suite(
                        prediction_table_xarray=ptx,
                        model_dict=uncertainty_calib_model_dict
                    )

                field_index = numpy.where(
                    ptx[prediction_io.FIELD_NAME_KEY].values == field_names[j]
                )[0][0]

                this_pred_matrix = numpy.mean(
                    ptx[prediction_io.PREDICTION_KEY].values[..., field_index, :],
                    axis=-1
                )
                this_target_matrix = (
                    ptx[prediction_io.TARGET_KEY].values[..., field_index]
                )
                this_weight_matrix = ptx[prediction_io.WEIGHT_KEY].values

                # TODO(thunderhoser): Hack to get rid of Canada.
                (
                    latitude_matrix_deg_n, longitude_matrix_deg_e
                ) = grids.latlng_vectors_to_matrices(
                    unique_latitudes_deg=ptx[prediction_io.LATITUDE_KEY].values,
                    unique_longitudes_deg=lng_conversion.convert_lng_positive_in_west(
                        ptx[prediction_io.LONGITUDE_KEY].values
                    )
                )

                bad_flag_matrix = numpy.logical_and(
                    latitude_matrix_deg_n > 50.,
                    longitude_matrix_deg_e > 221.
                )
                this_weight_matrix[bad_flag_matrix] = -1.

                bad_flag_matrix = numpy.logical_and(
                    latitude_matrix_deg_n > 59.5,
                    longitude_matrix_deg_e < 192.
                )
                this_weight_matrix[bad_flag_matrix] = -1.

                bad_flag_matrix = latitude_matrix_deg_n < 23.5
                this_weight_matrix[bad_flag_matrix] = -1.

                this_pred_matrix[
                    this_weight_matrix < MASK_PIXEL_IF_WEIGHT_BELOW
                ] = numpy.nan

                this_field_diff_values[i] = numpy.ravel(
                    this_pred_matrix - this_target_matrix
                )

            max_diff_value_by_field[j] = numpy.nanpercentile(
                numpy.concatenate(this_field_diff_values),
                max_diff_percentile_by_field[j]
            )

    for i in range(num_models):
        _plot_predictions_one_model(
            prediction_file_name=prediction_file_name_by_model[i],
            isotonic_model_dict=isotonic_model_dict,
            uncertainty_calib_model_dict=uncertainty_calib_model_dict,
            field_names=field_names,
            ensemble_percentiles=ensemble_percentiles,
            init_date_string=init_date_string,
            lead_time_days=lead_time_days,
            max_diff_value_by_field=max_diff_value_by_field,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            model_description_string=description_string_by_model[i],
            fancy_model_description_string=fancy_description_string_by_model[i],
            plot_actual=i == 0,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_name_by_model=getattr(
            INPUT_ARG_OBJECT, INPUT_DIRS_ARG_NAME
        ),
        description_string_by_model=getattr(
            INPUT_ARG_OBJECT, MODEL_DESCRIPTIONS_ARG_NAME
        ),
        ensemble_percentiles=numpy.array(
            getattr(INPUT_ARG_OBJECT, ENSEMBLE_PERCENTILES_ARG_NAME),
            dtype=float
        ),
        field_names=getattr(INPUT_ARG_OBJECT, FIELDS_ARG_NAME),
        init_date_string=getattr(INPUT_ARG_OBJECT, INIT_DATE_ARG_NAME),
        max_diff_value_by_field=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_VALUES_ARG_NAME), dtype=float
        ),
        max_diff_percentile_by_field=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_PERCENTILES_ARG_NAME), dtype=float
        ),
        isotonic_model_file_name=getattr(
            INPUT_ARG_OBJECT, ISOTONIC_MODEL_FILE_ARG_NAME
        ),
        uncertainty_calib_model_file_name=getattr(
            INPUT_ARG_OBJECT, UNCERTAINTY_CALIB_MODEL_FILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
