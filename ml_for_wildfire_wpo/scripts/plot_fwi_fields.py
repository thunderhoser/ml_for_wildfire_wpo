"""Plots FWI fields."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.io import border_io
from ml_for_wildfire_wpo.io import canadian_fwi_io
from ml_for_wildfire_wpo.utils import canadian_fwi_utils
from ml_for_wildfire_wpo.plotting import fwi_plotting
from ml_for_wildfire_wpo.plotting import plotting_utils

DATE_FORMAT_FOR_TITLE = '%Y-%m-%d'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIR_ARG_NAME = 'input_fwi_dir_name'
VALID_DATE_ARG_NAME = 'valid_date_string'
FIELD_NAMES_ARG_NAME = 'field_names'
EXTREME_THRESHOLDS_ARG_NAME = 'extreme_value_threshold_by_field'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing Canadian FWI data.  Files therein '
    'will be found by `canadian_fwi_io.find_file` and read by '
    '`canadian_fwi_io.read_file`.'
)
VALID_DATE_HELP_STRING = (
    'Valid date (format "yyyymmdd").  Will plot fields valid at local noon on '
    'this date.'
)
FIELD_NAMES_HELP_STRING = (
    'List of fields to plot.  Each field name must be accepted by '
    '`canadian_fwi_utils.check_field_name`.'
)
EXTREME_THRESHOLDS_HELP_STRING = (
    'List of extreme-value thresholds, one per field.  Grid points containing '
    'extreme values will be marked with stippling.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_DATE_ARG_NAME, type=str, required=True,
    help=VALID_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIELD_NAMES_ARG_NAME, type=str, nargs='+', required=True,
    help=FIELD_NAMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXTREME_THRESHOLDS_ARG_NAME, type=float, nargs='+', required=True,
    help=EXTREME_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_field(
        data_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e,
        field_name, extreme_value_threshold, title_string, output_file_name):
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
    :param extreme_value_threshold: Extreme-value threshold.
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

    row_indices, column_indices = numpy.where(
        data_matrix >= extreme_value_threshold
    )

    axes_object.plot(
        grid_longitudes_deg_e[column_indices],
        grid_latitudes_deg_n[row_indices],
        linestyle='None',
        marker='*', markersize=12, markeredgewidth=1,
        markerfacecolor=numpy.full(3, 1.), markeredgecolor=numpy.full(3, 0.)
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


def _run(fwi_dir_name, valid_date_string, field_names,
         extreme_value_threshold_by_field, output_dir_name):
    """Plots FWI fields.

    This is essentially the main method.

    :param fwi_dir_name: See documentation at top of this script.
    :param valid_date_string: Same.
    :param field_names: Same.
    :param extreme_value_threshold_by_field: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    assert all([f in canadian_fwi_utils.ALL_FIELD_NAMES for f in field_names])

    valid_date_unix_sec = time_conversion.string_to_unix_sec(
        valid_date_string, canadian_fwi_io.DATE_FORMAT
    )

    num_fields = len(field_names)
    error_checking.assert_is_numpy_array(
        extreme_value_threshold_by_field,
        exact_dimensions=numpy.array([num_fields], dtype=int)
    )

    # Do actual stuff.
    fwi_file_name = canadian_fwi_io.find_file(
        directory_name=fwi_dir_name,
        valid_date_string=valid_date_string,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(fwi_file_name))
    fwi_table_xarray = canadian_fwi_io.read_file(fwi_file_name)
    fwit = fwi_table_xarray

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    for j in range(len(field_names)):
        this_field_name_fancy = fwi_plotting.FIELD_NAME_TO_FANCY[
            field_names[j]
        ]
        data_matrix = canadian_fwi_utils.get_field(
            fwi_table_xarray=fwi_table_xarray,
            field_name=field_names[j]
        )

        title_string = '{0:s}{1:s}\nValid local noon {2:s}'.format(
            this_field_name_fancy[0].upper(),
            this_field_name_fancy[1:],
            time_conversion.unix_sec_to_string(
                valid_date_unix_sec, DATE_FORMAT_FOR_TITLE
            )
        )

        output_file_name = '{0:s}/{1:s}_{2:s}.jpg'.format(
            output_dir_name,
            field_names[j].replace('_', '-'),
            valid_date_string
        )

        _plot_one_field(
            data_matrix=data_matrix,
            grid_latitudes_deg_n=
            fwit.coords[canadian_fwi_utils.LATITUDE_DIM].values,
            grid_longitudes_deg_e=
            fwit.coords[canadian_fwi_utils.LONGITUDE_DIM].values,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            field_name=field_names[j],
            extreme_value_threshold=extreme_value_threshold_by_field[j],
            title_string=title_string,
            output_file_name=output_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        fwi_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        valid_date_string=getattr(INPUT_ARG_OBJECT, VALID_DATE_ARG_NAME),
        field_names=getattr(INPUT_ARG_OBJECT, FIELD_NAMES_ARG_NAME),
        extreme_value_threshold_by_field=numpy.array(
            getattr(INPUT_ARG_OBJECT, EXTREME_THRESHOLDS_ARG_NAME), dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
