"""Plots GFS-based FWI forecasts."""

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

import time_conversion
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import border_io
import gfs_daily_io
import gfs_daily_utils
import fwi_plotting
import plotting_utils

TOLERANCE = 1e-6
DATE_FORMAT = '%Y%m%d'
DATE_FORMAT_FOR_TITLE = '%Y-%m-%d'

DAYS_TO_SECONDS = 86400

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIR_ARG_NAME = 'input_dir_name'
FIELDS_ARG_NAME = 'field_names'
INIT_DATE_ARG_NAME = 'init_date_string'
LEAD_TIMES_ARG_NAME = 'lead_times_days'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`gfs_daily_io.find_file` and read by `gfs_daily_io.read_file`.'
)
FIELDS_HELP_STRING = 'List of fields to plot.'
INIT_DATE_HELP_STRING = (
    'Initialization date (i.e., model-run date, where the model is always run '
    'at 00Z).  Format should be "yyyymmdd".'
)
LEAD_TIMES_HELP_STRING = 'List of lead times.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
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
    '--' + LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=LEAD_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


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


def _run(input_dir_name, field_names, init_date_string, lead_times_days,
         output_dir_name):
    """Plots GFS-based FWI forecasts.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param field_names: Same.
    :param init_date_string: Same.
    :param lead_times_days: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    assert all([f in gfs_daily_utils.ALL_FWI_FIELD_NAMES for f in field_names])

    init_date_unix_sec = time_conversion.string_to_unix_sec(
        init_date_string, DATE_FORMAT
    )

    error_checking.assert_is_integer_numpy_array(lead_times_days)
    error_checking.assert_is_greater_numpy_array(lead_times_days, 0)

    # Do actual stuff.
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    daily_gfs_file_name = gfs_daily_io.find_file(
        directory_name=input_dir_name, init_date_string=init_date_string,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(daily_gfs_file_name))
    daily_gfs_table_xarray = gfs_daily_io.read_file(daily_gfs_file_name)
    dgfst = daily_gfs_table_xarray

    for this_forecast_day in lead_times_days:
        day_index = numpy.where(
            dgfst.coords[gfs_daily_utils.LEAD_TIME_DIM].values ==
            this_forecast_day
        )[0][0]

        for this_field_name in field_names:
            this_field_name_fancy = fwi_plotting.FIELD_NAME_TO_FANCY[
                this_field_name
            ]
            field_index = numpy.where(
                dgfst.coords[gfs_daily_utils.FIELD_DIM].values ==
                this_field_name
            )[0][0]

            data_matrix = dgfst[gfs_daily_utils.DATA_KEY_2D].values[
                day_index, ..., field_index
            ]

            title_string = (
                '{0:s}{1:s}\nInit 00Z {2:s}, valid local noon {3:s}'
            ).format(
                this_field_name_fancy[0].upper(),
                this_field_name_fancy[1:],
                time_conversion.unix_sec_to_string(
                    init_date_unix_sec, DATE_FORMAT_FOR_TITLE
                ),
                time_conversion.unix_sec_to_string(
                    init_date_unix_sec + this_forecast_day * DAYS_TO_SECONDS,
                    DATE_FORMAT_FOR_TITLE
                )
            )

            output_file_name = (
                '{0:s}/init={1:s}_forecast={2:02d}days_{3:s}.jpg'
            ).format(
                output_dir_name,
                init_date_string,
                this_forecast_day,
                this_field_name.replace('_', '-')
            )

            _plot_one_field(
                data_matrix=data_matrix,
                grid_latitudes_deg_n=
                dgfst.coords[gfs_daily_utils.LATITUDE_DIM].values,
                grid_longitudes_deg_e=
                dgfst.coords[gfs_daily_utils.LONGITUDE_DIM].values,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                field_name=this_field_name,
                title_string=title_string,
                output_file_name=output_file_name
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        field_names=getattr(INPUT_ARG_OBJECT, FIELDS_ARG_NAME),
        init_date_string=getattr(INPUT_ARG_OBJECT, INIT_DATE_ARG_NAME),
        lead_times_days=numpy.array(
            getattr(INPUT_ARG_OBJECT, LEAD_TIMES_ARG_NAME), dtype=int
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
