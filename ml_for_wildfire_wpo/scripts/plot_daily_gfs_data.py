"""Plots daily GFS data (model output)."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.io import border_io
from ml_for_wildfire_wpo.io import gfs_daily_io
from ml_for_wildfire_wpo.utils import gfs_daily_utils
from ml_for_wildfire_wpo.plotting import gfs_plotting
from ml_for_wildfire_wpo.scripts import plot_gfs_data

TOLERANCE = 1e-6
DATE_FORMAT = '%Y%m%d'

SEQUENTIAL_COLOUR_MAP_OBJECT = pyplot.get_cmap('viridis')
DIVERGING_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIR_ARG_NAME = 'input_dir_name'
FIELDS_ARG_NAME = 'field_names'
INIT_DATE_ARG_NAME = 'init_date_string'
LEAD_TIMES_ARG_NAME = 'lead_times_days'
MIN_COLOUR_PERCENTILE_ARG_NAME = 'min_colour_percentile'
MAX_COLOUR_PERCENTILE_ARG_NAME = 'max_colour_percentile'
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
MIN_COLOUR_PERCENTILE_HELP_STRING = (
    'The minimum value plotted in each map will be this percentile (from '
    '0...100) over the values in the map.'
)
MAX_COLOUR_PERCENTILE_HELP_STRING = 'Same as {0:s} but for max value.'.format(
    MIN_COLOUR_PERCENTILE_ARG_NAME
)
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
    '--' + INIT_DATE_ARG_NAME, type=str, nargs='+', required=True,
    help=INIT_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=LEAD_TIMES_HELP_STRING
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


def _run(input_dir_name, field_names, init_date_string, lead_times_days,
         min_colour_percentile, max_colour_percentile, output_dir_name):
    """Plots GFS data (model output).

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param field_names: Same.
    :param init_date_string: Same.
    :param lead_times_days: Same.
    :param min_colour_percentile: Same.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    for this_field_name in field_names:
        gfs_daily_utils.check_field_name(this_field_name)

    time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)

    error_checking.assert_is_integer_numpy_array(lead_times_days)
    error_checking.assert_is_greater_numpy_array(lead_times_days, 0)
    error_checking.assert_is_geq(min_colour_percentile, 0.)
    error_checking.assert_is_leq(min_colour_percentile, 10.)
    error_checking.assert_is_geq(max_colour_percentile, 90.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)

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
            dgfst.coords[gfs_daily_io.LEAD_TIME_DIM].values ==
            this_forecast_day
        )[0][0]

        for this_field_name in field_names:
            this_field_name_fancy = gfs_plotting.FIELD_NAME_TO_FANCY[
                this_field_name
            ]
            field_index = numpy.where(
                dgfst.coords[gfs_daily_io.FIELD_DIM].values == this_field_name
            )[0][0]

            data_matrix = dgfst[gfs_daily_io.DATA_KEY_2D].values[
                day_index, ..., field_index
            ]
            data_matrix, unit_string = gfs_plotting.field_to_plotting_units(
                data_matrix_default_units=data_matrix,
                field_name=this_field_name
            )

            title_string = '{0:s}{1:s} ({2:s})'.format(
                this_field_name_fancy[0].upper(),
                this_field_name_fancy[1:],
                unit_string
            )

            output_file_name = (
                '{0:s}/init={1:s}_forecast={2:02d}days_{3:s}.jpg'
            ).format(
                output_dir_name,
                init_date_string,
                this_forecast_day,
                this_field_name.replace('_', '-')
            )

            plot_gfs_data._plot_one_field(
                data_matrix=data_matrix,
                grid_latitudes_deg_n=
                dgfst.coords[gfs_daily_io.LATITUDE_DIM].values,
                grid_longitudes_deg_e=
                dgfst.coords[gfs_daily_io.LONGITUDE_DIM].values,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                field_name=this_field_name,
                min_colour_percentile=min_colour_percentile,
                max_colour_percentile=max_colour_percentile,
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
        min_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MIN_COLOUR_PERCENTILE_ARG_NAME
        ),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_COLOUR_PERCENTILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
