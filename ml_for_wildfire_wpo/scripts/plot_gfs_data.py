"""Plots GFS data (model output)."""

import os
import warnings
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.io import gfs_io
from ml_for_wildfire_wpo.io import border_io
from ml_for_wildfire_wpo.utils import gfs_utils
from ml_for_wildfire_wpo.plotting import plotting_utils
from ml_for_wildfire_wpo.plotting import gfs_plotting

DATE_FORMAT = '%Y%m%d'

SEQUENTIAL_COLOUR_MAP_OBJECT = pyplot.get_cmap('viridis')
DIVERGING_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIR_ARG_NAME = 'input_dir_name'
FIELDS_ARG_NAME = 'field_names'
PRESSURE_LEVELS_ARG_NAME = 'pressure_levels_mb'
INIT_DATES_ARG_NAME = 'init_date_strings'
FORECAST_HOURS_ARG_NAME = 'forecast_hours'
MIN_COLOUR_PERCENTILE_ARG_NAME = 'min_colour_percentile'
MAX_COLOUR_PERCENTILE_ARG_NAME = 'max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`gfs_io.find_file` and read by `gfs_io.read_file`.'
)
FIELDS_HELP_STRING = 'List of fields to plot.'
PRESSURE_LEVELS_HELP_STRING = (
    'List of pressure levels.  Will be used only for 3-D fields.'
)
INIT_DATES_HELP_STRING = (
    'List of initialization dates (i.e., model-run dates, where the model is '
    'always run at 00Z).  Format of each date should be "yyyymmdd".'
)
FORECAST_HOURS_HELP_STRING = 'List of forecast hours (lead times).'
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
    '--' + PRESSURE_LEVELS_ARG_NAME, type=int, nargs='+', required=True,
    help=PRESSURE_LEVELS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_DATES_ARG_NAME, type=str, nargs='+', required=True,
    help=INIT_DATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FORECAST_HOURS_ARG_NAME, type=int, nargs='+', required=True,
    help=FORECAST_HOURS_HELP_STRING
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


def _plot_one_field(
        data_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e,
        field_name, min_colour_percentile, max_colour_percentile,
        title_string, output_file_name):
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
    :param min_colour_percentile: See documentation at top of script.
    :param max_colour_percentile: Same.
    :param title_string: Title.
    :param output_file_name: Path to output file.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if gfs_plotting.use_diverging_colour_scheme(field_name):
        colour_map_object = DIVERGING_COLOUR_MAP_OBJECT
        max_colour_value = numpy.nanpercentile(
            numpy.absolute(data_matrix), max_colour_percentile
        )
        min_colour_value = -1 * max_colour_value
    else:
        colour_map_object = SEQUENTIAL_COLOUR_MAP_OBJECT
        min_colour_value = numpy.nanpercentile(
            data_matrix, min_colour_percentile
        )
        max_colour_value = numpy.nanpercentile(
            data_matrix, max_colour_percentile
        )

    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )

    is_longitude_positive_in_west = gfs_plotting.plot_field(
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


def _run(input_dir_name, field_names, pressure_levels_mb, init_date_strings,
         forecast_hours, min_colour_percentile, max_colour_percentile,
         output_dir_name):
    """Plots GFS data (model output).

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param field_names: Same.
    :param pressure_levels_mb: Same.
    :param init_date_strings: Same.
    :param forecast_hours: Same.
    :param min_colour_percentile: Same.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    for this_field_name in field_names:
        gfs_utils.check_field_name(this_field_name)
    for this_date_string in init_date_strings:
        time_conversion.string_to_unix_sec(this_date_string, DATE_FORMAT)

    error_checking.assert_is_integer_numpy_array(pressure_levels_mb)
    error_checking.assert_is_greater_numpy_array(pressure_levels_mb, 0)
    error_checking.assert_is_integer_numpy_array(forecast_hours)
    error_checking.assert_is_geq_numpy_array(forecast_hours, 0)
    error_checking.assert_is_geq(min_colour_percentile, 0.)
    error_checking.assert_is_leq(min_colour_percentile, 10.)
    error_checking.assert_is_geq(max_colour_percentile, 90.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)

    # Do actual stuff.
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    for this_date_string in init_date_strings:
        gfs_file_name = gfs_io.find_file(
            directory_name=input_dir_name, init_date_string=this_date_string,
            raise_error_if_missing=False
        )

        if not os.path.isfile(gfs_file_name):
            warning_string = (
                'POTENTIAL ERROR.  Cannot find GFS file at expected location: '
                '"{0:s}"'
            ).format(gfs_file_name)

            warnings.warn(warning_string)
            continue

        print('Reading data from: "{0:s}"...'.format(gfs_file_name))
        gfs_table_xarray = gfs_io.read_file(gfs_file_name)
        gfstx = gfs_table_xarray

        for this_forecast_hour in forecast_hours:
            hour_index = numpy.where(
                gfstx.coords[gfs_utils.FORECAST_HOUR_DIM].values ==
                this_forecast_hour
            )[0][0]

            for this_field_name in field_names:
                this_field_name_fancy = gfs_plotting.FIELD_NAME_TO_FANCY[
                    this_field_name
                ]

                if this_field_name in gfs_utils.ALL_2D_FIELD_NAMES:
                    field_index = numpy.where(
                        gfstx.coords[gfs_utils.FIELD_DIM_2D].values ==
                        this_field_name
                    )[0][0]

                    data_matrix = gfstx[gfs_utils.DATA_KEY_2D].values[
                        hour_index, ..., field_index
                    ]

                    data_matrix, unit_string = (
                        gfs_plotting.field_to_plotting_units(
                            data_matrix_default_units=data_matrix,
                            field_name=this_field_name
                        )
                    )

                    title_string = '{0:s}{1:s} ({2:s})'.format(
                        this_field_name_fancy[0].upper(),
                        this_field_name_fancy[1:],
                        unit_string
                    )

                    output_file_name = (
                        '{0:s}/init={1:s}_forecast={2:03d}_{3:s}.jpg'
                    ).format(
                        output_dir_name,
                        this_date_string,
                        this_forecast_hour,
                        this_field_name.replace('_', '-')
                    )

                    _plot_one_field(
                        data_matrix=data_matrix,
                        grid_latitudes_deg_n=
                        gfstx.coords[gfs_utils.LATITUDE_DIM].values,
                        grid_longitudes_deg_e=
                        gfstx.coords[gfs_utils.LONGITUDE_DIM].values,
                        border_latitudes_deg_n=border_latitudes_deg_n,
                        border_longitudes_deg_e=border_longitudes_deg_e,
                        field_name=this_field_name,
                        min_colour_percentile=min_colour_percentile,
                        max_colour_percentile=max_colour_percentile,
                        title_string=title_string,
                        output_file_name=output_file_name
                    )

                    continue

                field_index = numpy.where(
                    gfstx.coords[gfs_utils.FIELD_DIM_3D].values ==
                    this_field_name
                )[0][0]

                for this_pressure_mb in pressure_levels_mb:
                    pressure_index = numpy.where(
                        gfstx.coords[gfs_utils.PRESSURE_LEVEL_DIM].values ==
                        this_pressure_mb
                    )[0][0]

                    data_matrix = gfstx[gfs_utils.DATA_KEY_3D].values[
                        hour_index, ..., pressure_index, field_index
                    ]

                    data_matrix, unit_string = (
                        gfs_plotting.field_to_plotting_units(
                            data_matrix_default_units=data_matrix,
                            field_name=this_field_name
                        )
                    )

                    title_string = '{0:s}{1:s} ({2:s}) at {3:d} hPa'.format(
                        this_field_name_fancy[0].upper(),
                        this_field_name_fancy[1:],
                        unit_string,
                        this_pressure_mb
                    )

                    output_file_name = (
                        '{0:s}/init={1:s}_forecast={2:03d}_{3:s}_{4:04d}mb.jpg'
                    ).format(
                        output_dir_name,
                        this_date_string,
                        this_forecast_hour,
                        this_field_name.replace('_', '-'),
                        this_pressure_mb
                    )

                    _plot_one_field(
                        data_matrix=data_matrix,
                        grid_latitudes_deg_n=
                        gfstx.coords[gfs_utils.LATITUDE_DIM].values,
                        grid_longitudes_deg_e=
                        gfstx.coords[gfs_utils.LONGITUDE_DIM].values,
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
        pressure_levels_mb=numpy.array(
            getattr(INPUT_ARG_OBJECT, PRESSURE_LEVELS_ARG_NAME), dtype=int
        ),
        init_date_strings=getattr(INPUT_ARG_OBJECT, INIT_DATES_ARG_NAME),
        forecast_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, FORECAST_HOURS_ARG_NAME), dtype=int
        ),
        min_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MIN_COLOUR_PERCENTILE_ARG_NAME
        ),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_COLOUR_PERCENTILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
