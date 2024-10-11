"""Plots extreme cases."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from ml_for_wildfire_wpo.io import gfs_io
from ml_for_wildfire_wpo.io import border_io
from ml_for_wildfire_wpo.io import extreme_cases_io
from ml_for_wildfire_wpo.utils import gfs_utils
from ml_for_wildfire_wpo.utils import misc_utils
from ml_for_wildfire_wpo.machine_learning import neural_net
from ml_for_wildfire_wpo.plotting import gfs_plotting
from ml_for_wildfire_wpo.plotting import fwi_plotting
from ml_for_wildfire_wpo.scripts import plot_shapley_maps
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import imagemagick_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

EXTREME_CASES_FILE_ARG_NAME = 'input_extreme_cases_file_name'
GFS_DIRECTORY_ARG_NAME = 'input_gfs_directory_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
GFS_FCST_TARGET_DIR_ARG_NAME = 'input_gfs_fcst_target_dir_name'
GFS_NORM_FILE_ARG_NAME = 'input_gfs_norm_file_name'
PREDICTOR_COLOUR_LIMITS_ARG_NAME = 'predictor_colour_limits_prctile'
PLOT_LATITUDE_LIMITS_ARG_NAME = 'plot_latitude_limits_deg_n'
PLOT_LONGITUDE_LIMITS_ARG_NAME = 'plot_longitude_limits_deg_e'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXTREME_CASES_FILE_HELP_STRING = (
    'Path to file with metadata for extreme cases (will be read by '
    '`extreme_cases_io.read_file`).'
)
GFS_DIRECTORY_HELP_STRING = (
    'Name of directory with *unnormalized* GFS weather forecasts.  Files '
    'therein will be found by `gfs_io.find_file` and read by '
    '`gfs_io.read_file`.'
)
TARGET_DIR_HELP_STRING = (
    'Name of directory with *unnormalized* target fields.  Files therein will '
    'be found by `canadian_fwo_io.find_file` and read by '
    '`canadian_fwo_io.read_file`.'
)
GFS_FCST_TARGET_DIR_HELP_STRING = (
    'Name of directory with *unnormalized* GFS fire-weather forecasts.  Files '
    'therein will be found by `gfs_daily_io.find_file` and read by '
    '`gfs_daily_io.read_file`.'
)
GFS_NORM_FILE_HELP_STRING = (
    'Path to file with GFS-normalization parameters.  If you pass this '
    'normalization file, the percentiles in `{0:s}` will be computed once -- '
    'on values in this normalization file -- and the resulting colour limits '
    'will be used for every GFS plot.  If you leave this argument empty, the '
    'percentiles in `{0:s}` will be computed for every 2-D slice of GFS data.'
).format(
    PREDICTOR_COLOUR_LIMITS_ARG_NAME
)
PREDICTOR_COLOUR_LIMITS_HELP_STRING = (
    'Colour limits for predictor values, in terms of percentile (ranging from '
    '0...100).'
)
PLOT_LATITUDE_LIMITS_HELP_STRING = (
    'Length-2 list with min/max latitudes (deg north) to plot.  If you want to '
    'plot the full domain, just leave this argument empty.'
)
PLOT_LONGITUDE_LIMITS_HELP_STRING = (
    'Length-2 list with min/max longitudes (deg east) to plot.  If you want to '
    'plot the full domain, just leave this argument empty.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXTREME_CASES_FILE_ARG_NAME, type=str, required=True,
    help=EXTREME_CASES_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GFS_DIRECTORY_ARG_NAME, type=str, required=True,
    help=GFS_DIRECTORY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GFS_FCST_TARGET_DIR_ARG_NAME, type=str, required=True,
    help=GFS_FCST_TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GFS_NORM_FILE_ARG_NAME, type=str, required=True,
    help=GFS_NORM_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_COLOUR_LIMITS_ARG_NAME, type=float, nargs=2, required=True,
    help=PREDICTOR_COLOUR_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_LATITUDE_LIMITS_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1.], help=PLOT_LATITUDE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_LONGITUDE_LIMITS_ARG_NAME, type=float, nargs='+',
    required=False, default=[361.], help=PLOT_LONGITUDE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_extreme_case(
        extreme_cases_table_xarray, validation_option_dict, init_date_string,
        plot_latitude_limits_deg_n, plot_longitude_limits_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e,
        predictor_colour_limits_prctile, gfs_norm_param_table_xarray,
        output_dir_name):
    """Plots all data for one extreme case (i.e., one forecast-init date).

    P = number of points in border file

    :param extreme_cases_table_xarray: xarray table in format returned by
        `extreme_cases_io.read_file`.
    :param validation_option_dict: Dictionary with validation options for
        neural-net generator.
    :param init_date_string: Forecast-init date (format "yyyymmdd").
    :param plot_latitude_limits_deg_n: See documentation at top of this script.
    :param plot_longitude_limits_deg_e: Same.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg
        north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg
        east).
    :param predictor_colour_limits_prctile: See documentation at top of this
        script.
    :param gfs_norm_param_table_xarray: xarray table with GFS-normalization
        parameters, in format returned by `gfs_io.read_normalization_file`.
    :param output_dir_name: Path to output directory (figures will be saved
        here).
    """

    init_date_string_nice = (
        init_date_string[:4] +
        '-' + init_date_string[4:6] +
        '-' + init_date_string[6:]
    )

    vod = validation_option_dict
    all_target_field_names = vod[neural_net.TARGET_FIELDS_KEY]
    num_target_fields = len(all_target_field_names)

    xct = extreme_cases_table_xarray
    model_lead_time_days = xct.attrs[extreme_cases_io.MODEL_LEAD_TIME_KEY]
    target_field_name = xct.attrs[extreme_cases_io.TARGET_FIELD_KEY]

    try:
        data_dict = neural_net.create_data(
            option_dict=validation_option_dict,
            init_date_string=init_date_string,
            model_lead_time_days=model_lead_time_days
        )
        print(SEPARATOR_STRING)
    except:
        print(SEPARATOR_STRING)
        return

    predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    target_matrix = data_dict[neural_net.TARGETS_AND_WEIGHTS_KEY][
        0, ..., :num_target_fields
    ]
    model_input_layer_names = data_dict[neural_net.INPUT_LAYER_NAMES_KEY]
    grid_latitudes_deg_n = data_dict[neural_net.GRID_LATITUDES_KEY]
    grid_longitudes_deg_e = data_dict[neural_net.GRID_LONGITUDES_KEY]
    del data_dict

    if plot_latitude_limits_deg_n is not None:
        desired_row_indices = misc_utils.desired_latitudes_to_rows(
            grid_latitudes_deg_n=grid_latitudes_deg_n,
            start_latitude_deg_n=plot_latitude_limits_deg_n[0],
            end_latitude_deg_n=plot_latitude_limits_deg_n[1]
        )

        for k in range(len(predictor_matrices)):
            if len(predictor_matrices[k].shape) < 2:
                continue

            predictor_matrices[k] = (
                predictor_matrices[k][:, desired_row_indices, ...]
            )

    if plot_longitude_limits_deg_e is not None:
        desired_column_indices = misc_utils.desired_longitudes_to_columns(
            grid_longitudes_deg_e=grid_longitudes_deg_e,
            start_longitude_deg_e=plot_longitude_limits_deg_e[0],
            end_longitude_deg_e=plot_longitude_limits_deg_e[1]
        )

        for k in range(len(predictor_matrices)):
            if len(predictor_matrices[k].shape) < 2:
                continue

            predictor_matrices[k] = (
                predictor_matrices[k][:, :, desired_column_indices, ...]
            )

    gfs_lead_times_hours = vod[neural_net.MODEL_LEAD_TO_GFS_PRED_LEADS_KEY][
        model_lead_time_days
    ]
    gfs_field_names = vod[neural_net.GFS_PREDICTOR_FIELDS_KEY]
    gfs_field_names_2d = [
        f for f in gfs_field_names
        if f in gfs_utils.ALL_2D_FIELD_NAMES
    ]

    if len(gfs_field_names_2d) > 0:
        lyr_idx = model_input_layer_names.index(neural_net.GFS_2D_LAYER_NAME)
        gfs_2d_predictor_matrix = predictor_matrices[lyr_idx][0, ...]
    else:
        gfs_2d_predictor_matrix = numpy.array([])

    for t in range(len(gfs_lead_times_hours)):
        panel_file_names = [''] * len(gfs_field_names_2d)

        for f in range(len(gfs_field_names_2d)):
            this_predictor_matrix = gfs_2d_predictor_matrix[..., t, f] + 0.
            unit_string = gfs_plotting.FIELD_TO_PLOTTING_UNIT_STRING[
                gfs_field_names_2d[f]
            ]

            title_string = 'GFS {0:s} ({1:s}), {2:s} + {3:d} hours'.format(
                gfs_plotting.FIELD_NAME_TO_FANCY[gfs_field_names_2d[f]],
                unit_string,
                init_date_string_nice,
                gfs_lead_times_hours[t]
            )

            panel_file_names[f] = (
                '{0:s}/init={1:s}_gfs_{2:s}_{3:03d}hours.jpg'
            ).format(
                output_dir_name,
                init_date_string,
                gfs_field_names_2d[f].replace('_', '-'),
                gfs_lead_times_hours[t]
            )

            if gfs_norm_param_table_xarray is None:
                this_norm_param_table_xarray = None
            else:
                gfs_npt = gfs_norm_param_table_xarray

                t_idxs = numpy.where(
                    gfs_npt.coords[gfs_utils.FORECAST_HOUR_DIM].values ==
                    gfs_lead_times_hours[t]
                )[0]
                this_npt = gfs_npt.isel({gfs_utils.FORECAST_HOUR_DIM: t_idxs})

                f_idxs = numpy.where(
                    this_npt.coords[gfs_utils.FIELD_DIM_2D].values ==
                    gfs_field_names_2d[f]
                )[0]
                this_npt = this_npt.isel({gfs_utils.FIELD_DIM_2D: f_idxs})

                this_norm_param_table_xarray = this_npt

            figure_object = plot_shapley_maps._plot_one_gfs_field(
                data_matrix=this_predictor_matrix,
                grid_latitudes_deg_n=grid_latitudes_deg_n,
                grid_longitudes_deg_e=grid_longitudes_deg_e,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                field_name=gfs_field_names_2d[f],
                min_colour_percentile=predictor_colour_limits_prctile[0],
                max_colour_percentile=predictor_colour_limits_prctile[1],
                title_string=title_string,
                gfs_norm_param_table_xarray=this_norm_param_table_xarray
            )[0]

            print('Saving figure to: "{0:s}"...'.format(panel_file_names[f]))
            figure_object.savefig(
                panel_file_names[f], dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            imagemagick_utils.resize_image(
                input_file_name=panel_file_names[f],
                output_file_name=panel_file_names[f],
                output_size_pixels=PANEL_SIZE_PX
            )

        concat_figure_file_name = (
            '{0:s}/init={1:s}_gfs2d_{2:03d}hours.jpg'
        ).format(
            output_dir_name,
            init_date_string,
            gfs_lead_times_hours[t]
        )

        num_panel_rows = int(numpy.floor(
            numpy.sqrt(len(panel_file_names))
        ))
        num_panel_columns = int(numpy.ceil(
            float(len(panel_file_names)) / num_panel_rows
        ))

        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=panel_file_names,
            output_file_name=concat_figure_file_name,
            num_panel_rows=num_panel_rows,
            num_panel_columns=num_panel_columns,
            border_width_pixels=25
        )
        imagemagick_utils.trim_whitespace(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            border_width_pixels=0
        )

        for this_panel_file_name in panel_file_names:
            os.remove(this_panel_file_name)

    gfs_pressure_levels_mb = vod[neural_net.GFS_PRESSURE_LEVELS_KEY]
    gfs_field_names_3d = [
        f for f in gfs_field_names
        if f in gfs_utils.ALL_3D_FIELD_NAMES
    ]

    if len(gfs_field_names_3d) > 0:
        lyr_idx = model_input_layer_names.index(neural_net.GFS_3D_LAYER_NAME)
        gfs_3d_predictor_matrix = predictor_matrices[lyr_idx][0, ...]
    else:
        gfs_3d_predictor_matrix = numpy.array([])

    for p in range(len(gfs_pressure_levels_mb)):
        for t in range(len(gfs_lead_times_hours)):
            panel_file_names = [''] * len(gfs_field_names_2d)

            for f in range(len(gfs_field_names_3d)):
                this_predictor_matrix = (
                    gfs_3d_predictor_matrix[..., p, t, f] + 0.
                )
                unit_string = gfs_plotting.FIELD_TO_PLOTTING_UNIT_STRING[
                    gfs_field_names_3d[f]
                ]

                title_string = (
                    'GFS {0:s} ({1:s}), {2:.0f} mb, {3:s} + {4:d} hours'
                ).format(
                    gfs_plotting.FIELD_NAME_TO_FANCY[gfs_field_names_3d[f]],
                    unit_string,
                    gfs_pressure_levels_mb[p],
                    init_date_string_nice,
                    gfs_lead_times_hours[t]
                )

                panel_file_names[f] = (
                    '{0:s}/init={1:s}_gfs_{2:s}_{3:04d}mb_{4:03d}hours.jpg'
                ).format(
                    output_dir_name,
                    init_date_string,
                    gfs_field_names_3d[f].replace('_', '-'),
                    int(numpy.round(gfs_pressure_levels_mb[p])),
                    gfs_lead_times_hours[t]
                )

                if gfs_norm_param_table_xarray is None:
                    this_norm_param_table_xarray = None
                else:
                    gfs_npt = gfs_norm_param_table_xarray

                    p_idxs = numpy.where(
                        gfs_npt.coords[gfs_utils.PRESSURE_LEVEL_DIM].values ==
                        gfs_pressure_levels_mb[p]
                    )[0]
                    this_npt = gfs_npt.isel(
                        {gfs_utils.PRESSURE_LEVEL_DIM: p_idxs}
                    )

                    f_idxs = numpy.where(
                        this_npt.coords[gfs_utils.FIELD_DIM_3D].values ==
                        gfs_field_names_3d[f]
                    )[0]
                    this_npt = this_npt.isel({gfs_utils.FIELD_DIM_3D: f_idxs})

                    this_norm_param_table_xarray = this_npt

                figure_object = plot_shapley_maps._plot_one_gfs_field(
                    data_matrix=this_predictor_matrix,
                    grid_latitudes_deg_n=grid_latitudes_deg_n,
                    grid_longitudes_deg_e=grid_longitudes_deg_e,
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    field_name=gfs_field_names_3d[f],
                    min_colour_percentile=predictor_colour_limits_prctile[0],
                    max_colour_percentile=predictor_colour_limits_prctile[1],
                    title_string=title_string,
                    gfs_norm_param_table_xarray=this_norm_param_table_xarray
                )[0]

                print('Saving figure to: "{0:s}"...'.format(
                    panel_file_names[f]
                ))
                figure_object.savefig(
                    panel_file_names[f], dpi=FIGURE_RESOLUTION_DPI,
                    pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(figure_object)

                imagemagick_utils.resize_image(
                    input_file_name=panel_file_names[f],
                    output_file_name=panel_file_names[f],
                    output_size_pixels=PANEL_SIZE_PX
                )

            concat_figure_file_name = (
                '{0:s}/init={1:s}_gfs3d_{2:04d}mb_{3:03d}hours.jpg'
            ).format(
                output_dir_name,
                init_date_string,
                int(numpy.round(gfs_pressure_levels_mb[p])),
                gfs_lead_times_hours[t]
            )

            num_panel_rows = int(numpy.floor(
                numpy.sqrt(len(panel_file_names))
            ))
            num_panel_columns = int(numpy.ceil(
                float(len(panel_file_names)) / num_panel_rows
            ))

            print('Concatenating panels to: "{0:s}"...'.format(
                concat_figure_file_name
            ))
            imagemagick_utils.concatenate_images(
                input_file_names=panel_file_names,
                output_file_name=concat_figure_file_name,
                num_panel_rows=num_panel_rows,
                num_panel_columns=num_panel_columns,
                border_width_pixels=25
            )
            imagemagick_utils.trim_whitespace(
                input_file_name=concat_figure_file_name,
                output_file_name=concat_figure_file_name,
                border_width_pixels=0
            )

            for this_panel_file_name in panel_file_names:
                os.remove(this_panel_file_name)

    # TODO(thunderhoser): This code does not yet handle the case where lagged
    # targets or GFS fire-weather forecasts are missing.
    target_lag_times_days = vod[neural_net.MODEL_LEAD_TO_TARGET_LAGS_KEY][
        model_lead_time_days
    ]
    gfs_target_lead_times_days = vod[
        neural_net.MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY
    ][model_lead_time_days]

    target_lag_or_lead_times_days = numpy.concatenate([
        target_lag_times_days, -1 * gfs_target_lead_times_days
    ])

    lyr_idx = model_input_layer_names.index(
        neural_net.LAGLEAD_TARGET_LAYER_NAME
    )
    laglead_target_predictor_matrix = predictor_matrices[lyr_idx][0, ...]

    for t in range(len(target_lag_or_lead_times_days)):
        panel_file_names = [''] * len(all_target_field_names)

        for f in range(len(all_target_field_names)):
            this_predictor_matrix = laglead_target_predictor_matrix[..., t, f]

            if target_lag_or_lead_times_days[t] < 0:
                title_string = 'GFS-forecast {0:s}, {1:s} + {2:d} days'.format(
                    fwi_plotting.FIELD_NAME_TO_SIMPLE[
                        all_target_field_names[f]
                    ],
                    init_date_string_nice,
                    -1 * target_lag_or_lead_times_days[t]
                )
            else:
                title_string = 'Lagged-truth {0:s}, {1:s} - {2:d} days'.format(
                    fwi_plotting.FIELD_NAME_TO_SIMPLE[
                        all_target_field_names[f]
                    ],
                    init_date_string_nice,
                    target_lag_or_lead_times_days[t]
                )

            panel_file_names[f] = (
                '{0:s}/init={1:s}_laglead_target_{2:s}_{3:+03d}days.jpg'
            ).format(
                output_dir_name,
                init_date_string,
                all_target_field_names[f].replace('_', '-'),
                -1 * target_lag_or_lead_times_days[t]
            )

            figure_object = plot_shapley_maps._plot_one_fwi_field(
                data_matrix=this_predictor_matrix,
                grid_latitudes_deg_n=grid_latitudes_deg_n,
                grid_longitudes_deg_e=grid_longitudes_deg_e,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                field_name=all_target_field_names[f],
                title_string=title_string
            )[0]

            print('Saving figure to: "{0:s}"...'.format(panel_file_names[f]))
            figure_object.savefig(
                panel_file_names[f], dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            imagemagick_utils.resize_image(
                input_file_name=panel_file_names[f],
                output_file_name=panel_file_names[f],
                output_size_pixels=PANEL_SIZE_PX
            )

        concat_figure_file_name = (
            '{0:s}/init={1:s}_laglead_targets_{2:+03d}days.jpg'
        ).format(
            output_dir_name,
            init_date_string,
            -1 * target_lag_or_lead_times_days[t]
        )

        num_panel_rows = int(numpy.floor(
            numpy.sqrt(len(panel_file_names))
        ))
        num_panel_columns = int(numpy.ceil(
            float(len(panel_file_names)) / num_panel_rows
        ))

        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=panel_file_names,
            output_file_name=concat_figure_file_name,
            num_panel_rows=num_panel_rows,
            num_panel_columns=num_panel_columns,
            border_width_pixels=25
        )
        imagemagick_utils.trim_whitespace(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            border_width_pixels=0
        )

        for this_panel_file_name in panel_file_names:
            os.remove(this_panel_file_name)

    for f in range(len(all_target_field_names)):
        if all_target_field_names[f] != target_field_name:
            continue

        this_target_matrix = target_matrix[..., f]

        title_string = 'Ground-truth {0:s}, {1:s} + {2:d} days'.format(
            fwi_plotting.FIELD_NAME_TO_SIMPLE[
                all_target_field_names[f]
            ],
            init_date_string_nice,
            model_lead_time_days
        )

        output_file_name = (
            '{0:s}/init={1:s}_truth_{2:s}_{3:+03d}days.jpg'
        ).format(
            output_dir_name,
            init_date_string,
            all_target_field_names[f].replace('_', '-'),
            model_lead_time_days
        )

        figure_object = plot_shapley_maps._plot_one_fwi_field(
            data_matrix=this_target_matrix,
            grid_latitudes_deg_n=grid_latitudes_deg_n,
            grid_longitudes_deg_e=grid_longitudes_deg_e,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            field_name=all_target_field_names[f],
            title_string=title_string
        )[0]

        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


def _run(extreme_cases_file_name, gfs_directory_name, target_dir_name,
         gfs_forecast_target_dir_name, gfs_normalization_file_name,
         predictor_colour_limits_prctile, plot_latitude_limits_deg_n,
         plot_longitude_limits_deg_e, output_dir_name):
    """Plots extreme cases.

    This is effectively the main method.

    :param extreme_cases_file_name: See documentation at top of this script.
    :param gfs_directory_name: Same.
    :param target_dir_name: Same.
    :param gfs_forecast_target_dir_name: Same.
    :param gfs_normalization_file_name: Same.
    :param predictor_colour_limits_prctile: Same.
    :param plot_latitude_limits_deg_n: Same.
    :param plot_longitude_limits_deg_e: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    predictor_colour_limits_prctile = numpy.sort(
        predictor_colour_limits_prctile
    )
    error_checking.assert_is_geq(predictor_colour_limits_prctile[0], 0.)
    error_checking.assert_is_leq(predictor_colour_limits_prctile[0], 10.)
    error_checking.assert_is_geq(predictor_colour_limits_prctile[1], 90.)
    error_checking.assert_is_leq(predictor_colour_limits_prctile[1], 100.)

    if (
            len(plot_latitude_limits_deg_n) == 1 and
            plot_latitude_limits_deg_n[0] < 0
    ):
        plot_latitude_limits_deg_n = None

    if (
            len(plot_longitude_limits_deg_e) == 1 and
            plot_longitude_limits_deg_e[0] > 360
    ):
        plot_longitude_limits_deg_e = None

    if plot_latitude_limits_deg_n is not None:
        error_checking.assert_is_numpy_array(
            plot_latitude_limits_deg_n,
            exact_dimensions=numpy.array([2], dtype=int)
        )
        error_checking.assert_is_valid_lat_numpy_array(
            plot_latitude_limits_deg_n
        )

        plot_latitude_limits_deg_n = numpy.sort(plot_latitude_limits_deg_n)
        error_checking.assert_is_greater_numpy_array(
            numpy.diff(plot_latitude_limits_deg_n), 0.
        )

    if plot_longitude_limits_deg_e is not None:
        error_checking.assert_is_numpy_array(
            plot_longitude_limits_deg_e,
            exact_dimensions=numpy.array([2], dtype=int)
        )
        error_checking.assert_is_valid_lng_numpy_array(
            plot_longitude_limits_deg_e,
            positive_in_west_flag=False, negative_in_west_flag=False,
            allow_nan=False
        )
        error_checking.assert_is_greater_numpy_array(
            numpy.absolute(numpy.diff(plot_longitude_limits_deg_e)),
            0
        )

    if gfs_normalization_file_name is None:
        gfs_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            gfs_normalization_file_name
        ))
        gfs_norm_param_table_xarray = gfs_io.read_normalization_file(
            gfs_normalization_file_name
        )

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read metadata for extreme cases.
    print('Reading metadata for extreme cases from: "{0:s}"...'.format(
        extreme_cases_file_name
    ))
    extreme_cases_table_xarray = extreme_cases_io.read_file(
        extreme_cases_file_name
    )
    xct = extreme_cases_table_xarray
    model_file_name = xct.attrs[extreme_cases_io.MODEL_FILE_KEY]
    model_lead_time_days = xct.attrs[extreme_cases_io.MODEL_LEAD_TIME_KEY]

    # Read model metadata.
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    vod = model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]

    vod[neural_net.GFS_DIRECTORY_KEY] = gfs_directory_name
    vod[neural_net.TARGET_DIRECTORY_KEY] = target_dir_name
    vod[neural_net.GFS_FORECAST_TARGET_DIR_KEY] = gfs_forecast_target_dir_name
    vod[neural_net.GFS_NORM_FILE_KEY] = None
    vod[neural_net.TARGET_NORM_FILE_KEY] = None
    vod[neural_net.ERA5_NORM_FILE_KEY] = None
    vod[neural_net.SENTINEL_VALUE_KEY] = None
    vod[neural_net.MODEL_LEAD_TO_GFS_PRED_LEADS_KEY] = {
        model_lead_time_days: vod[
            neural_net.MODEL_LEAD_TO_GFS_PRED_LEADS_KEY
        ][model_lead_time_days]
    }
    vod[neural_net.MODEL_LEAD_TO_TARGET_LAGS_KEY] = {
        model_lead_time_days: vod[
            neural_net.MODEL_LEAD_TO_TARGET_LAGS_KEY
        ][model_lead_time_days]
    }
    vod[neural_net.MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY] = {
        model_lead_time_days: vod[
            neural_net.MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY
        ][model_lead_time_days]
    }

    validation_option_dict = vod
    print(SEPARATOR_STRING)

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()
    init_date_strings = xct[extreme_cases_io.INIT_DATE_KEY]

    for this_date_string in init_date_strings:
        _plot_one_extreme_case(
            extreme_cases_table_xarray=extreme_cases_table_xarray,
            validation_option_dict=validation_option_dict,
            init_date_string=this_date_string,
            plot_latitude_limits_deg_n=plot_latitude_limits_deg_n,
            plot_longitude_limits_deg_e=plot_longitude_limits_deg_e,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            predictor_colour_limits_prctile=predictor_colour_limits_prctile,
            gfs_norm_param_table_xarray=gfs_norm_param_table_xarray,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        extreme_cases_file_name=getattr(
            INPUT_ARG_OBJECT, EXTREME_CASES_FILE_ARG_NAME
        ),
        gfs_directory_name=getattr(INPUT_ARG_OBJECT, GFS_DIRECTORY_ARG_NAME),
        target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        gfs_forecast_target_dir_name=getattr(
            INPUT_ARG_OBJECT, GFS_FCST_TARGET_DIR_ARG_NAME
        ),
        gfs_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, GFS_NORM_FILE_ARG_NAME
        ),
        predictor_colour_limits_prctile=numpy.array(
            getattr(INPUT_ARG_OBJECT, PREDICTOR_COLOUR_LIMITS_ARG_NAME),
            dtype=float
        ),
        plot_latitude_limits_deg_n=numpy.array(
            getattr(INPUT_ARG_OBJECT, PLOT_LATITUDE_LIMITS_ARG_NAME),
            dtype=float
        ),
        plot_longitude_limits_deg_e=numpy.array(
            getattr(INPUT_ARG_OBJECT, PLOT_LONGITUDE_LIMITS_ARG_NAME),
            dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
