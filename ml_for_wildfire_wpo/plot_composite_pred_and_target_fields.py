"""Plots predictor and target fields from one composite."""

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

import file_system_utils
import error_checking
import imagemagick_utils
import gfs_io
import composite_io
import border_io
import misc_utils
import gfs_utils
import era5_constant_utils
import neural_net
import gfs_plotting
import fwi_plotting
import plot_shapley_maps

FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

COMPOSITE_FILE_ARG_NAME = 'input_composite_file_name'
GFS_NORM_FILE_ARG_NAME = 'input_gfs_norm_file_name'
PREDICTOR_COLOUR_LIMITS_ARG_NAME = 'predictor_colour_limits_prctile'
PLOT_LATITUDE_LIMITS_ARG_NAME = 'plot_latitude_limits_deg_n'
PLOT_LONGITUDE_LIMITS_ARG_NAME = 'plot_longitude_limits_deg_e'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

COMPOSITE_FILE_HELP_STRING = (
    'Path to file with composited fields (will be read by '
    '`composite_io.read_file`).'
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
    '--' + COMPOSITE_FILE_ARG_NAME, type=str, required=True,
    help=COMPOSITE_FILE_HELP_STRING
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


def _run(composite_file_name, gfs_normalization_file_name,
         predictor_colour_limits_prctile, plot_latitude_limits_deg_n,
         plot_longitude_limits_deg_e, output_dir_name):
    """Plots predictor and target fields from one composite.

    This is effectively the main method.

    :param composite_file_name: See documentation at top of this script.
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

    # Read composited data.
    print('Reading composited data from: "{0:s}"...'.format(
        composite_file_name
    ))
    composite_table_xarray = composite_io.read_file(composite_file_name)
    ctx = composite_table_xarray

    model_lead_time_days = ctx.attrs[composite_io.MODEL_LEAD_TIME_KEY]

    # Read model metadata.
    model_file_name = ctx.attrs[composite_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    vod = model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]

    # Prepare data from composite file for plotting.
    model_input_layer_names = [
        neural_net.GFS_3D_LAYER_NAME,
        neural_net.GFS_2D_LAYER_NAME,
        neural_net.ERA5_LAYER_NAME,
        neural_net.LAGLEAD_TARGET_LAYER_NAME,
        neural_net.PREDN_BASELINE_LAYER_NAME
    ]
    predictor_keys = [
        composite_io.PREDICTOR_3D_GFS_KEY,
        composite_io.PREDICTOR_2D_GFS_KEY,
        composite_io.PREDICTOR_ERA5_KEY,
        composite_io.PREDICTOR_LAGLEAD_TARGETS_KEY,
        composite_io.PREDICTOR_BASELINE_KEY
    ]
    predictor_matrices = [
        ctx[k].values if k in ctx.data_vars else None
        for k in predictor_keys
    ]

    good_flags = numpy.array(
        [pm is not None for pm in predictor_matrices], dtype=bool
    )
    good_indices = numpy.where(good_flags)[0]
    predictor_matrices = [predictor_matrices[k] for k in good_indices]
    model_input_layer_names = [
        model_input_layer_names[k] for k in good_indices
    ]

    target_matrix = ctx[composite_io.TARGET_VALUE_KEY].values

    if plot_latitude_limits_deg_n is not None:
        desired_row_indices = misc_utils.desired_latitudes_to_rows(
            grid_latitudes_deg_n=ctx[composite_io.LATITUDE_KEY].values,
            start_latitude_deg_n=plot_latitude_limits_deg_n[0],
            end_latitude_deg_n=plot_latitude_limits_deg_n[1]
        )
        ctx = ctx.isel({composite_io.ROW_DIM: desired_row_indices})

        for k in range(len(predictor_matrices)):
            if len(predictor_matrices[k].shape) < 2:
                continue

            predictor_matrices[k] = (
                predictor_matrices[k][:, desired_row_indices, ...]
            )

    if plot_longitude_limits_deg_e is not None:
        desired_column_indices = misc_utils.desired_longitudes_to_columns(
            grid_longitudes_deg_e=ctx[composite_io.LONGITUDE_KEY].values,
            start_longitude_deg_e=plot_longitude_limits_deg_e[0],
            end_longitude_deg_e=plot_longitude_limits_deg_e[1]
        )
        ctx = ctx.isel({composite_io.COLUMN_DIM: desired_column_indices})

        for k in range(len(predictor_matrices)):
            if len(predictor_matrices[k].shape) < 2:
                continue

            predictor_matrices[k] = (
                predictor_matrices[k][:, :, desired_column_indices, ...]
            )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

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
        gfs_2d_predictor_matrix = predictor_matrices[lyr_idx]
    else:
        gfs_2d_predictor_matrix = numpy.array([])

    for t in range(len(gfs_lead_times_hours)):
        if len(gfs_field_names_2d) == 0:
            continue

        panel_file_names = [''] * len(gfs_field_names_2d)

        for f in range(len(gfs_field_names_2d)):
            this_predictor_matrix = gfs_2d_predictor_matrix[..., t, f] + 0.
            unit_string = gfs_plotting.FIELD_TO_PLOTTING_UNIT_STRING[
                gfs_field_names_2d[f]
            ]
            title_string = 'GFS {0:s} ({1:s}), {2:d}-hour lead'.format(
                gfs_plotting.FIELD_NAME_TO_FANCY[gfs_field_names_2d[f]],
                unit_string,
                gfs_lead_times_hours[t]
            )

            panel_file_names[f] = (
                '{0:s}/gfs_{1:s}_{2:03d}hours.jpg'
            ).format(
                output_dir_name,
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

            figure_object, _ = plot_shapley_maps._plot_one_gfs_field(
                data_matrix=this_predictor_matrix,
                grid_latitudes_deg_n=ctx[composite_io.LATITUDE_KEY].values,
                grid_longitudes_deg_e=ctx[composite_io.LONGITUDE_KEY].values,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                field_name=gfs_field_names_2d[f],
                min_colour_percentile=predictor_colour_limits_prctile[0],
                max_colour_percentile=predictor_colour_limits_prctile[1],
                title_string=title_string,
                gfs_norm_param_table_xarray=this_norm_param_table_xarray
            )

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

        concat_figure_file_name = '{0:s}/gfs2d_{1:03d}hours.jpg'.format(
            output_dir_name, gfs_lead_times_hours[t]
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
        gfs_3d_predictor_matrix = predictor_matrices[lyr_idx]
    else:
        gfs_3d_predictor_matrix = numpy.array([])

    for p in range(len(gfs_pressure_levels_mb)):
        for t in range(len(gfs_lead_times_hours)):
            panel_file_names = [''] * len(gfs_field_names_3d)

            if len(gfs_field_names_3d) == 0:
                continue

            for f in range(len(gfs_field_names_3d)):
                this_predictor_matrix = (
                    gfs_3d_predictor_matrix[..., p, t, f] + 0.
                )
                unit_string = gfs_plotting.FIELD_TO_PLOTTING_UNIT_STRING[
                    gfs_field_names_3d[f]
                ]

                title_string = (
                    'GFS {0:s} ({1:s}), {2:.0f} mb, {3:d}-hour lead'
                ).format(
                    gfs_plotting.FIELD_NAME_TO_FANCY[gfs_field_names_3d[f]],
                    unit_string,
                    gfs_pressure_levels_mb[p],
                    gfs_lead_times_hours[t]
                )

                panel_file_names[f] = (
                    '{0:s}/gfs_{1:s}_{2:04d}mb_{3:03d}hours.jpg'
                ).format(
                    output_dir_name,
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

                figure_object, _ = plot_shapley_maps._plot_one_gfs_field(
                    data_matrix=this_predictor_matrix,
                    grid_latitudes_deg_n=ctx[composite_io.LATITUDE_KEY].values,
                    grid_longitudes_deg_e=ctx[composite_io.LONGITUDE_KEY].values,
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    field_name=gfs_field_names_3d[f],
                    min_colour_percentile=predictor_colour_limits_prctile[0],
                    max_colour_percentile=predictor_colour_limits_prctile[1],
                    title_string=title_string,
                    gfs_norm_param_table_xarray=this_norm_param_table_xarray
                )

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
                '{0:s}/gfs3d_{1:04d}mb_{2:03d}hours.jpg'
            ).format(
                output_dir_name,
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

    if vod[neural_net.MODEL_LEAD_TO_TARGET_LAGS_KEY] is None:
        target_lag_times_days = numpy.array([], dtype=int)
    else:
        target_lag_times_days = vod[neural_net.MODEL_LEAD_TO_TARGET_LAGS_KEY][
            model_lead_time_days
        ]

    if vod[neural_net.MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY] is None:
        gfs_target_lead_times_days = numpy.array([], dtype=int)
    else:
        gfs_target_lead_times_days = vod[
            neural_net.MODEL_LEAD_TO_GFS_TARGET_LEADS_KEY
        ][model_lead_time_days]

    target_lag_or_lead_times_days = numpy.concatenate([
        target_lag_times_days, -1 * gfs_target_lead_times_days
    ])
    all_target_field_names = vod[neural_net.TARGET_FIELDS_KEY]

    lyr_idx = model_input_layer_names.index(
        neural_net.LAGLEAD_TARGET_LAYER_NAME
    )
    laglead_target_predictor_matrix = predictor_matrices[lyr_idx]

    for t in range(len(target_lag_or_lead_times_days)):
        panel_file_names = [''] * len(all_target_field_names)

        for f in range(len(all_target_field_names)):
            this_predictor_matrix = laglead_target_predictor_matrix[..., t, f]

            if target_lag_or_lead_times_days[t] < 0:
                title_string = 'GFS-forecast {0:s}, {1:d}-day lead'.format(
                    fwi_plotting.FIELD_NAME_TO_SIMPLE[
                        all_target_field_names[f]
                    ],
                    -1 * target_lag_or_lead_times_days[t]
                )
            else:
                title_string = 'URMA {0:s}, {1:d}-day lag'.format(
                    fwi_plotting.FIELD_NAME_TO_SIMPLE[
                        all_target_field_names[f]
                    ],
                    target_lag_or_lead_times_days[t]
                )

            panel_file_names[f] = (
                '{0:s}/laglead_target_{1:s}_{2:+03d}days.jpg'
            ).format(
                output_dir_name,
                all_target_field_names[f].replace('_', '-'),
                -1 * target_lag_or_lead_times_days[t]
            )

            figure_object, _ = plot_shapley_maps._plot_one_fwi_field(
                data_matrix=this_predictor_matrix,
                grid_latitudes_deg_n=ctx[composite_io.LATITUDE_KEY].values,
                grid_longitudes_deg_e=ctx[composite_io.LONGITUDE_KEY].values,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                field_name=all_target_field_names[f],
                title_string=title_string
            )

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
            '{0:s}/laglead_targets_{1:+03d}days.jpg'
        ).format(
            output_dir_name,
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

    era5_constant_field_names = vod[
        neural_net.ERA5_CONSTANT_PREDICTOR_FIELDS_KEY
    ]

    if era5_constant_field_names is None:
        era5_constant_predictor_matrix = numpy.array([])
    else:
        lyr_idx = model_input_layer_names.index(neural_net.ERA5_LAYER_NAME)
        era5_constant_predictor_matrix = predictor_matrices[lyr_idx]

    panel_file_names = [''] * len(era5_constant_field_names)

    for f in range(len(era5_constant_field_names)):
        this_predictor_matrix = era5_constant_predictor_matrix[..., f]

        unit_string = era5_constant_utils.FIELD_NAME_TO_UNIT_STRING[
            era5_constant_field_names[f]
        ]
        if unit_string != '':
            unit_string = ' ({0:s})'.format(unit_string)

        title_string = 'ERA5 {0:s}{1:s}'.format(
            era5_constant_utils.FIELD_NAME_TO_FANCY[
                era5_constant_field_names[f]
            ],
            unit_string
        )

        panel_file_names[f] = '{0:s}/era5_{1:s}.jpg'.format(
            output_dir_name,
            era5_constant_field_names[f].replace('_', '-')
        )

        figure_object, _ = plot_shapley_maps._plot_one_era5_field(
            data_matrix=this_predictor_matrix,
            grid_latitudes_deg_n=ctx[composite_io.LATITUDE_KEY].values,
            grid_longitudes_deg_e=ctx[composite_io.LONGITUDE_KEY].values,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            min_colour_percentile=predictor_colour_limits_prctile[0],
            max_colour_percentile=predictor_colour_limits_prctile[1],
            title_string=title_string
        )

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

    if len(era5_constant_field_names) > 0:
        concat_figure_file_name = '{0:s}/era5.jpg'.format(output_dir_name)

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

    panel_file_names = [''] * len(all_target_field_names)

    for f in range(len(all_target_field_names)):
        this_target_matrix = target_matrix[..., f]
        title_string = 'Target {0:s}, {1:d}-day lead'.format(
            fwi_plotting.FIELD_NAME_TO_SIMPLE[all_target_field_names[f]],
            model_lead_time_days
        )
        panel_file_names[f] = '{0:s}/actual_{1:s}.jpg'.format(
            output_dir_name,
            all_target_field_names[f].replace('_', '-')
        )

        figure_object, _ = plot_shapley_maps._plot_one_fwi_field(
            data_matrix=this_target_matrix,
            grid_latitudes_deg_n=ctx[composite_io.LATITUDE_KEY].values,
            grid_longitudes_deg_e=ctx[composite_io.LONGITUDE_KEY].values,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            field_name=all_target_field_names[f],
            title_string=title_string
        )

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

    concat_figure_file_name = '{0:s}/actual.jpg'.format(output_dir_name)

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

    do_residual_prediction = vod[neural_net.DO_RESIDUAL_PREDICTION_KEY]
    if not do_residual_prediction:
        return

    lyr_idx = model_input_layer_names.index(
        neural_net.PREDN_BASELINE_LAYER_NAME
    )
    resid_baseline_predictor_matrix = predictor_matrices[lyr_idx]
    panel_file_names = [''] * len(all_target_field_names)

    for f in range(len(all_target_field_names)):
        this_predictor_matrix = resid_baseline_predictor_matrix[..., f]
        title_string = 'Baseline {0:s}, {1:d}-day lag'.format(
            fwi_plotting.FIELD_NAME_TO_SIMPLE[all_target_field_names[f]],
            numpy.min(target_lag_times_days)
        )
        panel_file_names[f] = '{0:s}/residual_baseline_{1:s}.jpg'.format(
            output_dir_name,
            all_target_field_names[f].replace('_', '-')
        )

        figure_object, _ = plot_shapley_maps._plot_one_fwi_field(
            data_matrix=this_predictor_matrix,
            grid_latitudes_deg_n=ctx[composite_io.LATITUDE_KEY].values,
            grid_longitudes_deg_e=ctx[composite_io.LONGITUDE_KEY].values,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            field_name=all_target_field_names[f],
            title_string=title_string
        )

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

    concat_figure_file_name = '{0:s}/residual_baseline.jpg'.format(
        output_dir_name
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


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        composite_file_name=getattr(INPUT_ARG_OBJECT, COMPOSITE_FILE_ARG_NAME),
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
