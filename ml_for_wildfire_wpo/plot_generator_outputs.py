"""Plots generator outputs."""

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
import border_io
import gfs_utils
import neural_net
import plotting_utils
import gfs_plotting
import fwi_plotting
import training_args

OUTER_GRID_LATITUDES_DEG_N = numpy.linspace(12, 78, num=265, dtype=float)
OUTER_GRID_LONGITUDES_DEG_E = numpy.linspace(166, 300, num=537, dtype=float)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = training_args.add_input_args(parser_object=INPUT_ARG_PARSER)


def _run(template_file_name, output_dir_name,
         inner_latitude_limits_deg_n, inner_longitude_limits_deg_e,
         outer_latitude_buffer_deg, outer_longitude_buffer_deg,
         gfs_predictor_field_names, gfs_pressure_levels_mb,
         gfs_predictor_lead_times_hours, gfs_normalization_file_name,
         era5_constant_file_name, era5_constant_predictor_field_names,
         target_field_names, target_lead_time_days, target_lag_times_days,
         gfs_forecast_target_lead_times_days, target_normalization_file_name,
         num_examples_per_batch, sentinel_value,
         gfs_dir_name_for_training, target_dir_name_for_training,
         gfs_forecast_target_dir_name_for_training,
         init_date_limit_strings_for_training,
         gfs_dir_name_for_validation, target_dir_name_for_validation,
         gfs_forecast_target_dir_name_for_validation,
         init_date_limit_strings_for_validation,
         num_epochs, num_training_batches_per_epoch,
         num_validation_batches_per_epoch,
         plateau_patience_epochs, plateau_learning_rate_multiplier,
         early_stopping_patience_epochs):
    """Trains neural net.

    This is effectively the main method.

    :param template_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    :param inner_latitude_limits_deg_n: Same.
    :param inner_longitude_limits_deg_e: Same.
    :param outer_latitude_buffer_deg: Same.
    :param outer_longitude_buffer_deg: Same.
    :param gfs_predictor_field_names: Same.
    :param gfs_pressure_levels_mb: Same.
    :param gfs_predictor_lead_times_hours: Same.
    :param gfs_normalization_file_name: Same.
    :param era5_constant_file_name: Same.
    :param era5_constant_predictor_field_names: Same.
    :param target_field_names: Same.
    :param target_lead_time_days: Same.
    :param target_lag_times_days: Same.
    :param gfs_forecast_target_lead_times_days: Same.
    :param target_normalization_file_name: Same.
    :param num_examples_per_batch: Same.
    :param sentinel_value: Same.
    :param gfs_dir_name_for_training: Same.
    :param target_dir_name_for_training: Same.
    :param gfs_forecast_target_dir_name_for_training: Same.
    :param init_date_limit_strings_for_training: Same.
    :param gfs_dir_name_for_validation: Same.
    :param target_dir_name_for_validation: Same.
    :param gfs_forecast_target_dir_name_for_validation: Same.
    :param init_date_limit_strings_for_validation: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if len(gfs_pressure_levels_mb) == 1 and gfs_pressure_levels_mb[0] <= 0:
        gfs_pressure_levels_mb = None
    if gfs_normalization_file_name == '':
        gfs_normalization_file_name = None
    if target_normalization_file_name == '':
        target_normalization_file_name = None

    if era5_constant_file_name == '':
        era5_constant_file_name = None
        era5_constant_predictor_field_names = None
    elif (
            len(era5_constant_predictor_field_names) == 1 and
            era5_constant_predictor_field_names[0] == ''
    ):
        era5_constant_predictor_field_names = None

    if (
            len(gfs_forecast_target_lead_times_days) == 1 and
            gfs_forecast_target_lead_times_days[0] <= 0
    ):
        gfs_forecast_target_lead_times_days = None
        gfs_forecast_target_dir_name_for_training = None
        gfs_forecast_target_dir_name_for_validation = None

    if (
            gfs_forecast_target_dir_name_for_training is None or
            gfs_forecast_target_dir_name_for_validation is None
    ):
        gfs_forecast_target_lead_times_days = None
        gfs_forecast_target_dir_name_for_training = None
        gfs_forecast_target_dir_name_for_validation = None

    training_option_dict = {
        neural_net.INNER_LATITUDE_LIMITS_KEY: inner_latitude_limits_deg_n,
        neural_net.INNER_LONGITUDE_LIMITS_KEY: inner_longitude_limits_deg_e,
        neural_net.OUTER_LATITUDE_BUFFER_KEY: outer_latitude_buffer_deg,
        neural_net.OUTER_LONGITUDE_BUFFER_KEY: outer_longitude_buffer_deg,
        neural_net.GFS_PREDICTOR_FIELDS_KEY: gfs_predictor_field_names,
        neural_net.GFS_PRESSURE_LEVELS_KEY: gfs_pressure_levels_mb,
        neural_net.GFS_PREDICTOR_LEADS_KEY: gfs_predictor_lead_times_hours,
        neural_net.GFS_NORM_FILE_KEY: gfs_normalization_file_name,
        neural_net.ERA5_CONSTANT_PREDICTOR_FIELDS_KEY:
            era5_constant_predictor_field_names,
        neural_net.ERA5_CONSTANT_FILE_KEY: era5_constant_file_name,
        neural_net.TARGET_FIELDS_KEY: target_field_names,
        neural_net.TARGET_LEAD_TIME_KEY: target_lead_time_days,
        neural_net.TARGET_LAG_TIMES_KEY: target_lag_times_days,
        neural_net.GFS_FCST_TARGET_LEAD_TIMES_KEY:
            gfs_forecast_target_lead_times_days,
        neural_net.TARGET_NORM_FILE_KEY: target_normalization_file_name,
        neural_net.BATCH_SIZE_KEY: num_examples_per_batch,
        neural_net.SENTINEL_VALUE_KEY: sentinel_value,
        neural_net.INIT_DATE_LIMITS_KEY: init_date_limit_strings_for_training,
        neural_net.GFS_DIRECTORY_KEY: gfs_dir_name_for_training,
        neural_net.TARGET_DIRECTORY_KEY: target_dir_name_for_training,
        neural_net.GFS_FORECAST_TARGET_DIR_KEY:
            gfs_forecast_target_dir_name_for_training
    }

    gfs_2d_field_names = [
        f for f in gfs_predictor_field_names
        if f in gfs_utils.ALL_2D_FIELD_NAMES
    ]
    gfs_3d_field_names = [
        f for f in gfs_predictor_field_names
        if f in gfs_utils.ALL_3D_FIELD_NAMES
    ]

    num_pressure_levels = len(gfs_pressure_levels_mb)
    num_lead_times = len(gfs_predictor_lead_times_hours)
    num_3d_fields = len(gfs_3d_field_names)
    num_2d_fields = len(gfs_2d_field_names)

    num_target_fields = len(target_field_names)
    num_target_lag_times = len(target_lag_times_days)
    num_gfs_forecast_target_lead_times = len(gfs_forecast_target_lead_times_days)

    generator_object = neural_net.data_generator(training_option_dict)
    example_index = -1

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()
    border_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        border_longitudes_deg_e
    )

    for _ in range(3):
        predictor_matrices, target_matrix_with_weights = next(generator_object)
        gfs_3d_predictor_matrix = predictor_matrices[0].astype(numpy.float64)
        gfs_2d_predictor_matrix = predictor_matrices[1].astype(numpy.float64)
        laglead_target_predictor_matrix = predictor_matrices[2].astype(numpy.float64)
        target_matrix_with_weights = target_matrix_with_weights.astype(numpy.float64)

        this_num_examples = target_matrix_with_weights.shape[0]

        for i in range(this_num_examples):
            example_index += 1

            for j in range(num_3d_fields):
                for k in range(num_pressure_levels):
                    for l in range(num_lead_times):
                        min_colour_value = numpy.percentile(
                            gfs_3d_predictor_matrix[i, ..., k, l, j], 1.
                        )
                        max_colour_value = numpy.percentile(
                            gfs_3d_predictor_matrix[i, ..., k, l, j], 99.
                        )

                        figure_object, axes_object = pyplot.subplots(
                            1, 1,
                            figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
                        )
                        gfs_plotting.plot_field(
                            data_matrix=
                            gfs_3d_predictor_matrix[i, ..., k, l, j],
                            grid_latitudes_deg_n=OUTER_GRID_LATITUDES_DEG_N,
                            grid_longitudes_deg_e=OUTER_GRID_LONGITUDES_DEG_E,
                            colour_map_object=pyplot.get_cmap('viridis'),
                            colour_norm_object=pyplot.Normalize(
                                vmin=min_colour_value, vmax=max_colour_value
                            ),
                            axes_object=axes_object,
                            plot_colour_bar=True
                        )

                        plotting_utils.plot_borders(
                            border_latitudes_deg_n=border_latitudes_deg_n,
                            border_longitudes_deg_e=border_longitudes_deg_e,
                            axes_object=axes_object,
                            line_colour=numpy.full(3, 0.)
                        )
                        plotting_utils.plot_grid_lines(
                            plot_latitudes_deg_n=OUTER_GRID_LATITUDES_DEG_N,
                            plot_longitudes_deg_e=OUTER_GRID_LONGITUDES_DEG_E,
                            axes_object=axes_object,
                            meridian_spacing_deg=20.,
                            parallel_spacing_deg=10.
                        )

                        axes_object.set_xlim(
                            numpy.min(OUTER_GRID_LONGITUDES_DEG_E),
                            numpy.max(OUTER_GRID_LONGITUDES_DEG_E)
                        )
                        axes_object.set_ylim(
                            numpy.min(OUTER_GRID_LATITUDES_DEG_N),
                            numpy.max(OUTER_GRID_LATITUDES_DEG_N)
                        )

                        title_string = (
                            '{0:s} at {1:d} mb and {2:d}-h lead'
                        ).format(
                            gfs_3d_field_names[j],
                            gfs_pressure_levels_mb[k],
                            gfs_predictor_lead_times_hours[l]
                        )
                        axes_object.set_title(title_string)

                        output_file_name = (
                            '{0:s}/example{1:03d}_{2:s}_{3:04d}mb_{4:03d}hours'
                            '.jpg'
                        ).format(
                            output_dir_name,
                            example_index,
                            gfs_3d_field_names[j].replace('_', '-'),
                            gfs_pressure_levels_mb[k],
                            gfs_predictor_lead_times_hours[l]
                        )

                        print('Saving figure to: "{0:s}"...'.format(
                            output_file_name
                        ))
                        figure_object.savefig(
                            output_file_name, dpi=FIGURE_RESOLUTION_DPI,
                            pad_inches=0, bbox_inches='tight'
                        )
                        pyplot.close(figure_object)

            for j in range(num_2d_fields):
                for l in range(num_lead_times):
                    min_colour_value = numpy.percentile(
                        gfs_2d_predictor_matrix[i, ..., l, j], 1.
                    )
                    max_colour_value = numpy.percentile(
                        gfs_2d_predictor_matrix[i, ..., l, j], 99.
                    )

                    figure_object, axes_object = pyplot.subplots(
                        1, 1,
                        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
                    )
                    gfs_plotting.plot_field(
                        data_matrix=gfs_2d_predictor_matrix[i, ..., l, j],
                        grid_latitudes_deg_n=OUTER_GRID_LATITUDES_DEG_N,
                        grid_longitudes_deg_e=OUTER_GRID_LONGITUDES_DEG_E,
                        colour_map_object=pyplot.get_cmap('viridis'),
                        colour_norm_object=pyplot.Normalize(
                            vmin=min_colour_value, vmax=max_colour_value
                        ),
                        axes_object=axes_object,
                        plot_colour_bar=True
                    )

                    plotting_utils.plot_borders(
                        border_latitudes_deg_n=border_latitudes_deg_n,
                        border_longitudes_deg_e=border_longitudes_deg_e,
                        axes_object=axes_object,
                        line_colour=numpy.full(3, 0.)
                    )
                    plotting_utils.plot_grid_lines(
                        plot_latitudes_deg_n=OUTER_GRID_LATITUDES_DEG_N,
                        plot_longitudes_deg_e=OUTER_GRID_LONGITUDES_DEG_E,
                        axes_object=axes_object,
                        meridian_spacing_deg=20.,
                        parallel_spacing_deg=10.
                    )

                    axes_object.set_xlim(
                        numpy.min(OUTER_GRID_LONGITUDES_DEG_E),
                        numpy.max(OUTER_GRID_LONGITUDES_DEG_E)
                    )
                    axes_object.set_ylim(
                        numpy.min(OUTER_GRID_LATITUDES_DEG_N),
                        numpy.max(OUTER_GRID_LATITUDES_DEG_N)
                    )

                    title_string = '{0:s} at {1:d}-h lead'.format(
                        gfs_2d_field_names[j],
                        gfs_predictor_lead_times_hours[l]
                    )
                    axes_object.set_title(title_string)

                    output_file_name = (
                        '{0:s}/example{1:03d}_{2:s}_{3:03d}hours.jpg'
                    ).format(
                        output_dir_name,
                        example_index,
                        gfs_2d_field_names[j].replace('_', '-'),
                        gfs_predictor_lead_times_hours[l]
                    )

                    print('Saving figure to: "{0:s}"...'.format(
                        output_file_name
                    ))
                    figure_object.savefig(
                        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
                        pad_inches=0, bbox_inches='tight'
                    )
                    pyplot.close(figure_object)

            for j in range(num_target_fields):
                for l in range(num_target_lag_times):
                    min_colour_value = numpy.percentile(
                        laglead_target_predictor_matrix[i, ..., l, j], 1.
                    )
                    max_colour_value = numpy.percentile(
                        laglead_target_predictor_matrix[i, ..., l, j], 99.
                    )

                    figure_object, axes_object = pyplot.subplots(
                        1, 1,
                        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
                    )
                    gfs_plotting.plot_field(
                        data_matrix=laglead_target_predictor_matrix[i, ..., l, j],
                        grid_latitudes_deg_n=OUTER_GRID_LATITUDES_DEG_N,
                        grid_longitudes_deg_e=OUTER_GRID_LONGITUDES_DEG_E,
                        colour_map_object=pyplot.get_cmap('viridis'),
                        colour_norm_object=pyplot.Normalize(
                            vmin=min_colour_value, vmax=max_colour_value
                        ),
                        axes_object=axes_object,
                        plot_colour_bar=True
                    )

                    plotting_utils.plot_borders(
                        border_latitudes_deg_n=border_latitudes_deg_n,
                        border_longitudes_deg_e=border_longitudes_deg_e,
                        axes_object=axes_object,
                        line_colour=numpy.full(3, 0.)
                    )
                    plotting_utils.plot_grid_lines(
                        plot_latitudes_deg_n=OUTER_GRID_LATITUDES_DEG_N,
                        plot_longitudes_deg_e=OUTER_GRID_LONGITUDES_DEG_E,
                        axes_object=axes_object,
                        meridian_spacing_deg=20.,
                        parallel_spacing_deg=10.
                    )

                    axes_object.set_xlim(
                        numpy.min(OUTER_GRID_LONGITUDES_DEG_E),
                        numpy.max(OUTER_GRID_LONGITUDES_DEG_E)
                    )
                    axes_object.set_ylim(
                        numpy.min(OUTER_GRID_LATITUDES_DEG_N),
                        numpy.max(OUTER_GRID_LATITUDES_DEG_N)
                    )

                    title_string = 'Actual {0:s} at {1:d}-day lag'.format(
                        target_field_names[j],
                        target_lag_times_days[l]
                    )
                    axes_object.set_title(title_string)

                    output_file_name = (
                        '{0:s}/example{1:03d}_{2:s}_minus{3:03d}days.jpg'
                    ).format(
                        output_dir_name,
                        example_index,
                        target_field_names[j].replace('_', '-'),
                        target_lag_times_days[l]
                    )

                    print('Saving figure to: "{0:s}"...'.format(
                        output_file_name
                    ))
                    figure_object.savefig(
                        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
                        pad_inches=0, bbox_inches='tight'
                    )
                    pyplot.close(figure_object)

            for j in range(num_target_fields):
                for this_l in range(num_gfs_forecast_target_lead_times):
                    l = this_l + num_target_lag_times

                    min_colour_value = numpy.percentile(
                        laglead_target_predictor_matrix[i, ..., l, j], 1.
                    )
                    max_colour_value = numpy.percentile(
                        laglead_target_predictor_matrix[i, ..., l, j], 99.
                    )

                    figure_object, axes_object = pyplot.subplots(
                        1, 1,
                        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
                    )
                    gfs_plotting.plot_field(
                        data_matrix=laglead_target_predictor_matrix[i, ..., l, j],
                        grid_latitudes_deg_n=OUTER_GRID_LATITUDES_DEG_N,
                        grid_longitudes_deg_e=OUTER_GRID_LONGITUDES_DEG_E,
                        colour_map_object=pyplot.get_cmap('viridis'),
                        colour_norm_object=pyplot.Normalize(
                            vmin=min_colour_value, vmax=max_colour_value
                        ),
                        axes_object=axes_object,
                        plot_colour_bar=True
                    )

                    plotting_utils.plot_borders(
                        border_latitudes_deg_n=border_latitudes_deg_n,
                        border_longitudes_deg_e=border_longitudes_deg_e,
                        axes_object=axes_object,
                        line_colour=numpy.full(3, 0.)
                    )
                    plotting_utils.plot_grid_lines(
                        plot_latitudes_deg_n=OUTER_GRID_LATITUDES_DEG_N,
                        plot_longitudes_deg_e=OUTER_GRID_LONGITUDES_DEG_E,
                        axes_object=axes_object,
                        meridian_spacing_deg=20.,
                        parallel_spacing_deg=10.
                    )

                    axes_object.set_xlim(
                        numpy.min(OUTER_GRID_LONGITUDES_DEG_E),
                        numpy.max(OUTER_GRID_LONGITUDES_DEG_E)
                    )
                    axes_object.set_ylim(
                        numpy.min(OUTER_GRID_LATITUDES_DEG_N),
                        numpy.max(OUTER_GRID_LATITUDES_DEG_N)
                    )

                    title_string = 'GFS-forecast {0:s} at {1:d}-day lead'.format(
                        target_field_names[j],
                        gfs_forecast_target_lead_times_days[this_l]
                    )
                    axes_object.set_title(title_string)

                    output_file_name = (
                        '{0:s}/example{1:03d}_{2:s}_{3:03d}days.jpg'
                    ).format(
                        output_dir_name,
                        example_index,
                        target_field_names[j].replace('_', '-'),
                        gfs_forecast_target_lead_times_days[this_l]
                    )

                    print('Saving figure to: "{0:s}"...'.format(
                        output_file_name
                    ))
                    figure_object.savefig(
                        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
                        pad_inches=0, bbox_inches='tight'
                    )
                    pyplot.close(figure_object)

            for j in range(num_target_fields):
                figure_object, axes_object = pyplot.subplots(
                    1, 1,
                    figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
                )

                colour_map_object, colour_norm_object = (
                    fwi_plotting.field_to_colour_scheme(
                        target_field_names[j]
                    )
                )
                fwi_plotting.plot_field(
                    data_matrix=target_matrix_with_weights[i, ..., j],
                    grid_latitudes_deg_n=OUTER_GRID_LATITUDES_DEG_N,
                    grid_longitudes_deg_e=OUTER_GRID_LONGITUDES_DEG_E,
                    colour_map_object=colour_map_object,
                    colour_norm_object=colour_norm_object,
                    axes_object=axes_object,
                    plot_colour_bar=True
                )

                plotting_utils.plot_borders(
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    axes_object=axes_object,
                    line_colour=numpy.full(3, 0.)
                )
                plotting_utils.plot_grid_lines(
                    plot_latitudes_deg_n=OUTER_GRID_LATITUDES_DEG_N,
                    plot_longitudes_deg_e=OUTER_GRID_LONGITUDES_DEG_E,
                    axes_object=axes_object,
                    meridian_spacing_deg=20.,
                    parallel_spacing_deg=10.
                )

                axes_object.set_xlim(
                    numpy.min(OUTER_GRID_LONGITUDES_DEG_E),
                    numpy.max(OUTER_GRID_LONGITUDES_DEG_E)
                )
                axes_object.set_ylim(
                    numpy.min(OUTER_GRID_LATITUDES_DEG_N),
                    numpy.max(OUTER_GRID_LATITUDES_DEG_N)
                )

                axes_object.set_title('Target field ({0:s})'.format(
                    target_field_names[j]
                ))

                output_file_name = '{0:s}/example{1:03d}_{2:s}_target.jpg'.format(
                    output_dir_name,
                    example_index,
                    target_field_names[j].replace('_', '-')
                )
                print('Saving figure to: "{0:s}"...'.format(
                    output_file_name
                ))
                figure_object.savefig(
                    output_file_name, dpi=FIGURE_RESOLUTION_DPI,
                    pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(figure_object)

            figure_object, axes_object = pyplot.subplots(
                1, 1,
                figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            fwi_plotting.plot_field(
                data_matrix=target_matrix_with_weights[i, ..., 1],
                grid_latitudes_deg_n=OUTER_GRID_LATITUDES_DEG_N,
                grid_longitudes_deg_e=OUTER_GRID_LONGITUDES_DEG_E,
                colour_map_object=pyplot.get_cmap('Greens'),
                colour_norm_object=pyplot.Normalize(vmin=0., vmax=1.),
                axes_object=axes_object,
                plot_colour_bar=True
            )

            plotting_utils.plot_borders(
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                axes_object=axes_object,
                line_colour=numpy.full(3, 0.)
            )
            plotting_utils.plot_grid_lines(
                plot_latitudes_deg_n=OUTER_GRID_LATITUDES_DEG_N,
                plot_longitudes_deg_e=OUTER_GRID_LONGITUDES_DEG_E,
                axes_object=axes_object,
                meridian_spacing_deg=20.,
                parallel_spacing_deg=10.
            )

            axes_object.set_xlim(
                numpy.min(OUTER_GRID_LONGITUDES_DEG_E),
                numpy.max(OUTER_GRID_LONGITUDES_DEG_E)
            )
            axes_object.set_ylim(
                numpy.min(OUTER_GRID_LATITUDES_DEG_N),
                numpy.max(OUTER_GRID_LATITUDES_DEG_N)
            )

            axes_object.set_title('Weights')

            output_file_name = '{0:s}/example{1:03d}_weights.jpg'.format(
                output_dir_name, example_index
            )
            print('Saving figure to: "{0:s}"...'.format(
                output_file_name
            ))
            figure_object.savefig(
                output_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        template_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.TEMPLATE_FILE_ARG_NAME
        ),
        output_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.OUTPUT_DIR_ARG_NAME
        ),
        inner_latitude_limits_deg_n=numpy.array(
            getattr(
                INPUT_ARG_OBJECT, training_args.INNER_LATITUDE_LIMITS_ARG_NAME
            ),
            dtype=float
        ),
        inner_longitude_limits_deg_e=numpy.array(
            getattr(
                INPUT_ARG_OBJECT, training_args.INNER_LONGITUDE_LIMITS_ARG_NAME
            ),
            dtype=float
        ),
        outer_latitude_buffer_deg=getattr(
            INPUT_ARG_OBJECT, training_args.OUTER_LATITUDE_BUFFER_ARG_NAME
        ),
        outer_longitude_buffer_deg=getattr(
            INPUT_ARG_OBJECT, training_args.OUTER_LONGITUDE_BUFFER_ARG_NAME
        ),
        gfs_predictor_field_names=getattr(
            INPUT_ARG_OBJECT, training_args.GFS_PREDICTORS_ARG_NAME
        ),
        gfs_pressure_levels_mb=numpy.array(
            getattr(
                INPUT_ARG_OBJECT, training_args.GFS_PRESSURE_LEVELS_ARG_NAME
            ),
            dtype=int
        ),
        gfs_predictor_lead_times_hours=numpy.array(
            getattr(
                INPUT_ARG_OBJECT, training_args.GFS_PREDICTOR_LEADS_ARG_NAME
            ),
            dtype=int
        ),
        gfs_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.GFS_NORM_FILE_ARG_NAME
        ),
        era5_constant_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.ERA5_CONSTANT_FILE_ARG_NAME
        ),
        era5_constant_predictor_field_names=getattr(
            INPUT_ARG_OBJECT, training_args.ERA5_CONSTANT_PREDICTORS_ARG_NAME
        ),
        target_field_names=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_FIELDS_ARG_NAME
        ),
        target_lead_time_days=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_LEAD_TIME_ARG_NAME
        ),
        target_lag_times_days=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.TARGET_LAG_TIMES_ARG_NAME),
            dtype=int
        ),
        gfs_forecast_target_lead_times_days=numpy.array(
            getattr(
                INPUT_ARG_OBJECT,
                training_args.GFS_FCST_TARGET_LEAD_TIMES_ARG_NAME
            ),
            dtype=int
        ),
        target_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_NORM_FILE_ARG_NAME
        ),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, training_args.BATCH_SIZE_ARG_NAME
        ),
        sentinel_value=getattr(
            INPUT_ARG_OBJECT, training_args.SENTINEL_VALUE_ARG_NAME
        ),
        gfs_dir_name_for_training=getattr(
            INPUT_ARG_OBJECT, training_args.GFS_TRAINING_DIR_ARG_NAME
        ),
        target_dir_name_for_training=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_TRAINING_DIR_ARG_NAME
        ),
        gfs_forecast_target_dir_name_for_training=getattr(
            INPUT_ARG_OBJECT,
            training_args.GFS_FCST_TARGET_TRAINING_DIR_ARG_NAME
        ),
        init_date_limit_strings_for_training=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_DATE_LIMITS_ARG_NAME
        ),
        gfs_dir_name_for_validation=getattr(
            INPUT_ARG_OBJECT, training_args.GFS_VALIDATION_DIR_ARG_NAME
        ),
        target_dir_name_for_validation=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_VALIDATION_DIR_ARG_NAME
        ),
        gfs_forecast_target_dir_name_for_validation=getattr(
            INPUT_ARG_OBJECT,
            training_args.GFS_FCST_TARGET_VALIDATION_DIR_ARG_NAME
        ),
        init_date_limit_strings_for_validation=getattr(
            INPUT_ARG_OBJECT, training_args.VALIDATION_DATE_LIMITS_ARG_NAME
        ),
        num_epochs=getattr(INPUT_ARG_OBJECT, training_args.NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_TRAINING_BATCHES_ARG_NAME
        ),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_VALIDATION_BATCHES_ARG_NAME
        ),
        plateau_patience_epochs=getattr(
            INPUT_ARG_OBJECT, training_args.PLATEAU_PATIENCE_ARG_NAME
        ),
        plateau_learning_rate_multiplier=getattr(
            INPUT_ARG_OBJECT, training_args.PLATEAU_MULTIPLIER_ARG_NAME
        ),
        early_stopping_patience_epochs=getattr(
            INPUT_ARG_OBJECT, training_args.EARLY_STOPPING_PATIENCE_ARG_NAME
        )
    )