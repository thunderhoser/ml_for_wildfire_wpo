"""Helper methods for normalization."""

import os
import sys
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gfs_io
import canadian_fwi_io
import era5_constant_io
import gfs_utils
import gfs_daily_utils
import canadian_fwi_utils
import era5_constant_utils

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'

ACCUM_PRECIP_FIELD_NAMES = [
    gfs_utils.PRECIP_NAME, gfs_utils.CONVECTIVE_PRECIP_NAME
]


def _update_z_score_params(z_score_param_dict, new_data_matrix):
    """Updates z-score parameters.

    :param z_score_param_dict: Dictionary with the following keys.
    z_score_param_dict['num_values']: Number of values on which current
        estimates are based.
    z_score_param_dict['mean_value']: Current mean.
    z_score_param_dict['mean_of_squares']: Current mean of squared values.

    :param new_data_matrix: numpy array with new values.  Will be used to
        update estimates in `z_score_param_dict`.
    :return: z_score_param_dict: Same as input, but with new estimates.
    """

    if numpy.all(numpy.isnan(new_data_matrix)):
        return z_score_param_dict

    new_num_values = numpy.sum(numpy.invert(numpy.isnan(new_data_matrix)))

    these_means = numpy.array([
        z_score_param_dict[MEAN_VALUE_KEY],
        numpy.nanmean(new_data_matrix)
    ])
    these_weights = numpy.array([
        z_score_param_dict[NUM_VALUES_KEY], new_num_values
    ])
    z_score_param_dict[MEAN_VALUE_KEY] = numpy.average(
        these_means, weights=these_weights)

    these_means = numpy.array([
        z_score_param_dict[MEAN_OF_SQUARES_KEY],
        numpy.nanmean(new_data_matrix ** 2)
    ])
    these_weights = numpy.array([
        z_score_param_dict[NUM_VALUES_KEY], new_num_values
    ])
    z_score_param_dict[MEAN_OF_SQUARES_KEY] = numpy.average(
        these_means, weights=these_weights)

    z_score_param_dict[NUM_VALUES_KEY] += new_num_values
    return z_score_param_dict


def _get_standard_deviation(z_score_param_dict):
    """Computes standard deviation.

    :param z_score_param_dict: See doc for `_update_z_score_params`.
    :return: standard_deviation: Standard deviation.
    """

    if z_score_param_dict[NUM_VALUES_KEY] == 0:
        return numpy.nan

    multiplier = float(
        z_score_param_dict[NUM_VALUES_KEY]
    ) / (z_score_param_dict[NUM_VALUES_KEY] - 1)

    return numpy.sqrt(multiplier * (
        z_score_param_dict[MEAN_OF_SQUARES_KEY] -
        z_score_param_dict[MEAN_VALUE_KEY] ** 2
    ))


def get_z_score_params_for_gfs(gfs_file_names):
    """Computes z-score parameters for GFS data.

    z-score parameters = mean and standard deviation for every variable

    :param gfs_file_names: 1-D list of paths to GFS files (will be read by
        `gfs_io.read_file`).
    :return: z_score_param_table_xarray: xarray table with z-score parameters.
        Metadata and variable names in this table should make it self-
        explanatory.
    """

    first_fwi_table_xarray = gfs_io.read_file(gfs_file_names[0])
    first_gfst = first_fwi_table_xarray

    field_names_3d = first_gfst.coords[gfs_utils.FIELD_DIM_3D].values.tolist()
    field_names_2d = first_gfst.coords[gfs_utils.FIELD_DIM_2D].values.tolist()
    pressure_levels_mb = numpy.round(
        first_gfst.coords[gfs_utils.PRESSURE_LEVEL_DIM].values
    ).astype(int)
    forecast_hours = numpy.round(
        first_gfst.coords[gfs_utils.FORECAST_HOUR_DIM].values
    ).astype(int)

    z_score_dict_dict_2d = {}

    for this_field_name in field_names_2d:
        if this_field_name not in ACCUM_PRECIP_FIELD_NAMES:
            z_score_dict_dict_2d[this_field_name] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.
            }
            continue

        for this_forecast_hour in forecast_hours:
            z_score_dict_dict_2d[this_field_name, this_forecast_hour] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.
            }

    z_score_dict_dict_3d = {}

    for this_field_name in field_names_3d:
        for this_pressure_level_mb in pressure_levels_mb:
            z_score_dict_dict_3d[this_field_name, this_pressure_level_mb] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.
            }

    for this_gfs_file_name in gfs_file_names:
        print('Reading data from: "{0:s}"...'.format(this_gfs_file_name))
        this_fwi_table_xarray = gfs_io.read_file(this_gfs_file_name)
        this_gfst = this_fwi_table_xarray

        for j in range(len(this_gfst.coords[gfs_utils.FIELD_DIM_2D].values)):
            f = this_gfst.coords[gfs_utils.FIELD_DIM_2D].values[j]

            if f not in ACCUM_PRECIP_FIELD_NAMES:
                z_score_dict_dict_2d[f] = _update_z_score_params(
                    z_score_param_dict=z_score_dict_dict_2d[f],
                    new_data_matrix=
                    this_gfst[gfs_utils.DATA_KEY_2D].values[..., j]
                )
                continue

            for k in range(
                    len(this_gfst.coords[gfs_utils.FORECAST_HOUR_DIM].values)
            ):
                h = int(numpy.round(
                    this_gfst.coords[gfs_utils.FORECAST_HOUR_DIM].values[k]
                ))

                z_score_dict_dict_2d[f, h] = _update_z_score_params(
                    z_score_param_dict=z_score_dict_dict_2d[f, h],
                    new_data_matrix=
                    this_gfst[gfs_utils.DATA_KEY_2D].values[k, ..., j]
                )

        for j in range(len(this_gfst.coords[gfs_utils.FIELD_DIM_3D].values)):
            for k in range(
                    len(this_gfst.coords[gfs_utils.PRESSURE_LEVEL_DIM].values)
            ):
                f = this_gfst.coords[gfs_utils.FIELD_DIM_3D].values[j]
                p = int(numpy.round(
                    this_gfst.coords[gfs_utils.PRESSURE_LEVEL_DIM].values[k]
                ))

                z_score_dict_dict_3d[f, p] = _update_z_score_params(
                    z_score_param_dict=z_score_dict_dict_3d[f, p],
                    new_data_matrix=
                    this_gfst[gfs_utils.DATA_KEY_3D].values[..., k, j]
                )

    num_pressure_levels = len(pressure_levels_mb)
    num_3d_fields = len(field_names_3d)
    num_2d_fields = len(field_names_2d)
    num_forecast_hours = len(forecast_hours)

    mean_value_matrix_3d = numpy.full(
        (num_pressure_levels, num_3d_fields), numpy.nan
    )
    stdev_matrix_3d = numpy.full(
        (num_pressure_levels, num_3d_fields), numpy.nan
    )
    mean_value_matrix_2d = numpy.full(
        (num_forecast_hours, num_2d_fields), numpy.nan
    )
    stdev_matrix_2d = numpy.full(
        (num_forecast_hours, num_2d_fields), numpy.nan
    )

    for j in range(num_3d_fields):
        for k in range(num_pressure_levels):
            f = field_names_3d[j]
            p = pressure_levels_mb[k]

            mean_value_matrix_3d[k, j] = (
                z_score_dict_dict_3d[f, p][MEAN_VALUE_KEY]
            )
            stdev_matrix_3d[k, j] = _get_standard_deviation(
                z_score_dict_dict_3d[f, p]
            )

            print((
                'Mean and standard deviation for {0:s} at {1:d} mb = '
                '{2:.4g}, {3:.4g}'
            ).format(
                field_names_3d[j],
                pressure_levels_mb[k],
                mean_value_matrix_3d[k, j],
                stdev_matrix_3d[k, j]
            ))

    for j in range(num_2d_fields):
        for k in range(num_forecast_hours):
            f = field_names_2d[j]
            h = forecast_hours[k]

            if f in ACCUM_PRECIP_FIELD_NAMES:
                mean_value_matrix_2d[k, j] = (
                    z_score_dict_dict_2d[f, h][MEAN_VALUE_KEY]
                )
                stdev_matrix_2d[k, j] = _get_standard_deviation(
                    z_score_dict_dict_2d[f, h]
                )

                print((
                    'Mean and standard deviation for {0:s} at {1:d}-hour lead'
                    ' = {2:.4g}, {3:.4g}'
                ).format(
                    field_names_2d[j],
                    forecast_hours[k],
                    mean_value_matrix_2d[k, j],
                    stdev_matrix_2d[k, j]
                ))
            else:
                mean_value_matrix_2d[k, j] = (
                    z_score_dict_dict_2d[f][MEAN_VALUE_KEY]
                )
                stdev_matrix_2d[k, j] = _get_standard_deviation(
                    z_score_dict_dict_2d[f]
                )

                print((
                    'Mean and standard deviation for {0:s} = {1:.4g}, {2:.4g}'
                ).format(
                    field_names_2d[j],
                    mean_value_matrix_2d[k, j],
                    stdev_matrix_2d[k, j]
                ))

    coord_dict = {
        gfs_utils.PRESSURE_LEVEL_DIM: pressure_levels_mb,
        gfs_utils.FORECAST_HOUR_DIM: forecast_hours,
        gfs_utils.FIELD_DIM_2D: field_names_2d,
        gfs_utils.FIELD_DIM_3D: field_names_3d
    }

    these_dim_3d = (gfs_utils.PRESSURE_LEVEL_DIM, gfs_utils.FIELD_DIM_3D)
    these_dim_2d = (gfs_utils.FORECAST_HOUR_DIM, gfs_utils.FIELD_DIM_2D)
    main_data_dict = {
        gfs_utils.MEAN_VALUE_KEY_3D: (
            these_dim_3d, mean_value_matrix_3d
        ),
        gfs_utils.STDEV_KEY_3D: (
            these_dim_3d, stdev_matrix_3d
        ),
        gfs_utils.MEAN_VALUE_KEY_2D: (
            these_dim_2d, mean_value_matrix_2d
        ),
        gfs_utils.STDEV_KEY_2D: (
            these_dim_2d, stdev_matrix_2d
        )
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def get_z_score_params_for_targets(target_file_names):
    """Computes z-score parameters for each target variable.

    z-score parameters = mean and standard deviation for every variable

    :param target_file_names: 1-D list of paths to target files (will be read by
        `canadian_fwi_io.read_file`).
    :return: z_score_param_table_xarray: xarray table with z-score parameters.
        Metadata and variable names in this table should make it self-
        explanatory.
    """

    first_fwi_table_xarray = canadian_fwi_io.read_file(target_file_names[0])
    first_fwit = first_fwi_table_xarray
    field_names = (
        first_fwit.coords[canadian_fwi_utils.FIELD_DIM].values.tolist()
    )

    z_score_dict_dict = {}
    for this_field_name in field_names:
        z_score_dict_dict[this_field_name] = {
            NUM_VALUES_KEY: 0,
            MEAN_VALUE_KEY: 0.,
            MEAN_OF_SQUARES_KEY: 0.
        }

    for this_target_file_name in target_file_names:
        print('Reading data from: "{0:s}"...'.format(this_target_file_name))
        this_fwi_table_xarray = canadian_fwi_io.read_file(this_target_file_name)
        this_fwit = this_fwi_table_xarray

        for j in range(
                len(this_fwit.coords[canadian_fwi_utils.FIELD_DIM].values)
        ):
            f = this_fwit.coords[canadian_fwi_utils.FIELD_DIM].values[j]
            z_score_dict_dict[f] = _update_z_score_params(
                z_score_param_dict=z_score_dict_dict[f],
                new_data_matrix=
                this_fwit[canadian_fwi_utils.DATA_KEY].values[..., j]
            )

    num_fields = len(field_names)
    mean_values = numpy.full(num_fields, numpy.nan)
    mean_squared_values = numpy.full(num_fields, numpy.nan)
    stdev_values = numpy.full(num_fields, numpy.nan)

    for j in range(num_fields):
        f = field_names[j]

        mean_values[j] = z_score_dict_dict[f][MEAN_VALUE_KEY]
        mean_squared_values[j] = z_score_dict_dict[f][MEAN_OF_SQUARES_KEY]
        stdev_values[j] = _get_standard_deviation(z_score_dict_dict[f])

        print((
            'Mean, squared mean, and standard deviation for {0:s} = '
            '{1:.4g}, {2:.4g}, {3:.4g}'
        ).format(
            field_names[j],
            mean_values[j], mean_squared_values[j], stdev_values[j]
        ))

    coord_dict = {canadian_fwi_utils.FIELD_DIM: field_names}

    these_dim = (canadian_fwi_utils.FIELD_DIM,)
    main_data_dict = {
        canadian_fwi_utils.MEAN_VALUE_KEY: (these_dim, mean_values),
        canadian_fwi_utils.MEAN_SQUARED_VALUE_KEY: (
            these_dim, mean_squared_values
        ),
        canadian_fwi_utils.STDEV_KEY: (these_dim, stdev_values)
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def get_z_score_params_for_era5_constants(era5_constant_file_name):
    """Computes z-score parameters for each ERA5 (time-constant) variable.

    z-score parameters = mean and standard deviation for every variable

    :param era5_constant_file_name: Path to input file (will be read by
        `era5_constant_io.read_file`).
    :return: z_score_param_table_xarray: xarray table with z-score parameters.
        Metadata and variable names in this table should make it self-
        explanatory.
    """

    print('Reading data from: "{0:s}"...'.format(era5_constant_file_name))
    era5_constant_table_xarray = era5_constant_io.read_file(
        era5_constant_file_name
    )

    e5ct = era5_constant_table_xarray
    field_names = e5ct.coords[era5_constant_utils.FIELD_DIM].values.tolist()

    z_score_dict_dict = {}
    for this_field_name in field_names:
        z_score_dict_dict[this_field_name] = {
            NUM_VALUES_KEY: 0,
            MEAN_VALUE_KEY: 0.,
            MEAN_OF_SQUARES_KEY: 0.
        }

    for j in range(len(field_names)):
        z_score_dict_dict[field_names[j]] = _update_z_score_params(
            z_score_param_dict=z_score_dict_dict[field_names[j]],
            new_data_matrix=e5ct[era5_constant_utils.DATA_KEY].values[..., j]
        )

    num_fields = len(field_names)
    mean_values = numpy.full(num_fields, numpy.nan)
    mean_squared_values = numpy.full(num_fields, numpy.nan)
    stdev_values = numpy.full(num_fields, numpy.nan)

    for j in range(num_fields):
        f = field_names[j]

        mean_values[j] = z_score_dict_dict[f][MEAN_VALUE_KEY]
        mean_squared_values[j] = z_score_dict_dict[f][MEAN_OF_SQUARES_KEY]
        stdev_values[j] = _get_standard_deviation(z_score_dict_dict[f])

        print((
            'Mean, squared mean, and standard deviation for {0:s} = '
            '{1:.4g}, {2:.4g}, {3:.4g}'
        ).format(
            field_names[j],
            mean_values[j], mean_squared_values[j], stdev_values[j]
        ))

    coord_dict = {era5_constant_utils.FIELD_DIM: field_names}

    these_dim = (era5_constant_utils.FIELD_DIM,)
    main_data_dict = {
        era5_constant_utils.MEAN_VALUE_KEY: (these_dim, mean_values),
        era5_constant_utils.MEAN_SQUARED_VALUE_KEY: (
            these_dim, mean_squared_values
        ),
        era5_constant_utils.STDEV_KEY: (these_dim, stdev_values)
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def denormalize_gfs_data_from_z_scores(gfs_table_xarray,
                                       z_score_param_table_xarray):
    """Returns GFS data from z-scores to physical units.

    :param gfs_table_xarray: xarray table with GFS data in z-scores.
    :param z_score_param_table_xarray: xarray table with normalization
        parameters (means and standard deviations), created by
        `get_z_score_params_for_gfs`.
    :return: gfs_table_xarray: Same as input but in physical units.
    """

    # TODO(thunderhoser): Still need unit test.

    gfst = gfs_table_xarray
    zspt = z_score_param_table_xarray

    field_names_3d = gfst.coords[gfs_utils.FIELD_DIM_3D].values.tolist()
    field_names_2d = gfst.coords[gfs_utils.FIELD_DIM_2D].values.tolist()
    pressure_levels_mb = numpy.round(
        gfst.coords[gfs_utils.PRESSURE_LEVEL_DIM].values
    ).astype(int)
    forecast_hours = numpy.round(
        gfst.coords[gfs_utils.FORECAST_HOUR_DIM].values
    ).astype(int)

    num_pressure_levels = len(pressure_levels_mb)
    num_forecast_hours = len(forecast_hours)
    num_3d_fields = len(field_names_3d)
    num_2d_fields = len(field_names_2d)

    data_matrix_3d = gfst[gfs_utils.DATA_KEY_3D].values
    data_matrix_2d = gfst[gfs_utils.DATA_KEY_2D].values

    for j in range(num_3d_fields):
        for k in range(num_pressure_levels):
            j_new = numpy.where(
                zspt.coords[gfs_utils.FIELD_DIM_3D].values == field_names_3d[j]
            )[0][0]

            k_new = numpy.where(
                numpy.round(
                    zspt.coords[gfs_utils.PRESSURE_LEVEL_DIM].values
                ).astype(int)
                == pressure_levels_mb[k]
            )[0][0]

            this_mean = zspt[gfs_utils.MEAN_VALUE_KEY_3D].values[k_new, j_new]
            this_stdev = zspt[gfs_utils.STDEV_KEY_3D].values[k_new, j_new]

            if numpy.isnan(this_stdev):
                data_matrix_3d[..., k, j] = this_mean
            else:
                data_matrix_3d[..., k, j] = (
                    this_mean + this_stdev * data_matrix_3d[..., k, j]
                )

    for j in range(num_2d_fields):
        j_new = numpy.where(
            zspt.coords[gfs_utils.FIELD_DIM_2D].values == field_names_2d[j]
        )[0][0]

        if field_names_2d[j] not in ACCUM_PRECIP_FIELD_NAMES:
            this_mean = zspt[gfs_utils.MEAN_VALUE_KEY_2D].values[0, j_new]
            this_stdev = zspt[gfs_utils.STDEV_KEY_2D].values[0, j_new]

            if numpy.isnan(this_stdev):
                data_matrix_2d[..., j] = this_mean
            else:
                data_matrix_2d[..., j] = (
                    this_mean + this_stdev * data_matrix_2d[..., j]
                )

            continue

        for k in range(num_forecast_hours):
            k_new = numpy.where(
                numpy.round(
                    zspt.coords[gfs_utils.FORECAST_HOUR_DIM].values
                ).astype(int)
                == forecast_hours[k]
            )[0][0]

            this_mean = zspt[gfs_utils.MEAN_VALUE_KEY_2D].values[
                k_new, j_new
            ]
            this_stdev = zspt[gfs_utils.STDEV_KEY_2D].values[k_new, j_new]

            if numpy.isnan(this_stdev):
                data_matrix_2d[k, ..., j] = this_mean
            else:
                data_matrix_2d[k, ..., j] = (
                    this_mean + this_stdev * data_matrix_2d[k, ..., j]
                )

    return gfs_table_xarray.assign({
        gfs_utils.DATA_KEY_3D: (
            gfs_table_xarray[gfs_utils.DATA_KEY_3D].dims,
            data_matrix_3d
        ),
        gfs_utils.DATA_KEY_2D: (
            gfs_table_xarray[gfs_utils.DATA_KEY_2D].dims,
            data_matrix_2d
        )
    })


def normalize_gfs_data_to_z_scores(gfs_table_xarray,
                                   z_score_param_table_xarray):
    """Normalizes GFS data from physical units to z-scores.

    :param gfs_table_xarray: xarray table with GFS data in physical units.
    :param z_score_param_table_xarray: xarray table with normalization
        parameters (means and standard deviations), created by
        `get_z_score_params_for_gfs`.
    :return: gfs_table_xarray: Same as input but in z-score units.
    """

    # TODO(thunderhoser): Still need unit test.

    gfst = gfs_table_xarray
    zspt = z_score_param_table_xarray

    field_names_3d = gfst.coords[gfs_utils.FIELD_DIM_3D].values.tolist()
    field_names_2d = gfst.coords[gfs_utils.FIELD_DIM_2D].values.tolist()
    pressure_levels_mb = numpy.round(
        gfst.coords[gfs_utils.PRESSURE_LEVEL_DIM].values
    ).astype(int)
    forecast_hours = numpy.round(
        gfst.coords[gfs_utils.FORECAST_HOUR_DIM].values
    ).astype(int)

    num_pressure_levels = len(pressure_levels_mb)
    num_forecast_hours = len(forecast_hours)
    num_3d_fields = len(field_names_3d)
    num_2d_fields = len(field_names_2d)

    data_matrix_3d = gfst[gfs_utils.DATA_KEY_3D].values
    data_matrix_2d = gfst[gfs_utils.DATA_KEY_2D].values

    for j in range(num_3d_fields):
        for k in range(num_pressure_levels):
            j_new = numpy.where(
                zspt.coords[gfs_utils.FIELD_DIM_3D].values == field_names_3d[j]
            )[0][0]

            k_new = numpy.where(
                numpy.round(
                    zspt.coords[gfs_utils.PRESSURE_LEVEL_DIM].values
                ).astype(int)
                == pressure_levels_mb[k]
            )[0][0]

            this_mean = zspt[gfs_utils.MEAN_VALUE_KEY_3D].values[k_new, j_new]
            this_stdev = zspt[gfs_utils.STDEV_KEY_3D].values[k_new, j_new]

            if numpy.isnan(this_stdev):
                data_matrix_3d[..., k, j] = 0.
            else:
                data_matrix_3d[..., k, j] = (
                    (data_matrix_3d[..., k, j] - this_mean) / this_stdev
                )

    for j in range(num_2d_fields):
        j_new = numpy.where(
            zspt.coords[gfs_utils.FIELD_DIM_2D].values == field_names_2d[j]
        )[0][0]

        if field_names_2d[j] not in ACCUM_PRECIP_FIELD_NAMES:
            this_mean = zspt[gfs_utils.MEAN_VALUE_KEY_2D].values[0, j_new]
            this_stdev = zspt[gfs_utils.STDEV_KEY_2D].values[0, j_new]

            if numpy.isnan(this_stdev):
                data_matrix_2d[..., j] = 0.
            else:
                data_matrix_2d[..., j] = (
                    (data_matrix_2d[..., j] - this_mean) / this_stdev
                )

            continue

        for k in range(num_forecast_hours):
            k_new = numpy.where(
                numpy.round(
                    zspt.coords[gfs_utils.FORECAST_HOUR_DIM].values
                ).astype(int)
                == forecast_hours[k]
            )[0][0]

            this_mean = zspt[gfs_utils.MEAN_VALUE_KEY_2D].values[
                k_new, j_new
            ]
            this_stdev = zspt[gfs_utils.STDEV_KEY_2D].values[k_new, j_new]

            if numpy.isnan(this_stdev):
                data_matrix_2d[k, ..., j] = 0.
            else:
                data_matrix_2d[k, ..., j] = (
                    (data_matrix_2d[k, ..., j] - this_mean) / this_stdev
                )

    return gfs_table_xarray.assign({
        gfs_utils.DATA_KEY_3D: (
            gfs_table_xarray[gfs_utils.DATA_KEY_3D].dims,
            data_matrix_3d
        ),
        gfs_utils.DATA_KEY_2D: (
            gfs_table_xarray[gfs_utils.DATA_KEY_2D].dims,
            data_matrix_2d
        )
    })


def normalize_targets_to_z_scores(fwi_table_xarray, z_score_param_table_xarray):
    """Normalizes target variables from physical units to z-scores.

    :param fwi_table_xarray: xarray table with FWI data in physical units.
    :param z_score_param_table_xarray: xarray table with normalization
        parameters (means and standard deviations), created by
        `get_z_score_params_for_targets`.
    :return: fwi_table_xarray: Same as input but in z-score units.
    """

    # TODO(thunderhoser): Still need denormalization method.
    # TODO(thunderhoser): Still need unit test.

    fwit = fwi_table_xarray
    zspt = z_score_param_table_xarray

    field_names = fwit.coords[canadian_fwi_utils.FIELD_DIM].values.tolist()
    num_fields = len(field_names)

    data_matrix = fwit[canadian_fwi_utils.DATA_KEY].values

    for j in range(num_fields):
        j_new = numpy.where(
            zspt.coords[canadian_fwi_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        this_mean = zspt[canadian_fwi_utils.MEAN_VALUE_KEY].values[j_new]
        this_stdev = zspt[canadian_fwi_utils.STDEV_KEY].values[j_new]

        if numpy.isnan(this_stdev):
            data_matrix[..., j] = 0.
        else:
            data_matrix[..., j] = (data_matrix[..., j] - this_mean) / this_stdev

    return fwi_table_xarray.assign({
        canadian_fwi_utils.DATA_KEY: (
            fwi_table_xarray[canadian_fwi_utils.DATA_KEY].dims,
            data_matrix
        )
    })


def normalize_gfs_fwi_forecasts_to_z_scores(
        daily_gfs_table_xarray, z_score_param_table_xarray):
    """Normalizes GFS-based FWI forecasts from physical units to z-scores.

    :param daily_gfs_table_xarray: xarray table with daily GFS-based FWI
        forecasts in physical units.
    :param z_score_param_table_xarray: xarray table with normalization
        parameters (means and standard deviations), created by
        `get_z_score_params_for_targets`.
    :return: daily_gfs_table_xarray: Same as input but in z-score units.
    """

    # TODO(thunderhoser): Still need denormalization method.
    # TODO(thunderhoser): Still need unit test.

    dgfst = daily_gfs_table_xarray
    zspt = z_score_param_table_xarray

    field_names = dgfst.coords[gfs_daily_utils.FIELD_DIM].values.tolist()
    num_fields = len(field_names)

    data_matrix = dgfst[gfs_daily_utils.DATA_KEY_2D].values

    for j in range(num_fields):
        j_new = numpy.where(
            zspt.coords[canadian_fwi_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        this_mean = zspt[canadian_fwi_utils.MEAN_VALUE_KEY].values[j_new]
        this_stdev = zspt[canadian_fwi_utils.STDEV_KEY].values[j_new]

        if numpy.isnan(this_stdev):
            data_matrix[..., j] = 0.
        else:
            data_matrix[..., j] = (data_matrix[..., j] - this_mean) / this_stdev

    return daily_gfs_table_xarray.assign({
        gfs_daily_utils.DATA_KEY_2D: (
            daily_gfs_table_xarray[gfs_daily_utils.DATA_KEY_2D].dims,
            data_matrix
        )
    })


def normalize_era5_constants_to_z_scores(
        era5_constant_table_xarray, z_score_param_table_xarray):
    """Normalizes ERA5 time-constant variables from physical units to z-scores.

    :param era5_constant_table_xarray: xarray table with ERA5 variables in
        physical units.
    :param z_score_param_table_xarray: xarray table with normalization
        parameters (means and standard deviations), created by
        `get_z_score_params_for_era5_constants`.
    :return: era5_constant_table_xarray: Same as input but in z-score units.
    """

    # TODO(thunderhoser): Still need denormalization method.
    # TODO(thunderhoser): Still need unit test.

    e5ct = era5_constant_table_xarray
    zspt = z_score_param_table_xarray

    field_names = e5ct.coords[era5_constant_utils.FIELD_DIM].values.tolist()
    num_fields = len(field_names)

    data_matrix = e5ct[era5_constant_utils.DATA_KEY].values

    for j in range(num_fields):
        j_new = numpy.where(
            zspt.coords[era5_constant_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        this_mean = zspt[era5_constant_utils.MEAN_VALUE_KEY].values[j_new]
        this_stdev = zspt[era5_constant_utils.STDEV_KEY].values[j_new]

        if numpy.isnan(this_stdev):
            data_matrix[..., j] = 0.
        else:
            data_matrix[..., j] = (data_matrix[..., j] - this_mean) / this_stdev

    return era5_constant_table_xarray.assign({
        era5_constant_utils.DATA_KEY: (
            era5_constant_table_xarray[era5_constant_utils.DATA_KEY].dims,
            data_matrix
        )
    })
