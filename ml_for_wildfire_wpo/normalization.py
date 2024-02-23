"""Helper methods for normalization."""

import os
import sys
import numpy
import xarray
import scipy.stats

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import gfs_io
import canadian_fwi_io
import era5_constant_io
import gfs_utils
import gfs_daily_utils
import canadian_fwi_utils
import era5_constant_utils

MIN_CUMULATIVE_DENSITY = 1e-6
MAX_CUMULATIVE_DENSITY = 1. - 1e-6
# MAX_CUMULATIVE_DENSITY = 0.9995  # To account for 16-bit floats.

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'
SAMPLE_VALUES_KEY = 'sample_values'

ACCUM_PRECIP_FIELD_NAMES = [
    gfs_utils.PRECIP_NAME, gfs_utils.CONVECTIVE_PRECIP_NAME
]


def _update_norm_params_1var_1file(norm_param_dict, new_data_matrix,
                                   num_sample_values_per_file, file_index):
    """Updates normalization params for one variable, based on one file.

    :param norm_param_dict: Dictionary with the following keys.
    norm_param_dict['num_values']: Number of values on which current
        estimates are based.
    norm_param_dict['mean_value']: Current mean.
    norm_param_dict['mean_of_squares']: Current mean of squared values.
    norm_param_dict['sample_values']: 1-D numpy array of sample values.

    :param new_data_matrix: numpy array with new values.  Will be used to
        update estimates in `norm_param_dict`.
    :param num_sample_values_per_file: Number of sample values to read.
    :param file_index: Index of current file.  If file_index == k, this means
        that the current file is the [k]th in the list.
    :return: norm_param_dict: Same as input, but with new estimates.
    """

    if numpy.all(numpy.isnan(new_data_matrix)):
        return norm_param_dict

    new_num_values = numpy.sum(numpy.invert(numpy.isnan(new_data_matrix)))

    these_means = numpy.array([
        norm_param_dict[MEAN_VALUE_KEY], numpy.nanmean(new_data_matrix)
    ])
    these_weights = numpy.array([
        norm_param_dict[NUM_VALUES_KEY], new_num_values
    ])
    norm_param_dict[MEAN_VALUE_KEY] = numpy.average(
        these_means, weights=these_weights
    )

    these_means = numpy.array([
        norm_param_dict[MEAN_OF_SQUARES_KEY],
        numpy.nanmean(new_data_matrix ** 2)
    ])
    these_weights = numpy.array([
        norm_param_dict[NUM_VALUES_KEY], new_num_values
    ])
    norm_param_dict[MEAN_OF_SQUARES_KEY] = numpy.average(
        these_means, weights=these_weights
    )

    norm_param_dict[NUM_VALUES_KEY] += new_num_values

    first_index = file_index * num_sample_values_per_file
    last_index = first_index + num_sample_values_per_file

    new_values_real = new_data_matrix[
        numpy.invert(numpy.isnan(new_data_matrix))
    ]
    numpy.random.shuffle(new_values_real)

    norm_param_dict[SAMPLE_VALUES_KEY][first_index:last_index] = (
        new_values_real[:num_sample_values_per_file]
    )

    return norm_param_dict


def _get_standard_deviation_1var(norm_param_dict):
    """Computes standard deviation for one variable.

    :param norm_param_dict: See doc for `_update_norm_params_1var_1file`.
    :return: standard_deviation: Standard deviation.
    """

    if norm_param_dict[NUM_VALUES_KEY] == 0:
        return numpy.nan

    multiplier = float(
        norm_param_dict[NUM_VALUES_KEY]
    ) / (norm_param_dict[NUM_VALUES_KEY] - 1)

    return numpy.sqrt(multiplier * (
        norm_param_dict[MEAN_OF_SQUARES_KEY] -
        norm_param_dict[MEAN_VALUE_KEY] ** 2
    ))


def _z_normalize_1var(data_values, reference_mean, reference_stdev):
    """Does z-score normalization for one variable.

    :param data_values: numpy array of data in physical units.
    :param reference_mean: Mean value from reference dataset.
    :param reference_stdev: Standard deviation from reference dataset.
    :return: data_values: Same as input but in z-scores now.
    """

    # TODO(thunderhoser): Still need unit test.

    if numpy.isnan(reference_stdev):
        data_values[:] = 0.
    else:
        data_values = (data_values - reference_mean) / reference_stdev

    return data_values


def _z_denormalize_1var(data_values, reference_mean, reference_stdev):
    """Does z-score *de*normalization for one variable.

    :param data_values: numpy array of data in z-score units.
    :param reference_mean: Mean value from reference dataset.
    :param reference_stdev: Standard deviation from reference dataset.
    :return: data_values: Same as input but in physical units now.
    """

    # TODO(thunderhoser): Still need unit test.

    if numpy.isnan(reference_stdev):
        data_values[:] = reference_mean
    else:
        data_values = reference_mean + reference_stdev * data_values

    return data_values


def _quantile_normalize_1var(data_values, reference_values_1d):
    """Does quantile normalization for one variable.

    :param data_values: numpy array of data in physical units.
    :param reference_values_1d: 1-D numpy array of reference values -- i.e.,
        values from reference dataset at equally spaced quantile levels.
    :return: data_values: Same as input but in z-scores now.
    """

    # TODO(thunderhoser): Still need unit test.

    data_values_1d = numpy.ravel(data_values)
    data_values_1d[numpy.invert(numpy.isfinite(data_values_1d))] = numpy.nan

    real_indices = numpy.where(
        numpy.invert(numpy.isnan(data_values_1d))
    )[0]

    if len(real_indices) == 0:
        return data_values

    if numpy.all(numpy.isnan(reference_values_1d)):
        data_values_1d[real_indices] = 0.
        return numpy.reshape(data_values_1d, data_values.shape)

    # The code below might fail due to non-unique values in reference_values_1d.

    # num_quantiles = len(reference_values_1d)
    # quantile_levels = numpy.linspace(0, 1, num=num_quantiles, dtype=float)
    #
    # interp_object = interp1d(
    #     x=reference_values_1d, y=quantile_levels, kind='linear',
    #     bounds_error=False, fill_value='extrapolate', assume_sorted=True
    # )
    # data_values_1d[real_indices] = interp_object(data_values_1d[real_indices])

    real_reference_values_1d = reference_values_1d[
        numpy.invert(numpy.isnan(reference_values_1d))
    ]

    search_indices = numpy.searchsorted(
        numpy.sort(real_reference_values_1d),
        data_values_1d[real_indices],
        side='left'
    ).astype(float)

    num_reference_vals = len(real_reference_values_1d)
    data_values_1d[real_indices] = search_indices / (num_reference_vals - 1)

    data_values_1d[real_indices] = numpy.minimum(
        data_values_1d[real_indices], MAX_CUMULATIVE_DENSITY
    )
    data_values_1d[real_indices] = numpy.maximum(
        data_values_1d[real_indices], MIN_CUMULATIVE_DENSITY
    )
    data_values_1d[real_indices] = scipy.stats.norm.ppf(
        data_values_1d[real_indices], loc=0., scale=1.
    )

    return numpy.reshape(data_values_1d, data_values.shape)


def _quantile_denormalize_1var(data_values, reference_values_1d):
    """Does quantile *de*normalization for one variable.

    :param data_values: numpy array of data in z-score units.
    :param reference_values_1d: 1-D numpy array of reference values -- i.e.,
        values from reference dataset at equally spaced quantile levels.
    :return: data_values: Same as input but in physical units now.
    """

    # TODO(thunderhoser): Still need unit test.

    data_values_1d = numpy.ravel(data_values)
    real_indices = numpy.where(
        numpy.invert(numpy.isnan(data_values_1d))
    )[0]

    if len(real_indices) == 0:
        return data_values
    if numpy.all(numpy.isnan(reference_values_1d)):
        return data_values

    data_values_1d[real_indices] = scipy.stats.norm.cdf(
        data_values_1d[real_indices], loc=0., scale=1.
    )
    real_reference_values_1d = reference_values_1d[
        numpy.invert(numpy.isnan(reference_values_1d))
    ]

    # Linear produces biased estimates (range of 0...0.1 in my test), while
    # lower produces unbiased estimates (range of -0.1...+0.1 in my test).
    data_values_1d[real_indices] = numpy.percentile(
        numpy.ravel(real_reference_values_1d),
        100 * data_values_1d[real_indices],
        interpolation='linear'
        # interpolation='lower'
    )

    return numpy.reshape(data_values_1d, data_values.shape)


def get_normalization_params_for_gfs(
        gfs_file_names, num_quantiles, num_sample_values_per_file):
    """Computes normalization parameters for GFS data.

    This method computes both z-score parameters (mean and standard deviation),
    as well as quantiles, for each variable.  The z-score parameters are used
    for simple z-score normalization, and the quantiles are used for quantile
    normalization (which is always followed by conversion to the standard normal
    distribution, i.e., z-scores).

    :param gfs_file_names: 1-D list of paths to GFS files (will be read by
        `gfs_io.read_file`).
    :param num_quantiles: Number of quantiles to store for each variable.  The
        quantile levels will be evenly spaced from 0 to 1 (i.e., the 0th to
        100th percentile).
    :param num_sample_values_per_file: Number of sample values per file to use
        for computing quantiles.  This value will be applied to each variable.
    :return: normalization_param_table_xarray: xarray table with normalization
        parameters.  Metadata and variable names in this table should make it
        self-explanatory.
    """

    error_checking.assert_is_geq(num_sample_values_per_file, 10)
    error_checking.assert_is_geq(num_quantiles, 100)

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

    norm_param_dict_dict_2d = {}
    num_sample_values_total = len(gfs_file_names) * num_sample_values_per_file

    for this_field_name in field_names_2d:
        if this_field_name not in ACCUM_PRECIP_FIELD_NAMES:
            norm_param_dict_dict_2d[this_field_name] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.,
                SAMPLE_VALUES_KEY:
                    numpy.full(num_sample_values_total, numpy.nan)
            }
            continue

        for this_forecast_hour in forecast_hours:
            norm_param_dict_dict_2d[this_field_name, this_forecast_hour] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.,
                SAMPLE_VALUES_KEY:
                    numpy.full(num_sample_values_total, numpy.nan)
            }

    norm_param_dict_dict_3d = {}

    for this_field_name in field_names_3d:
        for this_pressure_level_mb in pressure_levels_mb:
            norm_param_dict_dict_3d[this_field_name, this_pressure_level_mb] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.,
                SAMPLE_VALUES_KEY:
                    numpy.full(num_sample_values_total, numpy.nan)
            }

    for i in range(len(gfs_file_names)):
        print('Reading data from: "{0:s}"...'.format(gfs_file_names[i]))
        this_fwi_table_xarray = gfs_io.read_file(gfs_file_names[i])
        this_gfst = this_fwi_table_xarray

        for j in range(len(this_gfst.coords[gfs_utils.FIELD_DIM_2D].values)):
            f = this_gfst.coords[gfs_utils.FIELD_DIM_2D].values[j]

            if f not in ACCUM_PRECIP_FIELD_NAMES:
                norm_param_dict_dict_2d[f] = _update_norm_params_1var_1file(
                    norm_param_dict=norm_param_dict_dict_2d[f],
                    new_data_matrix=
                    this_gfst[gfs_utils.DATA_KEY_2D].values[..., j],
                    num_sample_values_per_file=num_sample_values_per_file,
                    file_index=i
                )

                continue

            for k in range(
                    len(this_gfst.coords[gfs_utils.FORECAST_HOUR_DIM].values)
            ):
                h = int(numpy.round(
                    this_gfst.coords[gfs_utils.FORECAST_HOUR_DIM].values[k]
                ))

                norm_param_dict_dict_2d[f, h] = _update_norm_params_1var_1file(
                    norm_param_dict=norm_param_dict_dict_2d[f, h],
                    new_data_matrix=
                    this_gfst[gfs_utils.DATA_KEY_2D].values[k, ..., j],
                    num_sample_values_per_file=num_sample_values_per_file,
                    file_index=i
                )

        for j in range(len(this_gfst.coords[gfs_utils.FIELD_DIM_3D].values)):
            for k in range(
                    len(this_gfst.coords[gfs_utils.PRESSURE_LEVEL_DIM].values)
            ):
                f = this_gfst.coords[gfs_utils.FIELD_DIM_3D].values[j]
                p = int(numpy.round(
                    this_gfst.coords[gfs_utils.PRESSURE_LEVEL_DIM].values[k]
                ))

                norm_param_dict_dict_3d[f, p] = _update_norm_params_1var_1file(
                    norm_param_dict=norm_param_dict_dict_3d[f, p],
                    new_data_matrix=
                    this_gfst[gfs_utils.DATA_KEY_3D].values[..., k, j],
                    num_sample_values_per_file=num_sample_values_per_file,
                    file_index=i
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
    quantile_matrix_3d = numpy.full(
        (num_pressure_levels, num_3d_fields, num_quantiles), numpy.nan
    )

    mean_value_matrix_2d = numpy.full(
        (num_forecast_hours, num_2d_fields), numpy.nan
    )
    stdev_matrix_2d = numpy.full(
        (num_forecast_hours, num_2d_fields), numpy.nan
    )
    quantile_matrix_2d = numpy.full(
        (num_forecast_hours, num_2d_fields, num_quantiles), numpy.nan
    )

    quantile_levels = numpy.linspace(0, 1, num=num_quantiles, dtype=float)

    for j in range(num_3d_fields):
        for k in range(num_pressure_levels):
            f = field_names_3d[j]
            p = pressure_levels_mb[k]

            mean_value_matrix_3d[k, j] = (
                norm_param_dict_dict_3d[f, p][MEAN_VALUE_KEY]
            )
            stdev_matrix_3d[k, j] = _get_standard_deviation_1var(
                norm_param_dict_dict_3d[f, p]
            )
            quantile_matrix_3d[k, j, :] = numpy.nanpercentile(
                norm_param_dict_dict_3d[f, p][SAMPLE_VALUES_KEY],
                100 * quantile_levels
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

            for m in range(num_quantiles)[::10]:
                print((
                    '{0:.2f}th percentile for {1:s} at {2:d} mb = {3:.4g}'
                ).format(
                    100 * quantile_levels[m],
                    field_names_3d[j],
                    pressure_levels_mb[k],
                    quantile_matrix_3d[k, j, m]
                ))

    for j in range(num_2d_fields):
        for k in range(num_forecast_hours):
            f = field_names_2d[j]
            h = forecast_hours[k]

            if f in ACCUM_PRECIP_FIELD_NAMES:
                mean_value_matrix_2d[k, j] = (
                    norm_param_dict_dict_2d[f, h][MEAN_VALUE_KEY]
                )
                stdev_matrix_2d[k, j] = _get_standard_deviation_1var(
                    norm_param_dict_dict_2d[f, h]
                )
                quantile_matrix_2d[k, j, :] = numpy.nanpercentile(
                    norm_param_dict_dict_2d[f, h][SAMPLE_VALUES_KEY],
                    100 * quantile_levels
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
                    norm_param_dict_dict_2d[f][MEAN_VALUE_KEY]
                )
                stdev_matrix_2d[k, j] = _get_standard_deviation_1var(
                    norm_param_dict_dict_2d[f]
                )
                quantile_matrix_2d[k, j, :] = numpy.nanpercentile(
                    norm_param_dict_dict_2d[f][SAMPLE_VALUES_KEY],
                    100 * quantile_levels
                )

                if k > 0:
                    continue

                print((
                    'Mean and standard deviation for {0:s} = {1:.4g}, {2:.4g}'
                ).format(
                    field_names_2d[j],
                    mean_value_matrix_2d[k, j],
                    stdev_matrix_2d[k, j]
                ))

            for m in range(num_quantiles)[::10]:
                print((
                    '{0:.2f}th percentile for {1:s}{2:s} = {3:.4g}'
                ).format(
                    100 * quantile_levels[m],
                    field_names_2d[j],
                    ' at {0:d}-hour lead'.format(forecast_hours[k])
                    if f in ACCUM_PRECIP_FIELD_NAMES
                    else '',
                    quantile_matrix_2d[k, j, m]
                ))

    coord_dict = {
        gfs_utils.PRESSURE_LEVEL_DIM: pressure_levels_mb,
        gfs_utils.FORECAST_HOUR_DIM: forecast_hours,
        gfs_utils.FIELD_DIM_2D: field_names_2d,
        gfs_utils.FIELD_DIM_3D: field_names_3d,
        gfs_utils.QUANTILE_LEVEL_DIM: quantile_levels
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

    these_dim_3d = (
        gfs_utils.PRESSURE_LEVEL_DIM, gfs_utils.FIELD_DIM_3D,
        gfs_utils.QUANTILE_LEVEL_DIM
    )
    these_dim_2d = (
        gfs_utils.FORECAST_HOUR_DIM, gfs_utils.FIELD_DIM_2D,
        gfs_utils.QUANTILE_LEVEL_DIM
    )
    main_data_dict.update({
        gfs_utils.QUANTILE_KEY_3D: (these_dim_3d, quantile_matrix_3d),
        gfs_utils.QUANTILE_KEY_2D: (these_dim_2d, quantile_matrix_2d)
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def get_normalization_params_for_targets(target_file_names, num_quantiles,
                                         num_sample_values_per_file):
    """Computes normalization parameters for each target variable.

    This method computes both z-score parameters (mean and standard deviation),
    as well as quantiles, for each variable -- just like
    `get_normalization_params_for_gfs`.

    :param target_file_names: 1-D list of paths to target files (will be read by
        `canadian_fwi_io.read_file`).
    :param num_quantiles: See doc for `get_normalization_params_for_gfs`.
    :param num_sample_values_per_file: Same.
    :return: norm_param_table_xarray: Same.
    """

    error_checking.assert_is_geq(num_sample_values_per_file, 10)
    error_checking.assert_is_geq(num_quantiles, 100)

    first_fwi_table_xarray = canadian_fwi_io.read_file(target_file_names[0])
    first_fwit = first_fwi_table_xarray
    field_names = (
        first_fwit.coords[canadian_fwi_utils.FIELD_DIM].values.tolist()
    )

    norm_param_dict_dict = {}
    num_sample_values_total = (
        len(target_file_names) * num_sample_values_per_file
    )

    for this_field_name in field_names:
        norm_param_dict_dict[this_field_name] = {
            NUM_VALUES_KEY: 0,
            MEAN_VALUE_KEY: 0.,
            MEAN_OF_SQUARES_KEY: 0.,
            SAMPLE_VALUES_KEY: numpy.full(num_sample_values_total, numpy.nan)
        }

    quantile_levels = numpy.linspace(0, 1, num=num_quantiles, dtype=float)

    for i in range(len(target_file_names)):
        print('Reading data from: "{0:s}"...'.format(target_file_names[i]))
        this_fwi_table_xarray = canadian_fwi_io.read_file(target_file_names[i])
        this_fwit = this_fwi_table_xarray

        for j in range(
                len(this_fwit.coords[canadian_fwi_utils.FIELD_DIM].values)
        ):
            f = this_fwit.coords[canadian_fwi_utils.FIELD_DIM].values[j]
            norm_param_dict_dict[f] = _update_norm_params_1var_1file(
                norm_param_dict=norm_param_dict_dict[f],
                new_data_matrix=
                this_fwit[canadian_fwi_utils.DATA_KEY].values[..., j],
                num_sample_values_per_file=num_sample_values_per_file,
                file_index=i
            )

    num_fields = len(field_names)
    mean_values = numpy.full(num_fields, numpy.nan)
    mean_squared_values = numpy.full(num_fields, numpy.nan)
    stdev_values = numpy.full(num_fields, numpy.nan)
    quantile_matrix = numpy.full((num_fields, num_quantiles), numpy.nan)

    for j in range(num_fields):
        f = field_names[j]

        mean_values[j] = norm_param_dict_dict[f][MEAN_VALUE_KEY]
        mean_squared_values[j] = norm_param_dict_dict[f][MEAN_OF_SQUARES_KEY]
        stdev_values[j] = _get_standard_deviation_1var(norm_param_dict_dict[f])
        quantile_matrix[j, :] = numpy.nanpercentile(
            norm_param_dict_dict[f][SAMPLE_VALUES_KEY],
            100 * quantile_levels
        )

        print((
            'Mean, squared mean, and standard deviation for {0:s} = '
            '{1:.4g}, {2:.4g}, {3:.4g}'
        ).format(
            field_names[j],
            mean_values[j], mean_squared_values[j], stdev_values[j]
        ))

        for m in range(num_quantiles)[::10]:
            print('{0:.2f}th percentile for {1:s} = {2:.4g}'.format(
                100 * quantile_levels[m],
                field_names[j],
                quantile_matrix[j, m]
            ))

    coord_dict = {
        canadian_fwi_utils.FIELD_DIM: field_names,
        canadian_fwi_utils.QUANTILE_LEVEL_DIM: quantile_levels
    }

    these_dim_1d = (canadian_fwi_utils.FIELD_DIM,)
    these_dim_2d = (
        canadian_fwi_utils.FIELD_DIM, canadian_fwi_utils.QUANTILE_LEVEL_DIM
    )

    main_data_dict = {
        canadian_fwi_utils.MEAN_VALUE_KEY: (these_dim_1d, mean_values),
        canadian_fwi_utils.MEAN_SQUARED_VALUE_KEY: (
            these_dim_1d, mean_squared_values
        ),
        canadian_fwi_utils.STDEV_KEY: (these_dim_1d, stdev_values),
        canadian_fwi_utils.QUANTILE_KEY: (these_dim_2d, quantile_matrix)
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def get_normalization_params_for_era5_const(era5_constant_file_name,
                                            num_quantiles):
    """Computes normalization parameters for each ERA5 (time-constant) variable.

    This method computes both z-score parameters (mean and standard deviation),
    as well as quantiles, for each variable -- just like
    `get_normalization_params_for_gfs`.

    :param era5_constant_file_name: Path to input file (will be read by
        `era5_constant_io.read_file`).
    :param num_quantiles: See doc for `get_normalization_params_for_gfs`.
    :return: norm_param_table_xarray: Same.
    """

    error_checking.assert_is_geq(num_quantiles, 100)

    print('Reading data from: "{0:s}"...'.format(era5_constant_file_name))
    era5_constant_table_xarray = era5_constant_io.read_file(
        era5_constant_file_name
    )

    e5ct = era5_constant_table_xarray
    field_names = e5ct.coords[era5_constant_utils.FIELD_DIM].values.tolist()
    num_sample_values_total = (
        e5ct[era5_constant_utils.DATA_KEY].values[..., 0].size
    )

    norm_param_dict_dict = {}
    for this_field_name in field_names:
        norm_param_dict_dict[this_field_name] = {
            NUM_VALUES_KEY: 0,
            MEAN_VALUE_KEY: 0.,
            MEAN_OF_SQUARES_KEY: 0.,
            SAMPLE_VALUES_KEY: numpy.full(num_sample_values_total, numpy.nan)
        }

    quantile_levels = numpy.linspace(0, 1, num=num_quantiles, dtype=float)

    for j in range(len(field_names)):
        norm_param_dict_dict[field_names[j]] = _update_norm_params_1var_1file(
            norm_param_dict=norm_param_dict_dict[field_names[j]],
            new_data_matrix=e5ct[era5_constant_utils.DATA_KEY].values[..., j],
            num_sample_values_per_file=num_sample_values_total,
            file_index=1
        )

    num_fields = len(field_names)
    mean_values = numpy.full(num_fields, numpy.nan)
    mean_squared_values = numpy.full(num_fields, numpy.nan)
    stdev_values = numpy.full(num_fields, numpy.nan)
    quantile_matrix = numpy.full((num_fields, num_quantiles), numpy.nan)

    for j in range(num_fields):
        f = field_names[j]

        mean_values[j] = norm_param_dict_dict[f][MEAN_VALUE_KEY]
        mean_squared_values[j] = norm_param_dict_dict[f][MEAN_OF_SQUARES_KEY]
        stdev_values[j] = _get_standard_deviation_1var(norm_param_dict_dict[f])
        quantile_matrix[j, :] = numpy.nanpercentile(
            norm_param_dict_dict[f][SAMPLE_VALUES_KEY],
            100 * quantile_levels
        )

        print((
            'Mean, squared mean, and standard deviation for {0:s} = '
            '{1:.4g}, {2:.4g}, {3:.4g}'
        ).format(
            field_names[j],
            mean_values[j], mean_squared_values[j], stdev_values[j]
        ))

        for m in range(num_quantiles)[::10]:
            print('{0:.2f}th percentile for {1:s} = {2:.4g}'.format(
                100 * quantile_levels[m],
                field_names[j],
                quantile_matrix[j, m]
            ))

    coord_dict = {
        era5_constant_utils.FIELD_DIM: field_names,
        era5_constant_utils.QUANTILE_LEVEL_DIM: quantile_levels
    }

    these_dim_1d = (era5_constant_utils.FIELD_DIM,)
    these_dim_2d = (
        era5_constant_utils.FIELD_DIM, era5_constant_utils.QUANTILE_LEVEL_DIM
    )

    main_data_dict = {
        era5_constant_utils.MEAN_VALUE_KEY: (these_dim_1d, mean_values),
        era5_constant_utils.MEAN_SQUARED_VALUE_KEY: (
            these_dim_1d, mean_squared_values
        ),
        era5_constant_utils.STDEV_KEY: (these_dim_1d, stdev_values),
        era5_constant_utils.QUANTILE_KEY: (these_dim_2d, quantile_matrix)
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def normalize_gfs_data(gfs_table_xarray, norm_param_table_xarray,
                       use_quantile_norm):
    """Normalizes GFS data.

    :param gfs_table_xarray: xarray table with GFS data in physical units.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, created by `get_normalization_params_for_gfs`.
    :param use_quantile_norm: Boolean flag.  If True, will use quantile
        normalization and then convert ranks to standard normal distribution.
        If False, will just use z-score normalization.
    :return: gfs_table_xarray: Same as input but normalized.
    """

    gfst = gfs_table_xarray
    npt = norm_param_table_xarray
    error_checking.assert_is_boolean(use_quantile_norm)

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
                npt.coords[gfs_utils.FIELD_DIM_3D].values == field_names_3d[j]
            )[0][0]

            k_new = numpy.where(
                numpy.round(
                    npt.coords[gfs_utils.PRESSURE_LEVEL_DIM].values
                ).astype(int)
                == pressure_levels_mb[k]
            )[0][0]

            if use_quantile_norm:
                data_matrix_3d[..., k, j] = _quantile_normalize_1var(
                    data_values=data_matrix_3d[..., k, j],
                    reference_values_1d=
                    npt[gfs_utils.QUANTILE_KEY_3D].values[k_new, j_new]
                )
            else:
                data_matrix_3d[..., k, j] = _z_normalize_1var(
                    data_values=data_matrix_3d[..., k, j],
                    reference_mean=
                    npt[gfs_utils.MEAN_VALUE_KEY_3D].values[k_new, j_new],
                    reference_stdev=
                    npt[gfs_utils.STDEV_KEY_3D].values[k_new, j_new]
                )

    for j in range(num_2d_fields):
        j_new = numpy.where(
            npt.coords[gfs_utils.FIELD_DIM_2D].values == field_names_2d[j]
        )[0][0]

        if field_names_2d[j] not in ACCUM_PRECIP_FIELD_NAMES:
            if use_quantile_norm:
                data_matrix_2d[..., j] = _quantile_normalize_1var(
                    data_values=data_matrix_2d[..., j],
                    reference_values_1d=
                    npt[gfs_utils.QUANTILE_KEY_2D].values[0, j_new, :]
                )
            else:
                data_matrix_2d[..., j] = _z_normalize_1var(
                    data_values=data_matrix_2d[..., j],
                    reference_mean=
                    npt[gfs_utils.MEAN_VALUE_KEY_2D].values[0, j_new],
                    reference_stdev=npt[gfs_utils.STDEV_KEY_2D].values[0, j_new]
                )

            continue

        for k in range(num_forecast_hours):
            k_new = numpy.where(
                numpy.round(
                    npt.coords[gfs_utils.FORECAST_HOUR_DIM].values
                ).astype(int)
                == forecast_hours[k]
            )[0][0]

            if use_quantile_norm:
                data_matrix_2d[k, ..., j] = _quantile_normalize_1var(
                    data_values=data_matrix_2d[k, ..., j],
                    reference_values_1d=
                    npt[gfs_utils.QUANTILE_KEY_2D].values[k_new, j_new, :]
                )
            else:
                data_matrix_2d[k, ..., j] = _z_normalize_1var(
                    data_values=data_matrix_2d[k, ..., j],
                    reference_mean=
                    npt[gfs_utils.MEAN_VALUE_KEY_2D].values[k_new, j_new],
                    reference_stdev=
                    npt[gfs_utils.STDEV_KEY_2D].values[k_new, j_new]
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


def denormalize_gfs_data(gfs_table_xarray, norm_param_table_xarray,
                         use_quantile_norm):
    """Denormalizes GFS data.

    :param gfs_table_xarray: xarray table with GFS data in z-scores.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, created by `get_normalization_params_for_gfs`.
    :param use_quantile_norm: Boolean flag.  If True, will assume that
        normalization method was quantile normalization followed by converting
        ranks to standard normal distribution.  If False, will assume just
        z-score normalization.
    :return: gfs_table_xarray: Same as input but in physical units.
    """

    gfst = gfs_table_xarray
    npt = norm_param_table_xarray
    error_checking.assert_is_boolean(use_quantile_norm)

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
                npt.coords[gfs_utils.FIELD_DIM_3D].values == field_names_3d[j]
            )[0][0]

            k_new = numpy.where(
                numpy.round(
                    npt.coords[gfs_utils.PRESSURE_LEVEL_DIM].values
                ).astype(int)
                == pressure_levels_mb[k]
            )[0][0]

            if use_quantile_norm:
                data_matrix_3d[..., k, j] = _quantile_denormalize_1var(
                    data_values=data_matrix_3d[..., k, j],
                    reference_values_1d=
                    npt[gfs_utils.QUANTILE_KEY_3D].values[k_new, j_new]
                )
            else:
                data_matrix_3d[..., k, j] = _z_denormalize_1var(
                    data_values=data_matrix_3d[..., k, j],
                    reference_mean=
                    npt[gfs_utils.MEAN_VALUE_KEY_3D].values[k_new, j_new],
                    reference_stdev=
                    npt[gfs_utils.STDEV_KEY_3D].values[k_new, j_new]
                )

    for j in range(num_2d_fields):
        j_new = numpy.where(
            npt.coords[gfs_utils.FIELD_DIM_2D].values == field_names_2d[j]
        )[0][0]

        if field_names_2d[j] not in ACCUM_PRECIP_FIELD_NAMES:
            if use_quantile_norm:
                data_matrix_2d[..., j] = _quantile_denormalize_1var(
                    data_values=data_matrix_2d[..., j],
                    reference_values_1d=
                    npt[gfs_utils.QUANTILE_KEY_2D].values[0, j_new, :]
                )
            else:
                data_matrix_2d[..., j] = _z_denormalize_1var(
                    data_values=data_matrix_2d[..., j],
                    reference_mean=
                    npt[gfs_utils.MEAN_VALUE_KEY_2D].values[0, j_new],
                    reference_stdev=npt[gfs_utils.STDEV_KEY_2D].values[0, j_new]
                )

            continue

        for k in range(num_forecast_hours):
            k_new = numpy.where(
                numpy.round(
                    npt.coords[gfs_utils.FORECAST_HOUR_DIM].values
                ).astype(int)
                == forecast_hours[k]
            )[0][0]

            if use_quantile_norm:
                data_matrix_2d[k, ..., j] = _quantile_denormalize_1var(
                    data_values=data_matrix_2d[k, ..., j],
                    reference_values_1d=
                    npt[gfs_utils.QUANTILE_KEY_2D].values[k_new, j_new, :]
                )
            else:
                data_matrix_2d[k, ..., j] = _z_denormalize_1var(
                    data_values=data_matrix_2d[k, ..., j],
                    reference_mean=
                    npt[gfs_utils.MEAN_VALUE_KEY_2D].values[k_new, j_new],
                    reference_stdev=
                    npt[gfs_utils.STDEV_KEY_2D].values[k_new, j_new]
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


def normalize_targets(fwi_table_xarray, norm_param_table_xarray,
                      use_quantile_norm):
    """Normalizes target variables.

    :param fwi_table_xarray: xarray table with FWI data in physical units.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, created by `get_normalization_params_for_targets`.
    :param use_quantile_norm: Boolean flag.  If True, will use quantile
        normalization and then convert ranks to standard normal distribution.
        If False, will just use z-score normalization.
    :return: fwi_table_xarray: Same as input but normalized.
    """

    # TODO(thunderhoser): Still need denormalization method.

    fwit = fwi_table_xarray
    npt = norm_param_table_xarray

    field_names = fwit.coords[canadian_fwi_utils.FIELD_DIM].values.tolist()
    num_fields = len(field_names)

    data_matrix = fwit[canadian_fwi_utils.DATA_KEY].values

    for j in range(num_fields):
        j_new = numpy.where(
            npt.coords[canadian_fwi_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        if use_quantile_norm:
            data_matrix[..., j] = _quantile_normalize_1var(
                data_values=data_matrix[..., j],
                reference_values_1d=
                npt[canadian_fwi_utils.QUANTILE_KEY].values[j_new, :]
            )
        else:
            data_matrix[..., j] = _z_normalize_1var(
                data_values=data_matrix[..., j],
                reference_mean=
                npt[canadian_fwi_utils.MEAN_VALUE_KEY].values[j_new],
                reference_stdev=
                npt[canadian_fwi_utils.STDEV_KEY].values[j_new]
            )

    return fwi_table_xarray.assign({
        canadian_fwi_utils.DATA_KEY: (
            fwi_table_xarray[canadian_fwi_utils.DATA_KEY].dims,
            data_matrix
        )
    })


def normalize_gfs_fwi_forecasts(daily_gfs_table_xarray, norm_param_table_xarray,
                                use_quantile_norm):
    """Normalizes GFS-based FWI forecasts.

    :param daily_gfs_table_xarray: xarray table with daily GFS-based FWI
        forecasts in physical units.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, created by `get_normalization_params_for_targets`.
    :param use_quantile_norm: Boolean flag.  If True, will use quantile
        normalization and then convert ranks to standard normal distribution.
        If False, will just use z-score normalization.
    :return: daily_gfs_table_xarray: Same as input but normalized.
    """

    # TODO(thunderhoser): Still need denormalization method.

    dgfst = daily_gfs_table_xarray
    npt = norm_param_table_xarray

    field_names = dgfst.coords[gfs_daily_utils.FIELD_DIM].values.tolist()
    num_fields = len(field_names)

    data_matrix = dgfst[gfs_daily_utils.DATA_KEY_2D].values

    for j in range(num_fields):
        j_new = numpy.where(
            npt.coords[canadian_fwi_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        if use_quantile_norm:
            data_matrix[..., j] = _quantile_normalize_1var(
                data_values=data_matrix[..., j],
                reference_values_1d=
                npt[canadian_fwi_utils.QUANTILE_KEY].values[j_new, :]
            )
        else:
            data_matrix[..., j] = _z_normalize_1var(
                data_values=data_matrix[..., j],
                reference_mean=
                npt[canadian_fwi_utils.MEAN_VALUE_KEY].values[j_new],
                reference_stdev=
                npt[canadian_fwi_utils.STDEV_KEY].values[j_new]
            )

    return daily_gfs_table_xarray.assign({
        gfs_daily_utils.DATA_KEY_2D: (
            daily_gfs_table_xarray[gfs_daily_utils.DATA_KEY_2D].dims,
            data_matrix
        )
    })


def normalize_era5_constants(
        era5_constant_table_xarray, norm_param_table_xarray, use_quantile_norm):
    """Normalizes ERA5 time-constant variables.

    :param era5_constant_table_xarray: xarray table with ERA5 variables in
        physical units.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, created by `get_normalization_params_for_era5_const`.
    :param use_quantile_norm: Boolean flag.  If True, will use quantile
        normalization and then convert ranks to standard normal distribution.
        If False, will just use z-score normalization.
    :return: era5_constant_table_xarray: Same as input but normalized.
    """

    # TODO(thunderhoser): Still need denormalization method.

    e5ct = era5_constant_table_xarray
    npt = norm_param_table_xarray

    field_names = e5ct.coords[era5_constant_utils.FIELD_DIM].values.tolist()
    num_fields = len(field_names)

    data_matrix = e5ct[era5_constant_utils.DATA_KEY].values

    for j in range(num_fields):
        j_new = numpy.where(
            npt.coords[era5_constant_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        if use_quantile_norm:
            data_matrix[..., j] = _quantile_normalize_1var(
                data_values=data_matrix[..., j],
                reference_values_1d=
                npt[era5_constant_utils.QUANTILE_KEY].values[j_new, :]
            )
        else:
            data_matrix[..., j] = _z_normalize_1var(
                data_values=data_matrix[..., j],
                reference_mean=
                npt[era5_constant_utils.MEAN_VALUE_KEY].values[j_new],
                reference_stdev=
                npt[era5_constant_utils.STDEV_KEY].values[j_new]
            )

    return era5_constant_table_xarray.assign({
        era5_constant_utils.DATA_KEY: (
            era5_constant_table_xarray[era5_constant_utils.DATA_KEY].dims,
            data_matrix
        )
    })
