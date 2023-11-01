"""Helper methods for GFS model."""

import warnings
import numpy
import xarray
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6

DATE_FORMAT = '%Y%m%d'
HOURS_TO_SECONDS = 3600

FORECAST_HOUR_DIM = 'forecast_hour'
LATITUDE_DIM = 'latitude_deg_n'
LONGITUDE_DIM = 'longitude_deg_e'
PRESSURE_LEVEL_DIM = 'pressure_level_mb'
FIELD_DIM_2D = 'field_name_2d'
FIELD_DIM_3D = 'field_name_3d'

DATA_KEY_2D = 'data_matrix_2d'
DATA_KEY_3D = 'data_matrix_3d'

MEAN_VALUE_KEY_3D = 'mean_value_3d'
STDEV_KEY_3D = 'standard_deviation_3d'
MEAN_VALUE_KEY_2D = 'mean_value_2d'
STDEV_KEY_2D = 'standard_deviation_2d'

TEMPERATURE_NAME = 'temperature_kelvins'
SPECIFIC_HUMIDITY_NAME = 'specific_humidity_kg_kg01'
GEOPOTENTIAL_HEIGHT_NAME = 'geopotential_height_m_asl'
U_WIND_NAME = 'u_wind_m_s01'
V_WIND_NAME = 'v_wind_m_s01'

TEMPERATURE_2METRE_NAME = 'temperature_2m_agl_kelvins'
DEWPOINT_2METRE_NAME = 'dewpoint_2m_agl_kelvins'
SPECIFIC_HUMIDITY_2METRE_NAME = 'specific_humidity_2m_agl_kg_kg01'
U_WIND_10METRE_NAME = 'u_wind_10m_agl_m_s01'
V_WIND_10METRE_NAME = 'v_wind_10m_agl_m_s01'
SURFACE_PRESSURE_NAME = 'surface_pressure_pascals'

PRECIP_NAME = 'accumulated_precip_metres'
CONVECTIVE_PRECIP_NAME = 'accumulated_convective_precip_metres'

SOIL_TEMPERATURE_0TO10CM_NAME = 'soil_temperature_0to10cm_kelvins'
SOIL_TEMPERATURE_10TO40CM_NAME = 'soil_temperature_10to40cm_kelvins'
SOIL_TEMPERATURE_40TO100CM_NAME = 'soil_temperature_40to100cm_kelvins'
SOIL_TEMPERATURE_100TO200CM_NAME = 'soil_temperature_100to200cm_kelvins'
SOIL_MOISTURE_0TO10CM_NAME = 'volumetric_soil_moisture_fraction_0to10cm'
SOIL_MOISTURE_10TO40CM_NAME = 'volumetric_soil_moisture_fraction_10to40cm'
SOIL_MOISTURE_40TO100CM_NAME = 'volumetric_soil_moisture_fraction_40to100cm'
SOIL_MOISTURE_100TO200CM_NAME = 'volumetric_soil_moisture_fraction_100to200cm'

PRECIPITABLE_WATER_NAME = 'precipitable_water_metres'
SNOW_DEPTH_NAME = 'snow_depth_metres'
WATER_EQUIV_SNOW_DEPTH_NAME = 'water_equivalent_snow_depth_metres'
CAPE_FULL_COLUMN_NAME = 'conv_avail_pot_energy_full_column_j_kg01'
CAPE_BOTTOM_180MB_NAME = 'conv_avail_pot_energy_bottom_180mb_j_kg01'
CAPE_BOTTOM_255MB_NAME = 'conv_avail_pot_energy_bottom_255mb_j_kg01'

DOWNWARD_SURFACE_SHORTWAVE_FLUX_NAME = 'downward_surface_shortwave_flux_w_m02'
DOWNWARD_SURFACE_LONGWAVE_FLUX_NAME = 'downward_surface_longwave_flux_w_m02'
UPWARD_SURFACE_SHORTWAVE_FLUX_NAME = 'upward_surface_shortwave_flux_w_m02'
UPWARD_SURFACE_LONGWAVE_FLUX_NAME = 'upward_surface_longwave_flux_w_m02'
UPWARD_TOA_SHORTWAVE_FLUX_NAME = 'upward_toa_shortwave_flux_w_m02'
UPWARD_TOA_LONGWAVE_FLUX_NAME = 'upward_toa_longwave_flux_w_m02'

VEGETATION_FRACTION_NAME = 'vegetation_fraction'
SOIL_TYPE_NAME = 'soil_type'

ALL_3D_FIELD_NAMES = [
    TEMPERATURE_NAME, SPECIFIC_HUMIDITY_NAME, GEOPOTENTIAL_HEIGHT_NAME,
    U_WIND_NAME, V_WIND_NAME
]

ALL_2D_FIELD_NAMES = [
    TEMPERATURE_2METRE_NAME, DEWPOINT_2METRE_NAME,
    SPECIFIC_HUMIDITY_2METRE_NAME, U_WIND_10METRE_NAME, V_WIND_10METRE_NAME,
    SURFACE_PRESSURE_NAME,
    PRECIP_NAME, CONVECTIVE_PRECIP_NAME,
    SOIL_TEMPERATURE_0TO10CM_NAME, SOIL_TEMPERATURE_10TO40CM_NAME,
    SOIL_TEMPERATURE_40TO100CM_NAME, SOIL_TEMPERATURE_100TO200CM_NAME,
    SOIL_MOISTURE_0TO10CM_NAME, SOIL_MOISTURE_10TO40CM_NAME,
    SOIL_MOISTURE_40TO100CM_NAME, SOIL_MOISTURE_100TO200CM_NAME,
    PRECIPITABLE_WATER_NAME, SNOW_DEPTH_NAME, WATER_EQUIV_SNOW_DEPTH_NAME,
    CAPE_FULL_COLUMN_NAME, CAPE_BOTTOM_180MB_NAME, CAPE_BOTTOM_255MB_NAME,
    DOWNWARD_SURFACE_SHORTWAVE_FLUX_NAME,
    DOWNWARD_SURFACE_LONGWAVE_FLUX_NAME,
    UPWARD_SURFACE_SHORTWAVE_FLUX_NAME, UPWARD_SURFACE_LONGWAVE_FLUX_NAME,
    UPWARD_TOA_SHORTWAVE_FLUX_NAME, UPWARD_TOA_LONGWAVE_FLUX_NAME,
    VEGETATION_FRACTION_NAME, SOIL_TYPE_NAME
]

ALL_FIELD_NAMES = ALL_3D_FIELD_NAMES + ALL_2D_FIELD_NAMES

NO_0HOUR_ANALYSIS_FIELD_NAMES = [
    PRECIP_NAME, CONVECTIVE_PRECIP_NAME,
    DOWNWARD_SURFACE_SHORTWAVE_FLUX_NAME,
    DOWNWARD_SURFACE_LONGWAVE_FLUX_NAME,
    UPWARD_SURFACE_SHORTWAVE_FLUX_NAME, UPWARD_SURFACE_LONGWAVE_FLUX_NAME,
    UPWARD_TOA_SHORTWAVE_FLUX_NAME, UPWARD_TOA_LONGWAVE_FLUX_NAME
]

MAYBE_MISSING_FIELD_NAMES = [
    PRECIP_NAME, CONVECTIVE_PRECIP_NAME,
    VEGETATION_FRACTION_NAME, SOIL_TYPE_NAME
]

ALL_FORECAST_HOURS = numpy.concatenate([
    numpy.linspace(0, 240, num=81, dtype=int),
    numpy.linspace(252, 384, num=12, dtype=int)
])


def check_field_name(field_name):
    """Ensures that field name is valid.

    :param field_name: String (must be in list `ALL_FIELD_NAMES`).
    :raises: ValueError: if `field_name not in ALL_FIELD_NAMES`.
    """

    error_checking.assert_is_string(field_name)
    if field_name in ALL_FIELD_NAMES:
        return

    error_string = (
        'Field name "{0:s}" is not in the list of accepted field names '
        '(below):\n{1:s}'
    ).format(
        field_name, str(ALL_FIELD_NAMES)
    )

    raise ValueError(error_string)


def concat_over_forecast_hours(gfs_tables_xarray):
    """Concatenates GFS tables over forecast hour.

    :param gfs_tables_xarray: 1-D list of xarray tables with GFS data.
    :return: gfs_table_xarray: Single xarray table with GFS data.
    """

    return xarray.concat(
        gfs_tables_xarray, dim=FORECAST_HOUR_DIM, data_vars='all',
        coords='minimal', compat='identical', join='exact'
    )


def subset_by_row(gfs_table_xarray, desired_row_indices):
    """Subsets GFS table by grid row.

    :param gfs_table_xarray: xarray table with GFS forecasts.
    :param desired_row_indices: 1-D numpy array with indices of desired rows.
    :return: gfs_table_xarray: Same as input but maybe with fewer rows.
    """

    error_checking.assert_is_numpy_array(desired_row_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_row_indices)
    error_checking.assert_is_geq_numpy_array(desired_row_indices, 0)

    return gfs_table_xarray.isel({LATITUDE_DIM: desired_row_indices})


def subset_by_column(gfs_table_xarray, desired_column_indices):
    """Subsets GFS table by grid column.

    :param gfs_table_xarray: xarray table with GFS forecasts.
    :param desired_column_indices: 1-D numpy array with indices of desired
        columns.
    :return: gfs_table_xarray: Same as input but maybe with fewer columns.
    """

    error_checking.assert_is_numpy_array(
        desired_column_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)

    return gfs_table_xarray.isel({LONGITUDE_DIM: desired_column_indices})


def subset_by_forecast_hour(gfs_table_xarray, desired_forecast_hours):
    """Subsets GFS table by forecast hour.

    :param gfs_table_xarray: xarray table with GFS forecasts.
    :param desired_forecast_hours: 1-D numpy array of desired forecast hours.
    :return: gfs_table_xarray: Same as input but maybe with fewer forecast
        hours.
    """

    error_checking.assert_is_numpy_array(
        desired_forecast_hours, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_forecast_hours)
    error_checking.assert_is_geq_numpy_array(desired_forecast_hours, 0)

    return gfs_table_xarray.sel({FORECAST_HOUR_DIM: desired_forecast_hours})


def get_field(gfs_table_xarray, field_name, pressure_levels_mb=None):
    """Extracts one field from GFS table.

    H = number of forecast hours
    M = number of rows in grid
    N = number of columns in grid
    P = number of pressure levels

    :param gfs_table_xarray: xarray table with GFS forecasts.
    :param field_name: Field name.
    :param pressure_levels_mb: length-P numpy array of pressure levels.  If the
        field is 2-D rather than 3-D, leave this argument alone.
    :return: data_matrix: H-by-M-by-N (if field is 2-D) or H-by-M-by-N-by-P (if
        field is 3-D) numpy array of data values.
    """

    check_field_name(field_name)

    if field_name in ALL_2D_FIELD_NAMES:
        k = numpy.where(
            gfs_table_xarray.coords[FIELD_DIM_2D].values == field_name
        )[0][0]
        return gfs_table_xarray[DATA_KEY_2D].values[..., k]

    k = numpy.where(
        gfs_table_xarray.coords[FIELD_DIM_3D].values == field_name
    )[0][0]

    js = numpy.array([
        numpy.where(
            gfs_table_xarray.coords[PRESSURE_LEVEL_DIM].values == p
        )[0][0]
        for p in pressure_levels_mb
    ], dtype=int)

    return gfs_table_xarray[DATA_KEY_3D].values[..., k][..., js]


def read_nonprecip_field_different_times(
        gfs_table_xarray, init_date_string, field_name,
        valid_time_matrix_unix_sec, pressure_level_mb=None):
    """Reads any field other than precip, with different valid time at each px.

    M = number of rows in grid
    N = number of columns in grid

    :param gfs_table_xarray: xarray table with GFS data for one model run, in
        format returned by `gfs_io.read_file`.
    :param init_date_string: Initialization date (format "yyyymmdd") for model
        run.  Will assume that the init time is 0000 UTC on this day.
    :param field_name: Field name.
    :param valid_time_matrix_unix_sec: M-by-N numpy array of valid times.  Will
        interpolate between forecast hours where necessary.
    :param pressure_level_mb: Pressure level.  If the field is 2-D rather than
        3-D, leave this argument alone.
    :return: data_matrix: M-by-N numpy array of data values.
    """

    # Check input args.
    init_time_unix_sec = time_conversion.string_to_unix_sec(
        init_date_string, DATE_FORMAT
    )

    num_grid_rows = len(gfs_table_xarray.coords[LATITUDE_DIM].values)
    num_grid_columns = len(gfs_table_xarray.coords[LONGITUDE_DIM].values)
    expected_dim = numpy.array([num_grid_rows, num_grid_columns], dtype=int)

    error_checking.assert_is_integer_numpy_array(valid_time_matrix_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_time_matrix_unix_sec, exact_dimensions=expected_dim
    )
    error_checking.assert_is_geq_numpy_array(
        valid_time_matrix_unix_sec, init_time_unix_sec
    )

    # Do actual stuff.
    data_matrix_orig_fcst_hours = get_field(
        gfs_table_xarray=gfs_table_xarray,
        field_name=field_name,
        pressure_levels_mb=(
            None if pressure_level_mb is None
            else numpy.array([pressure_level_mb], dtype=int)
        )
    )

    if pressure_level_mb is not None:
        data_matrix_orig_fcst_hours = data_matrix_orig_fcst_hours[..., 0]

    gfs_valid_times_unix_sec = (
        init_time_unix_sec +
        gfs_table_xarray.coords[FORECAST_HOUR_DIM].values * HOURS_TO_SECONDS
    )

    bad_hour_flags = numpy.any(
        numpy.isnan(data_matrix_orig_fcst_hours), axis=(1, 2)
    )
    good_hour_indices = numpy.where(numpy.invert(bad_hour_flags))[0]
    data_matrix_orig_fcst_hours = data_matrix_orig_fcst_hours[
        good_hour_indices, ...
    ]
    gfs_valid_times_unix_sec = gfs_valid_times_unix_sec[good_hour_indices]

    unique_desired_valid_times_unix_sec = numpy.unique(
        valid_time_matrix_unix_sec
    )

    interp_object = interp1d(
        x=gfs_valid_times_unix_sec,
        y=data_matrix_orig_fcst_hours,
        kind='linear', axis=0, assume_sorted=True, bounds_error=False,
        fill_value=(
            data_matrix_orig_fcst_hours[0, ...],
            data_matrix_orig_fcst_hours[-1, ...]
        )
    )
    interp_data_matrix_3d = interp_object(unique_desired_valid_times_unix_sec)

    interp_data_matrix_2d = numpy.full(
        (num_grid_rows, num_grid_columns), numpy.nan
    )

    for i in range(len(unique_desired_valid_times_unix_sec)):
        rowcol_indices = numpy.where(
            valid_time_matrix_unix_sec == unique_desired_valid_times_unix_sec[i]
        )
        interp_data_matrix_2d[rowcol_indices] = (
            interp_data_matrix_3d[i, ...][rowcol_indices]
        )

    assert not numpy.any(numpy.isnan(interp_data_matrix_2d))
    return interp_data_matrix_2d


def read_24hour_precip_different_times(
        gfs_table_xarray, init_date_string, valid_time_matrix_unix_sec):
    """Reads 24-hour accumulated precip, with different valid time at each px.

    :param gfs_table_xarray: See doc for `read_nonprecip_field_different_times`.
    :param init_date_string: Same.
    :param valid_time_matrix_unix_sec: Same.
    :return: data_matrix: Same.
    """

    # Check input args.
    init_time_unix_sec = time_conversion.string_to_unix_sec(
        init_date_string, DATE_FORMAT
    )

    num_grid_rows = len(gfs_table_xarray.coords[LATITUDE_DIM].values)
    num_grid_columns = len(gfs_table_xarray.coords[LONGITUDE_DIM].values)
    expected_dim = numpy.array([num_grid_rows, num_grid_columns], dtype=int)

    error_checking.assert_is_integer_numpy_array(valid_time_matrix_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_time_matrix_unix_sec, exact_dimensions=expected_dim
    )
    error_checking.assert_is_geq_numpy_array(
        valid_time_matrix_unix_sec, init_time_unix_sec + 24 * HOURS_TO_SECONDS
    )

    # Do actual stuff.
    orig_full_run_precip_matrix_metres = get_field(
        gfs_table_xarray=gfs_table_xarray,
        field_name=PRECIP_NAME,
        pressure_levels_mb=None
    )
    orig_full_run_precip_matrix_metres[0, ...] = 0.
    assert not numpy.any(numpy.isnan(orig_full_run_precip_matrix_metres))

    orig_forecast_hours = gfs_table_xarray.coords[FORECAST_HOUR_DIM].values

    if not numpy.all(numpy.isin(ALL_FORECAST_HOURS, orig_forecast_hours)):
        missing_hour_array_string = str(
            ALL_FORECAST_HOURS[
                numpy.isin(ALL_FORECAST_HOURS, orig_forecast_hours) == False
            ]
        )

        warning_string = (
            'POTENTIAL ERROR: Expected {0:d} forecast hours in GFS table.  '
            'Instead, got {1:d} forecast hours, with the following hours '
            'missing:\n{2:s}'
        ).format(
            len(ALL_FORECAST_HOURS),
            len(orig_forecast_hours),
            missing_hour_array_string
        )

        warnings.warn(warning_string)

        interp_object = interp1d(
            x=orig_forecast_hours,
            y=orig_full_run_precip_matrix_metres,
            kind='linear', axis=0, assume_sorted=True, bounds_error=False,
            fill_value=(
                orig_full_run_precip_matrix_metres[0, ...],
                orig_full_run_precip_matrix_metres[-1, ...]
            )
        )
        orig_full_run_precip_matrix_metres = interp_object(ALL_FORECAST_HOURS)
        orig_forecast_hours = ALL_FORECAST_HOURS + 0

    gfs_valid_times_unix_sec = (
        init_time_unix_sec + orig_forecast_hours * HOURS_TO_SECONDS
    )
    unique_desired_valid_times_unix_sec = numpy.unique(
        valid_time_matrix_unix_sec
    )

    interp_object = interp1d(
        x=gfs_valid_times_unix_sec,
        y=orig_full_run_precip_matrix_metres,
        kind='linear', axis=0, assume_sorted=True, bounds_error=False,
        fill_value=(
            orig_full_run_precip_matrix_metres[0, ...],
            orig_full_run_precip_matrix_metres[-1, ...]
        )
    )
    interp_end_precip_matrix_metres_3d = interp_object(
        unique_desired_valid_times_unix_sec
    )
    interp_start_precip_matrix_metres_3d = interp_object(
        unique_desired_valid_times_unix_sec - 24 * HOURS_TO_SECONDS
    )

    interp_24hour_precip_matrix_metres = numpy.full(
        (num_grid_rows, num_grid_columns), numpy.nan
    )

    for i in range(len(unique_desired_valid_times_unix_sec)):
        rowcol_indices = numpy.where(
            valid_time_matrix_unix_sec == unique_desired_valid_times_unix_sec[i]
        )
        interp_24hour_precip_matrix_metres[rowcol_indices] = (
            interp_end_precip_matrix_metres_3d[i, ...][rowcol_indices] -
            interp_start_precip_matrix_metres_3d[i, ...][rowcol_indices]
        )

    assert not numpy.any(numpy.isnan(interp_24hour_precip_matrix_metres))
    assert numpy.all(interp_24hour_precip_matrix_metres >= -1 * TOLERANCE)
    interp_24hour_precip_matrix_metres = numpy.maximum(
        interp_24hour_precip_matrix_metres, 0.
    )

    return interp_24hour_precip_matrix_metres


def precip_from_incremental_to_full_run(gfs_table_xarray):
    """Converts precip from incremental values to full-run values.

    "Incremental value" = an accumulation between two forecast hours
    "Full-run value" = accumulation over the entire model run, up to a forecast
                       hour

    This method works for both accumulated-precip variables: total and
    convective precip.

    :param gfs_table_xarray: xarray table with GFS forecasts.
    :return: gfs_table_xarray: Same as input but with full-run precip.
    """

    forecast_hours = gfs_table_xarray.coords[FORECAST_HOUR_DIM].values
    num_forecast_hours = len(forecast_hours)
    assert numpy.array_equal(forecast_hours, ALL_FORECAST_HOURS)

    data_matrix_2d = gfs_table_xarray[DATA_KEY_2D].values

    for j in range(num_forecast_hours)[::-1]:
        if forecast_hours[j] == 0:
            continue

        if forecast_hours[j] > 240:
            later_flags = forecast_hours > 240
            earlier_flags = numpy.logical_and(
                forecast_hours <= 240,
                numpy.mod(forecast_hours, 6) == 0
            )
            addend_flags = numpy.logical_or(earlier_flags, later_flags)
        else:
            addend_flags = numpy.mod(forecast_hours, 6) == 0

            if numpy.mod(forecast_hours[j], 6) == 3:
                addend_flags = numpy.logical_or(
                    addend_flags, forecast_hours == forecast_hours[j]
                )

        these_flags = numpy.logical_and(
            forecast_hours > 0,
            forecast_hours <= forecast_hours[j]
        )
        addend_flags = numpy.logical_and(addend_flags, these_flags)
        addend_indices = numpy.where(addend_flags)[0]

        for this_field_name in [PRECIP_NAME, CONVECTIVE_PRECIP_NAME]:
            k = numpy.where(
                gfs_table_xarray.coords[FIELD_DIM_2D].values == this_field_name
            )[0][0]

            data_matrix_2d[j, ..., k] = numpy.sum(
                data_matrix_2d[addend_indices, ..., k], axis=0
            )

    gfs_table_xarray = gfs_table_xarray.assign({
        DATA_KEY_2D: (
            gfs_table_xarray[DATA_KEY_2D].dims, data_matrix_2d
        )
    })

    return gfs_table_xarray
