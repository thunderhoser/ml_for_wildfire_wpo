"""Input/output methods for raw GFS data.

Each raw file should be a GRIB2 file downloaded from the NOAA High-performance
Storage System (HPSS) with the following options:

- One model run (init time)
- One forecast hour (valid time)
- Global domain
- 0.25-deg resolution
- 3-D variables (at 1000, 900, 800, 700, 600, 500, 400, 300, and 200 mb) =
  Temperature, specific humidity, geopotential height, u-wind, v-wind
- Variables at 2 m above ground level (AGL) = temperature, specific humidity,
  dewpoint
- Variables at 10 m AGL: u-wind, v-wind
- Variables at 0 m AGL: pressure
- Other variables: accumulated precip, accumulated convective precip,
  precipitable water, snow depth, water-equivalent snow depth, downward
  shortwave radiative flux, upward shortwave radiative flux, downward
  longwave radiative flux, upward longwave radiative flux, CAPE, soil
  temperature, volumetric soil-moisture fraction, vegetation type, soil type
"""

import os
import sys
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grib_io
import time_conversion
import number_rounding
import longitude_conversion as lng_conversion
import error_checking
import gfs_utils

TOLERANCE = 1e-6
MM_TO_METRES = 0.001

DATE_FORMAT = '%Y%m%d'
JULIAN_DATE_FORMAT = '%y%j'

LATITUDE_SPACING_DEG = 0.25
LONGITUDE_SPACING_DEG = 0.25

GRID_LATITUDES_DEG_N = numpy.linspace(-90, 90, num=721, dtype=float)
GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E = numpy.linspace(
    0, 359.75, num=1440, dtype=float
)
GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E = (
    lng_conversion.convert_lng_negative_in_west(
        GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )
)

PRESSURE_LEVELS_MB = numpy.linspace(200, 1000, num=9, dtype=int)

FIELD_NAME_TO_GRIB_NAME = {
    gfs_utils.TEMPERATURE_NAME: 'TMP',
    gfs_utils.SPECIFIC_HUMIDITY_NAME: 'SPFH',
    gfs_utils.GEOPOTENTIAL_HEIGHT_NAME: 'HGT',
    gfs_utils.U_WIND_NAME: 'UGRD',
    gfs_utils.V_WIND_NAME: 'VGRD',
    gfs_utils.TEMPERATURE_2METRE_NAME: 'TMP:2 m above ground',
    gfs_utils.DEWPOINT_2METRE_NAME: 'DPT:2 m above ground',
    gfs_utils.SPECIFIC_HUMIDITY_2METRE_NAME: 'SPFH:2 m above ground',
    gfs_utils.U_WIND_10METRE_NAME: 'UGRD:10 m above ground',
    gfs_utils.V_WIND_10METRE_NAME: 'VGRD:10 m above ground',
    gfs_utils.SURFACE_PRESSURE_NAME: 'PRES:surface',
    gfs_utils.PRECIP_NAME: 'APCP:surface',
    gfs_utils.CONVECTIVE_PRECIP_NAME: 'ACPCP:surface',
    gfs_utils.SOIL_TEMPERATURE_0TO10CM_NAME: 'TSOIL:0-0.1 m below ground',
    gfs_utils.SOIL_TEMPERATURE_10TO40CM_NAME: 'TSOIL:0.1-0.4 m below ground',
    gfs_utils.SOIL_TEMPERATURE_40TO100CM_NAME: 'TSOIL:0.4-1 m below ground',
    gfs_utils.SOIL_TEMPERATURE_100TO200CM_NAME: 'TSOIL:1-2 m below ground',
    gfs_utils.SOIL_MOISTURE_0TO10CM_NAME: 'SOILW:0-0.1 m below ground',
    gfs_utils.SOIL_MOISTURE_10TO40CM_NAME: 'SOILW:0.1-0.4 m below ground',
    gfs_utils.SOIL_MOISTURE_40TO100CM_NAME: 'SOILW:0.4-1 m below ground',
    gfs_utils.SOIL_MOISTURE_100TO200CM_NAME: 'SOILW:1-2 m below ground',
    gfs_utils.PRECIPITABLE_WATER_NAME: 'PWAT:entire atmosphere',
    gfs_utils.SNOW_DEPTH_NAME: 'SNOD:surface',
    gfs_utils.WATER_EQUIV_SNOW_DEPTH_NAME: 'WEASD:surface',
    gfs_utils.CAPE_FULL_COLUMN_NAME: 'CAPE:surface',
    gfs_utils.CAPE_BOTTOM_180MB_NAME: 'CAPE:180-0 mb above ground',
    gfs_utils.CAPE_BOTTOM_255MB_NAME: 'CAPE:255-0 mb above ground',
    gfs_utils.DOWNWARD_SURFACE_SHORTWAVE_FLUX_METRES: 'DSWRF:surface',
    gfs_utils.DOWNWARD_SURFACE_LONGWAVE_FLUX_METRES: 'DLWRF:surface',
    gfs_utils.UPWARD_SURFACE_SHORTWAVE_FLUX_METRES: 'USWRF:surface',
    gfs_utils.UPWARD_SURFACE_LONGWAVE_FLUX_METRES: 'ULWRF:surface',
    gfs_utils.UPWARD_TOA_SHORTWAVE_FLUX_METRES: 'USWRF:top of atmosphere',
    gfs_utils.UPWARD_TOA_LONGWAVE_FLUX_METRES: 'ULWRF:top of atmosphere'
}

FIELD_NAME_TO_CONV_FACTOR = {
    gfs_utils.TEMPERATURE_NAME: 1.,
    gfs_utils.SPECIFIC_HUMIDITY_NAME: 1.,
    gfs_utils.GEOPOTENTIAL_HEIGHT_NAME: 1.,
    gfs_utils.U_WIND_NAME: 1.,
    gfs_utils.V_WIND_NAME: 1.,
    gfs_utils.TEMPERATURE_2METRE_NAME: 1.,
    gfs_utils.DEWPOINT_2METRE_NAME: 1.,
    gfs_utils.SPECIFIC_HUMIDITY_2METRE_NAME: 1.,
    gfs_utils.U_WIND_10METRE_NAME: 1.,
    gfs_utils.V_WIND_10METRE_NAME: 1.,
    gfs_utils.SURFACE_PRESSURE_NAME: 1.,
    gfs_utils.PRECIP_NAME: MM_TO_METRES,
    gfs_utils.CONVECTIVE_PRECIP_NAME: MM_TO_METRES,
    gfs_utils.SOIL_TEMPERATURE_0TO10CM_NAME: 1.,
    gfs_utils.SOIL_TEMPERATURE_10TO40CM_NAME: 1.,
    gfs_utils.SOIL_TEMPERATURE_40TO100CM_NAME: 1.,
    gfs_utils.SOIL_TEMPERATURE_100TO200CM_NAME: 1.,
    gfs_utils.SOIL_MOISTURE_0TO10CM_NAME: 1.,
    gfs_utils.SOIL_MOISTURE_10TO40CM_NAME: 1.,
    gfs_utils.SOIL_MOISTURE_40TO100CM_NAME: 1.,
    gfs_utils.SOIL_MOISTURE_100TO200CM_NAME: 1.,
    gfs_utils.PRECIPITABLE_WATER_NAME: MM_TO_METRES,
    gfs_utils.SNOW_DEPTH_NAME: 1.,
    gfs_utils.WATER_EQUIV_SNOW_DEPTH_NAME: MM_TO_METRES,
    gfs_utils.CAPE_FULL_COLUMN_NAME: 1.,
    gfs_utils.CAPE_BOTTOM_180MB_NAME: 1.,
    gfs_utils.CAPE_BOTTOM_255MB_NAME: 1.,
    gfs_utils.DOWNWARD_SURFACE_SHORTWAVE_FLUX_METRES: 1.,
    gfs_utils.DOWNWARD_SURFACE_LONGWAVE_FLUX_METRES: 1.,
    gfs_utils.UPWARD_SURFACE_SHORTWAVE_FLUX_METRES: 1.,
    gfs_utils.UPWARD_SURFACE_LONGWAVE_FLUX_METRES: 1.,
    gfs_utils.UPWARD_TOA_SHORTWAVE_FLUX_METRES: 1.,
    gfs_utils.UPWARD_TOA_LONGWAVE_FLUX_METRES: 1.
}


def find_file(directory_name, init_date_string, forecast_hour,
              raise_error_if_missing=True):
    """Finds GRIB2 file with GFS data for one init time and one valid time.

    :param directory_name: Path to input directory.
    :param init_date_string: Initialization date (format "yyyymmdd").  The model
        run should be started at 00Z on this day.
    :param forecast_hour: Forecast hour (i.e., lead time) as an integer.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: gfs_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    _ = time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)
    error_checking.assert_is_integer(forecast_hour)
    error_checking.assert_is_geq(forecast_hour, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    init_date_unix_sec = time_conversion.string_to_unix_sec(
        init_date_string, DATE_FORMAT
    )
    init_date_string_julian = time_conversion.unix_sec_to_string(
        init_date_unix_sec, JULIAN_DATE_FORMAT
    )

    gfs_file_name = '{0:s}/{1:s}/{2:s}{3:08d}'.format(
        directory_name,
        init_date_string,
        init_date_string_julian,
        forecast_hour
    )

    if raise_error_if_missing and not os.path.isfile(gfs_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            gfs_file_name
        )
        raise ValueError(error_string)

    return gfs_file_name


def file_name_to_init_date(gfs_file_name):
    """Parses initialization date from name of GFS file.

    :param gfs_file_name: File path.
    :return: init_date_string: Initialization date (format "yyyymmdd").
    """

    error_checking.assert_is_string(gfs_file_name)

    subdir_names = gfs_file_name.split('/')
    assert len(subdir_names) >= 2

    init_date_string = subdir_names[-2]
    _ = time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)

    return init_date_string


def file_name_to_forecast_hour(gfs_file_name):
    """Parses forecast hour from name of GFS file.

    :param gfs_file_name: File path.
    :return: forecast_hour: Forecast hour (i.e., lead time) as an integer.
    """

    error_checking.assert_is_string(gfs_file_name)

    extensionless_file_name = os.path.split(gfs_file_name)[1]
    assert len(extensionless_file_name) == 13

    forecast_hour = int(extensionless_file_name[-8:])
    error_checking.assert_is_integer(forecast_hour)
    error_checking.assert_is_geq(forecast_hour, 0)

    return forecast_hour


def read_file(grib2_file_name, desired_row_indices, desired_column_indices,
              wgrib2_exe_name, temporary_dir_name):
    """Reads GFS data from GRIB2 file into xarray table.

    :param grib2_file_name: Path to input file.
    :param desired_row_indices: 1-D numpy array with indices of desired grid
        rows.
    :param desired_column_indices: 1-D numpy array with indices of desired grid
        columns.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param temporary_dir_name: Path to temporary directory for text files
        created by wgrib2.
    :return: gfs_table_xarray: xarray table with all data.  Metadata and
        variable names should make this table self-explanatory.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(
        desired_column_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_column_indices, len(GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E)
    )

    error_checking.assert_is_numpy_array(
        desired_row_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_row_indices)
    error_checking.assert_is_geq_numpy_array(desired_row_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_row_indices, len(GRID_LATITUDES_DEG_N)
    )
    error_checking.assert_equals_numpy_array(numpy.diff(desired_row_indices), 1)

    # Do actual stuff.
    field_names_3d = gfs_utils.ALL_3D_FIELD_NAMES
    field_names_2d = gfs_utils.ALL_2D_FIELD_NAMES

    num_grid_rows = len(desired_row_indices)
    num_grid_columns = len(desired_column_indices)
    num_3d_field_names = len(field_names_3d)
    num_pressure_levels = len(PRESSURE_LEVELS_MB)
    these_dim = (
        num_grid_rows, num_grid_columns, num_pressure_levels, num_3d_field_names
    )
    data_matrix_3d = numpy.full(these_dim, numpy.nan)

    num_2d_field_names = len(field_names_2d)
    data_matrix_2d = numpy.full(
        (num_grid_rows, num_grid_columns, num_2d_field_names), numpy.nan
    )

    for f in range(num_3d_field_names):
        for p in range(num_pressure_levels):
            this_search_string = '{0:s}:{1:d} mb'.format(
                FIELD_NAME_TO_GRIB_NAME[field_names_3d[f]],
                PRESSURE_LEVELS_MB[p]
            )

            print('Reading line "{0:s}" from GRIB2 file: "{1:s}"...'.format(
                this_search_string, grib2_file_name
            ))
            this_data_matrix = grib_io.read_field_from_grib_file(
                grib_file_name=grib2_file_name,
                field_name_grib1=this_search_string,
                num_grid_rows=num_grid_rows,
                num_grid_columns=num_grid_columns,
                wgrib_exe_name=wgrib2_exe_name,
                wgrib2_exe_name=wgrib2_exe_name,
                temporary_dir_name=temporary_dir_name,
                sentinel_value=None,
                raise_error_if_fails=True
            )

            this_data_matrix = this_data_matrix[desired_row_indices, :]
            this_data_matrix = this_data_matrix[:, desired_column_indices]
            assert not numpy.any(numpy.isnan(this_data_matrix))

            data_matrix_3d[..., f, p] = (
                this_data_matrix * FIELD_NAME_TO_CONV_FACTOR[field_names_3d[f]]
            )

    for f in range(num_2d_field_names):
        print('Reading line "{0:s}" from GRIB2 file: "{1:s}"...'.format(
            FIELD_NAME_TO_GRIB_NAME[field_names_2d[f]],
            grib2_file_name
        ))
        this_data_matrix = grib_io.read_field_from_grib_file(
            grib_file_name=grib2_file_name,
            field_name_grib1=FIELD_NAME_TO_GRIB_NAME[field_names_2d[f]],
            num_grid_rows=num_grid_rows,
            num_grid_columns=num_grid_columns,
            wgrib_exe_name=wgrib2_exe_name,
            wgrib2_exe_name=wgrib2_exe_name,
            temporary_dir_name=temporary_dir_name,
            sentinel_value=None,
            raise_error_if_fails=True
        )

        this_data_matrix = this_data_matrix[desired_row_indices, :]
        this_data_matrix = this_data_matrix[:, desired_column_indices]
        assert not numpy.any(numpy.isnan(this_data_matrix))

        data_matrix_2d[..., f] = (
            this_data_matrix * FIELD_NAME_TO_CONV_FACTOR[field_names_2d[f]]
        )

    forecast_hour = file_name_to_forecast_hour(grib2_file_name)

    coord_dict = {
        gfs_utils.FORECAST_HOUR_DIM: numpy.array([forecast_hour], dtype=int),
        gfs_utils.LATITUDE_DIM: GRID_LATITUDES_DEG_N[desired_row_indices],
        gfs_utils.LONGITUDE_DIM:
            GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[desired_column_indices],
        gfs_utils.PRESSURE_LEVEL_DIM: PRESSURE_LEVELS_MB,
        gfs_utils.FIELD_DIM_2D: field_names_2d,
        gfs_utils.FIELD_DIM_3D: field_names_3d
    }

    these_dim_3d = (
        gfs_utils.FORECAST_HOUR_DIM, gfs_utils.LATITUDE_DIM,
        gfs_utils.LONGITUDE_DIM, gfs_utils.PRESSURE_LEVEL_DIM,
        gfs_utils.FIELD_DIM_3D
    )
    these_dim_2d = (
        gfs_utils.FORECAST_HOUR_DIM, gfs_utils.LATITUDE_DIM,
        gfs_utils.LONGITUDE_DIM, gfs_utils.FIELD_DIM_2D
    )
    main_data_dict = {
        gfs_utils.DATA_KEY_3D: (
            these_dim_3d, numpy.expand_dims(data_matrix_3d, axis=0)
        ),
        gfs_utils.DATA_KEY_2D: (
            these_dim_2d, numpy.expand_dims(data_matrix_2d, axis=0)
        )
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def desired_longitudes_to_columns(start_longitude_deg_e, end_longitude_deg_e):
    """Converts desired longitudes to desired grid columns.

    :param start_longitude_deg_e: Longitude at start of desired range.  This may
        be in either format (positive or negative values in western hemisphere).
    :param end_longitude_deg_e: Longitude at end of desired range.  This may
        be in either format.
    :return: desired_column_indices: 1-D numpy array with indices of desired
        columns.
    """

    start_longitude_deg_e = number_rounding.floor_to_nearest(
        start_longitude_deg_e, LONGITUDE_SPACING_DEG
    )
    end_longitude_deg_e = number_rounding.ceiling_to_nearest(
        end_longitude_deg_e, LONGITUDE_SPACING_DEG
    )
    error_checking.assert_is_greater(
        numpy.absolute(start_longitude_deg_e - end_longitude_deg_e),
        TOLERANCE
    )

    start_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        start_longitude_deg_e, allow_nan=False
    )
    end_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        end_longitude_deg_e, allow_nan=False
    )

    if end_longitude_deg_e > start_longitude_deg_e:
        are_longitudes_positive_in_west = True
    else:
        start_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            start_longitude_deg_e, allow_nan=False
        )
        end_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            end_longitude_deg_e, allow_nan=False
        )
        are_longitudes_positive_in_west = False

    if are_longitudes_positive_in_west:
        grid_longitudes_deg_e = GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    else:
        grid_longitudes_deg_e = GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E

    num_longitudes = 1 + int(numpy.absolute(numpy.round(
        (end_longitude_deg_e - start_longitude_deg_e) / LONGITUDE_SPACING_DEG
    )))

    if end_longitude_deg_e > start_longitude_deg_e:
        desired_longitudes_deg_e = numpy.linspace(
            start_longitude_deg_e, end_longitude_deg_e, num=num_longitudes
        )

        desired_column_indices = numpy.array([
            numpy.where(
                numpy.absolute(grid_longitudes_deg_e - d) <= TOLERANCE
            )[0][0]
            for d in desired_longitudes_deg_e
        ], dtype=int)
    else:
        undesired_longitudes_deg_e = numpy.linspace(
            end_longitude_deg_e, start_longitude_deg_e, num=num_longitudes
        )[1:-1]

        undesired_column_indices = numpy.array([
            numpy.where(
                numpy.absolute(grid_longitudes_deg_e - d) <= TOLERANCE
            )[0][0]
            for d in undesired_longitudes_deg_e
        ], dtype=int)

        all_column_indices = numpy.linspace(
            0, len(grid_longitudes_deg_e) - 1, num=len(grid_longitudes_deg_e),
            dtype=int
        )
        desired_column_indices = (
            set(all_column_indices.tolist()) -
            set(undesired_column_indices.tolist())
        )
        desired_column_indices = numpy.array(
            list(desired_column_indices), dtype=int
        )

        break_index = 1 + numpy.where(
            numpy.diff(desired_column_indices) > 1
        )[0][0]

        desired_column_indices = numpy.concatenate((
            desired_column_indices[break_index:],
            desired_column_indices[:break_index]
        ))

    return desired_column_indices


def desired_latitudes_to_rows(start_latitude_deg_n, end_latitude_deg_n):
    """Converts desired latitudes to desired grid rows.

    :param start_latitude_deg_n: Latitude at start of desired range (deg north).
    :param end_latitude_deg_n: Latitude at end of desired range (deg north).
    :return: desired_row_indices: 1-D numpy array with indices of desired rows.
    """

    start_latitude_deg_n = number_rounding.floor_to_nearest(
        start_latitude_deg_n, LONGITUDE_SPACING_DEG
    )
    end_latitude_deg_n = number_rounding.ceiling_to_nearest(
        end_latitude_deg_n, LONGITUDE_SPACING_DEG
    )
    error_checking.assert_is_greater(
        numpy.absolute(start_latitude_deg_n - end_latitude_deg_n),
        TOLERANCE
    )

    error_checking.assert_is_valid_latitude(start_latitude_deg_n)
    error_checking.assert_is_valid_latitude(end_latitude_deg_n)

    num_latitudes = 1 + int(numpy.absolute(numpy.round(
        (end_latitude_deg_n - start_latitude_deg_n) / LATITUDE_SPACING_DEG
    )))
    desired_latitudes_deg_n = numpy.linspace(
        start_latitude_deg_n, end_latitude_deg_n, num=num_latitudes
    )
    desired_row_indices = numpy.array([
        numpy.where(numpy.absolute(GRID_LATITUDES_DEG_N - d) <= TOLERANCE)[0][0]
        for d in desired_latitudes_deg_n
    ], dtype=int)

    return desired_row_indices
