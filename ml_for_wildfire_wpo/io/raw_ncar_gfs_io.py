"""Input/output methods for raw GFS data downloaded from NCAR RDA.

Each raw file should be a GRIB2 file downloaded from the NCAR Research Data
Archive (https://rda.ucar.edu/datasets/ds084.1/) with the following
specifications:

- One model run (init time) at 0000 UTC
- One forecast hour (valid time)
- Global domain
- 0.25-deg resolution
- Four variables: accumulated precip, accumulated convective precip,
  vegetation type, soil type
"""

import os
import warnings
import numpy
import xarray
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.io import raw_gfs_io
from ml_for_wildfire_wpo.utils import gfs_utils

DAYS_TO_HOURS = 24
DATE_FORMAT = '%Y%m%d'
SENTINEL_VALUE = 9.999e20

FIELD_NAMES_TO_READ = [
    gfs_utils.PRECIP_NAME,
    gfs_utils.CONVECTIVE_PRECIP_NAME,
    gfs_utils.VEGETATION_FRACTION_NAME,
    gfs_utils.SOIL_TYPE_NAME
]

FIELD_NAME_TO_GRIB_NAME = raw_gfs_io.FIELD_NAME_TO_GRIB_NAME
FIELD_NAME_TO_CONV_FACTOR = raw_gfs_io.FIELD_NAME_TO_CONV_FACTOR

GRID_LATITUDES_DEG_N = raw_gfs_io.GRID_LATITUDES_DEG_N
GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E = (
    raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
)
GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E = (
    raw_gfs_io.GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E
)


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

    gfs_file_name = '{0:s}/gfs.0p25.{1:s}00.f{2:03d}.grib2'.format(
        directory_name,
        init_date_string,
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

    pathless_file_name = os.path.split(gfs_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    init_date_string = extensionless_file_name.split('.')[2][:-2]
    _ = time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)

    return init_date_string


def file_name_to_forecast_hour(gfs_file_name):
    """Parses forecast hour from name of GFS file.

    :param gfs_file_name: File path.
    :return: forecast_hour: Forecast hour (i.e., lead time) as an integer.
    """

    error_checking.assert_is_string(gfs_file_name)

    pathless_file_name = os.path.split(gfs_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    forecast_hour_string = extensionless_file_name.split('.')[3][1:]
    forecast_hour = int(forecast_hour_string)
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
    forecast_hour = file_name_to_forecast_hour(grib2_file_name)
    field_names_2d = FIELD_NAMES_TO_READ

    num_grid_rows = len(desired_row_indices)
    num_grid_columns = len(desired_column_indices)
    num_2d_field_names = len(field_names_2d)
    data_matrix_2d = numpy.full(
        (num_grid_rows, num_grid_columns, num_2d_field_names), numpy.nan
    )

    for f in range(num_2d_field_names):
        if (
                forecast_hour == 0 and
                field_names_2d[f] in gfs_utils.NO_0HOUR_ANALYSIS_FIELD_NAMES
        ):
            continue

        if field_names_2d[f] in [
                gfs_utils.PRECIP_NAME, gfs_utils.CONVECTIVE_PRECIP_NAME
        ]:
            if numpy.mod(forecast_hour, DAYS_TO_HOURS) == 0:
                grib_search_string = '{0:s}:0-{1:d} day acc'.format(
                    FIELD_NAME_TO_GRIB_NAME[field_names_2d[f]],
                    int(numpy.round(float(forecast_hour) / DAYS_TO_HOURS))
                )
            else:
                grib_search_string = '{0:s}:0-{1:d} hour acc'.format(
                    FIELD_NAME_TO_GRIB_NAME[field_names_2d[f]],
                    forecast_hour
                )
        else:
            grib_search_string = FIELD_NAME_TO_GRIB_NAME[field_names_2d[f]]

        print('Reading line "{0:s}" from GRIB2 file: "{1:s}"...'.format(
            grib_search_string, grib2_file_name
        ))
        this_data_matrix = grib_io.read_field_from_grib_file(
            grib_file_name=grib2_file_name,
            field_name_grib1=grib_search_string,
            num_grid_rows=len(GRID_LATITUDES_DEG_N),
            num_grid_columns=len(GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E),
            wgrib_exe_name=wgrib2_exe_name,
            wgrib2_exe_name=wgrib2_exe_name,
            temporary_dir_name=temporary_dir_name,
            sentinel_value=SENTINEL_VALUE,
            raise_error_if_fails=
            field_names_2d[f] not in gfs_utils.MAYBE_MISSING_FIELD_NAMES
        )

        if this_data_matrix is None:
            warning_string = (
                'POTENTIAL ERROR: Cannot find line "{0:s}" in GRIB2 file: '
                '"{1:s}"'
            ).format(
                grib_search_string, grib2_file_name
            )

            warnings.warn(warning_string)
            continue

        this_data_matrix = this_data_matrix[desired_row_indices, :]
        this_data_matrix = this_data_matrix[:, desired_column_indices]
        # assert not numpy.any(numpy.isnan(this_data_matrix))

        data_matrix_2d[..., f] = (
            this_data_matrix * FIELD_NAME_TO_CONV_FACTOR[field_names_2d[f]]
        )

    coord_dict = {
        gfs_utils.FORECAST_HOUR_DIM: numpy.array([forecast_hour], dtype=int),
        gfs_utils.LATITUDE_DIM: GRID_LATITUDES_DEG_N[desired_row_indices],
        gfs_utils.LONGITUDE_DIM:
            GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[desired_column_indices],
        gfs_utils.FIELD_DIM_2D: field_names_2d
    }

    these_dim = (
        gfs_utils.FORECAST_HOUR_DIM, gfs_utils.LATITUDE_DIM,
        gfs_utils.LONGITUDE_DIM, gfs_utils.FIELD_DIM_2D
    )
    main_data_dict = {
        gfs_utils.DATA_KEY_2D: (
            these_dim, numpy.expand_dims(data_matrix_2d, axis=0)
        )
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)
