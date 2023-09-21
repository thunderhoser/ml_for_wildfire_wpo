"""Input/output methods for constant fields in raw ERA5 reanalysis data.

The raw file should be a GRIB file downloaded from the Copernicus Climate Data
Store (https://cds.climate.copernicus.eu/cdsapp#!/dataset/
reanalysis-era5-single-levels?tab=form) with the following options:

- Variables = "angle of subgrid-scale orography,"
              "anisotropy of subgrid-scale orography,"
              "geopotential," "land-sea mask,"
              "slope of subgrid-scale orography,"
              "standard deviation of filtered subgrid-scale orography," and
              "standard deviation of orography"

- Year = any one year
- Month = any one month
- Day = any one day
- Time = any one hour
- Geographical area = full globe
- Format = GRIB
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
import error_checking
import raw_gfs_io
import era5_constant_utils

GRID_LATITUDES_DEG_N = raw_gfs_io.GRID_LATITUDES_DEG_N
GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E = (
    raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
)

FIELD_NAME_TO_GRIB_NAME = {
    era5_constant_utils.SUBGRID_OROGRAPHY_SINE_NAME: None,
    era5_constant_utils.SUBGRID_OROGRAPHY_COSINE_NAME: None,
    era5_constant_utils.SUBGRID_OROGRAPHY_ANISOTROPY_NAME: 'ISOR:sfc',
    era5_constant_utils.GEOPOTENTIAL_NAME: 'Z:sfc',
    era5_constant_utils.LAND_SEA_MASK_NAME: 'LSM:sfc',
    era5_constant_utils.SUBGRID_OROGRAPHY_SLOPE_NAME: 'SLOR:sfc',
    era5_constant_utils.SUBGRID_OROGRAPHY_STDEV_NAME: 'SDFOR:sfc',
    era5_constant_utils.RESOLVED_OROGRAPHY_STDEV_NAME: 'SDOR:sfc'
}


def read_file(grib_file_name, desired_row_indices, desired_column_indices,
              wgrib_exe_name, temporary_dir_name):
    """Extracts all constant fields from GRIB file.

    M = number of rows in grid
    N = number of columns in grid

    :param grib_file_name: Path to input file.
    :param desired_row_indices: length-M numpy array with indices of desired
        grid rows from ERA5 data.
    :param desired_column_indices: length-N numpy array with indices of desired
        grid columns from ERA5 data.
    :param wgrib_exe_name: Path to wgrib executable.
    :param temporary_dir_name: Path to temporary directory for text files
        created by wgrib.
    :return: era5_constant_table_xarray: xarray table with all data.  Metadata
        and variable names should make this table self-explanatory.
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
    num_grid_rows = len(desired_row_indices)
    num_grid_columns = len(desired_column_indices)
    field_names = era5_constant_utils.ALL_FIELD_NAMES
    num_fields = len(field_names)

    data_matrix = numpy.full(
        (num_grid_rows, num_grid_columns, num_fields), numpy.nan
    )

    for f in range(num_fields):
        if field_names[f] == era5_constant_utils.SUBGRID_OROGRAPHY_SINE_NAME:
            continue

        if field_names[f] == era5_constant_utils.SUBGRID_OROGRAPHY_COSINE_NAME:
            grib_search_string = 'ANOR:sfc'
        else:
            grib_search_string = FIELD_NAME_TO_GRIB_NAME[field_names[f]]

        print('Reading line "{0:s}" from GRIB file: "{1:s}"...'.format(
            grib_search_string, grib_file_name
        ))
        this_data_matrix = grib_io.read_field_from_grib_file(
            grib_file_name=grib_file_name,
            field_name_grib1=grib_search_string,
            num_grid_rows=len(GRID_LATITUDES_DEG_N),
            num_grid_columns=len(GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E),
            wgrib_exe_name=wgrib_exe_name,
            wgrib2_exe_name=wgrib_exe_name,
            temporary_dir_name=temporary_dir_name,
            sentinel_value=None,
            raise_error_if_fails=True
        )

        this_data_matrix = numpy.flip(this_data_matrix, axis=0)
        this_data_matrix = this_data_matrix[desired_row_indices, :]
        this_data_matrix = this_data_matrix[:, desired_column_indices]
        assert not numpy.any(numpy.isnan(this_data_matrix))

        if field_names[f] == era5_constant_utils.SUBGRID_OROGRAPHY_COSINE_NAME:
            data_matrix[..., f] = numpy.cos(this_data_matrix)

            g = field_names.index(
                era5_constant_utils.SUBGRID_OROGRAPHY_SINE_NAME
            )
            data_matrix[..., g] = numpy.sin(this_data_matrix)
        else:
            data_matrix[..., f] = this_data_matrix + 0.

    coord_dict = {
        era5_constant_utils.LATITUDE_DIM:
            GRID_LATITUDES_DEG_N[desired_row_indices],
        era5_constant_utils.LONGITUDE_DIM:
            GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[desired_column_indices],
        era5_constant_utils.FIELD_DIM: field_names
    }

    these_dim = (
        era5_constant_utils.LATITUDE_DIM,
        era5_constant_utils.LONGITUDE_DIM,
        era5_constant_utils.FIELD_DIM
    )
    main_data_dict = {
        era5_constant_utils.DATA_KEY: (these_dim, data_matrix)
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)
