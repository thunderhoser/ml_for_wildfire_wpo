"""Converts daily GFS data to TIF format.

TIF format is required by the fwiRaster script in R.
"""

import os
import sys
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import longitude_conversion as longitude_conv
import temperature_conversions as temperature_conv
import file_system_utils
import error_checking
import gfs_daily_io
import canadian_fwi_io
import misc_utils
import gfs_daily_utils
import canadian_fwi_utils

DATE_FORMAT = '%Y%m%d'

UNITLESS_TO_PERCENT = 100.
METRES_TO_MM = 1000.
METRES_PER_SECOND_TO_KMH = 3.6

FWI_FIELD_NAMES = [
    canadian_fwi_utils.FFMC_NAME,
    canadian_fwi_utils.DMC_NAME,
    canadian_fwi_utils.DC_NAME,
    canadian_fwi_utils.ISI_NAME,
    canadian_fwi_utils.BUI_NAME,
    canadian_fwi_utils.FWI_NAME,
    canadian_fwi_utils.DSR_NAME
]

DAILY_GFS_DIR_ARG_NAME = 'input_daily_gfs_dir_name'
CANADIAN_FWI_DIR_ARG_NAME = 'input_canadian_fwi_dir_name'
INIT_DATE_ARG_NAME = 'init_date_string'
MAX_LEAD_TIME_ARG_NAME = 'max_lead_time_days'
OUTPUT_DIR_ARG_NAME = 'output_tif_dir_name'

DAILY_GFS_DIR_HELP_STRING = (
    'Name of directory with daily GFS data.  Files therein will be found by '
    '`gfs_daily_io.find_file` and read by `gfs_daily_io.read_file`.'
)
CANADIAN_FWI_DIR_HELP_STRING = (
    'Name of directory with Canadian FWI data (reanalyses, not forecasts like '
    'the GFS).  Files therein will be found by `canadian_fwi_io.find_file` and '
    'read by `canadian_fwi_io.read_file`.  To convert GFS-forecast weather '
    'variables into GFS-forecast FWIs, we need initial FWI values, which will '
    'come from this dataset.'
)
INIT_DATE_HELP_STRING = (
    'Initialization date for GFS model run (format "yyyymmdd").'
)
MAX_LEAD_TIME_HELP_STRING = (
    'Max lead time.  Will compute GFS-forecast FWI values for all lead times '
    'from 1 day up to this max.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  TIF files will be written here (one per day) '
    'by rioxarray, to exact locations determined by `gfs_daily_io.find_file` '
    '(except with a different extension).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + DAILY_GFS_DIR_ARG_NAME, type=str, required=True,
    help=DAILY_GFS_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CANADIAN_FWI_DIR_ARG_NAME, type=str, required=True,
    help=CANADIAN_FWI_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_DATE_ARG_NAME, type=str, required=True,
    help=INIT_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LEAD_TIME_ARG_NAME, type=int, required=True,
    help=MAX_LEAD_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _find_output_file(directory_name, init_date_string, lead_time_days,
                      raise_error_if_missing=True):
    """Finds TIF file with GFS-based FWI inputs for 1 init time & 1 lead time.

    :param directory_name: Path to input directory.
    :param init_date_string: Initialization date (format "yyyymmdd").
    :param lead_time_days: Lead time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: tif_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    tif_file_name = (
        '{0:s}/init={1:s}/gfs_fwi_inputs_init={1:s}_lead={2:02d}days.tif'
    ).format(
        directory_name, init_date_string, lead_time_days
    )

    if raise_error_if_missing and not os.path.isfile(tif_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            tif_file_name
        )
        raise ValueError(error_string)

    return tif_file_name


def _run(daily_gfs_dir_name, canadian_fwi_dir_name, init_date_string,
         max_lead_time_days, output_dir_name):
    """Converts daily GFS data to TIF format.

    This is effectively the main method.

    :param daily_gfs_dir_name: See documentation at top of file.
    :param canadian_fwi_dir_name: Same.
    :param init_date_string: Same.
    :param max_lead_time_days: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    error_checking.assert_is_geq(max_lead_time_days, 1)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    daily_gfs_file_name = gfs_daily_io.find_file(
        directory_name=daily_gfs_dir_name,
        init_date_string=init_date_string,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(daily_gfs_file_name))
    daily_gfs_table_xarray = gfs_daily_io.read_file(daily_gfs_file_name)

    fwi_file_name = canadian_fwi_io.find_file(
        directory_name=canadian_fwi_dir_name,
        valid_date_string=init_date_string,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(fwi_file_name))
    fwi_table_xarray = canadian_fwi_io.read_file(fwi_file_name)

    desired_row_indices = misc_utils.desired_latitudes_to_rows(
        grid_latitudes_deg_n=
        daily_gfs_table_xarray.coords[gfs_daily_io.LATITUDE_DIM].values,
        start_latitude_deg_n=
        fwi_table_xarray.coords[canadian_fwi_utils.LATITUDE_DIM].values[0],
        end_latitude_deg_n=
        fwi_table_xarray.coords[canadian_fwi_utils.LATITUDE_DIM].values[-1]
    )

    desired_column_indices = misc_utils.desired_longitudes_to_columns(
        grid_longitudes_deg_e=
        daily_gfs_table_xarray.coords[gfs_daily_io.LONGITUDE_DIM].values,
        start_longitude_deg_e=
        fwi_table_xarray.coords[canadian_fwi_utils.LONGITUDE_DIM].values[0],
        end_longitude_deg_e=
        fwi_table_xarray.coords[canadian_fwi_utils.LONGITUDE_DIM].values[-1]
    )

    daily_gfs_table_xarray = daily_gfs_table_xarray.isel({
        gfs_daily_io.LATITUDE_DIM: desired_row_indices
    })
    daily_gfs_table_xarray = daily_gfs_table_xarray.isel({
        gfs_daily_io.LONGITUDE_DIM: desired_column_indices
    })

    # TODO(thunderhoser): I am assuming here that the domain crosses the
    # International Date Line.
    longitudes_deg_e = longitude_conv.convert_lng_positive_in_west(
        daily_gfs_table_xarray.coords[gfs_daily_io.LONGITUDE_DIM].values
    )
    latitudes_deg_n = (
        daily_gfs_table_xarray.coords[gfs_daily_io.LATITUDE_DIM].values
    )
    latitude_matrix_deg_n = grids.latlng_vectors_to_matrices(
        unique_latitudes_deg=latitudes_deg_n,
        unique_longitudes_deg=longitudes_deg_e
    )[0]

    # Dimensions are currently lead_time x latitude x longitude x field.
    data_matrix_channels_last = (
        daily_gfs_table_xarray[gfs_daily_io.DATA_KEY_2D].values
    )
    field_names = (
        daily_gfs_table_xarray.coords[gfs_daily_io.FIELD_DIM].values.tolist()
    )

    temp_index = field_names.index(gfs_daily_utils.TEMPERATURE_2METRE_NAME)
    temperature_matrix_deg_c = temperature_conv.kelvins_to_celsius(
        data_matrix_channels_last[..., temp_index]
    )

    rh_index = field_names.index(gfs_daily_utils.RELATIVE_HUMIDITY_2METRE_NAME)
    rh_matrix_percent = (
        UNITLESS_TO_PERCENT * data_matrix_channels_last[..., rh_index]
    )

    u_index = field_names.index(gfs_daily_utils.U_WIND_10METRE_NAME)
    v_index = field_names.index(gfs_daily_utils.V_WIND_10METRE_NAME)
    wind_speed_matrix_km_h01 = METRES_PER_SECOND_TO_KMH * numpy.sqrt(
        data_matrix_channels_last[..., u_index] ** 2 +
        data_matrix_channels_last[..., v_index] ** 2
    )

    precip_index = field_names.index(gfs_daily_utils.PRECIP_NAME)
    precip_matrix_mm = (
        METRES_TO_MM * data_matrix_channels_last[..., precip_index]
    )

    latitude_matrix_deg_n = numpy.repeat(
        numpy.expand_dims(latitude_matrix_deg_n, axis=0),
        axis=0, repeats=data_matrix_channels_last.shape[0]
    )

    # Dimensions are now field x lead_time x latitude x longitude.
    data_matrix_channels_first = numpy.stack((
        latitude_matrix_deg_n, temperature_matrix_deg_c, rh_matrix_percent,
        wind_speed_matrix_km_h01, precip_matrix_mm
    ), axis=0)

    fwi_matrix_channels_first = numpy.stack([
        canadian_fwi_utils.get_field(
            fwi_table_xarray=fwi_table_xarray, field_name=f
        ) for f in FWI_FIELD_NAMES
    ], axis=0)

    day1_index = numpy.where(
        daily_gfs_table_xarray.coords[gfs_daily_io.LEAD_TIME_DIM].values == 1
    )[0][0]

    data_matrix_day0 = numpy.concatenate((
        data_matrix_channels_first[:, day1_index, ...],
        fwi_matrix_channels_first
    ), axis=0)

    num_fields = data_matrix_day0.shape[0]
    coord_dict = {
        'band': numpy.linspace(1, num_fields, num=num_fields, dtype=int),
        'latitude_deg_n': latitudes_deg_n,
        'longitude_deg_e': longitudes_deg_e
    }

    gfs_table_xarray_day0 = xarray.DataArray(
        data=data_matrix_day0, coords=coord_dict
    )
    gfs_table_xarray_day0.rio.set_spatial_dims(
        x_dim='longitude_deg_e', y_dim='latitude_deg_n'
    )

    output_file_name = _find_output_file(
        directory_name=output_dir_name,
        init_date_string=init_date_string,
        lead_time_days=0,
        raise_error_if_missing=False
    )

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_file_name
    )
    gfs_table_xarray_day0.rio.to_raster(output_file_name)

    num_fields = data_matrix_channels_first.shape[0]
    coord_dict = {
        'band': numpy.linspace(1, num_fields, num=num_fields, dtype=int),
        'latitude_deg_n': latitudes_deg_n,
        'longitude_deg_e': longitudes_deg_e
    }

    for this_lead_time_days in range(max_lead_time_days + 1):
        if this_lead_time_days == 0:
            continue

        k = numpy.where(
            daily_gfs_table_xarray.coords[gfs_daily_io.LEAD_TIME_DIM].values
            == this_lead_time_days
        )[0][0]

        gfs_table_xarray_day_k = xarray.DataArray(
            data=data_matrix_channels_first[:, k, ...],
            coords=coord_dict
        )
        gfs_table_xarray_day_k.rio.set_spatial_dims(
            x_dim='longitude_deg_e', y_dim='latitude_deg_n'
        )

        output_file_name = _find_output_file(
            directory_name=output_dir_name,
            init_date_string=init_date_string,
            lead_time_days=this_lead_time_days,
            raise_error_if_missing=False
        )

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        file_system_utils.mkdir_recursive_if_necessary(
            file_name=output_file_name
        )
        gfs_table_xarray_day_k.rio.to_raster(output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        daily_gfs_dir_name=getattr(INPUT_ARG_OBJECT, DAILY_GFS_DIR_ARG_NAME),
        canadian_fwi_dir_name=getattr(
            INPUT_ARG_OBJECT, CANADIAN_FWI_DIR_ARG_NAME
        ),
        init_date_string=getattr(INPUT_ARG_OBJECT, INIT_DATE_ARG_NAME),
        max_lead_time_days=getattr(INPUT_ARG_OBJECT, MAX_LEAD_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
