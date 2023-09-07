"""Converts ERA5 data to TIF format.

TIF format is required by the fwiRaster script in R.
"""

import os
import argparse
import numpy
import xarray
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import temperature_conversions as temperature_conv
from gewittergefahr.gg_utils import longitude_conversion as longitude_conv
from gewittergefahr.gg_utils import file_system_utils
from ml_for_wildfire_wpo.io import era5_io
from ml_for_wildfire_wpo.utils import era5_utils

UNITLESS_TO_PERCENT = 100.
METRES_TO_MM = 1000.
METRES_PER_SECOND_TO_KMH = 3.6

INPUT_DIR_ARG_NAME = 'input_netcdf_dir_name'
START_DATE_ARG_NAME = 'start_date_string'
END_DATE_ARG_NAME = 'end_date_string'
OUTPUT_DIR_ARG_NAME = 'output_tif_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing one NetCDF file per day.  Files '
    'therein will be found by `era5_io.find_file`.'
)
START_DATE_HELP_STRING = (
    'Start date (format "yyyymmdd").  This script will process all dates in '
    'the continuous period {0:s}...{1:s}.'
).format(START_DATE_ARG_NAME, END_DATE_ARG_NAME)

END_DATE_HELP_STRING = 'Same as {0:s} but end date.'.format(START_DATE_ARG_NAME)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  TIF files will be written here (one per day) '
    'by rioxarray, to exact locations determined by `era5_io.find_file` '
    '(except with a different extension).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + START_DATE_ARG_NAME, type=str, required=True,
    help=START_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + END_DATE_ARG_NAME, type=str, required=True,
    help=END_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, start_date_string, end_date_string, output_dir_name):
    """Converts ERA5 data to TIF format.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param start_date_string: Same.
    :param end_date_string: Same.
    :param output_dir_name: Same.
    """

    valid_date_strings = time_conversion.get_spc_dates_in_range(
        start_date_string, end_date_string
    )

    for this_date_string in valid_date_strings:
        input_file_name = era5_io.find_file(
            directory_name=input_dir_name, valid_date_string=this_date_string,
            raise_error_if_missing=True
        )

        print('Reading data from: "{0:s}"...'.format(input_file_name))
        orig_era5_table_xarray = era5_io.read_file(input_file_name)

        # TODO(thunderhoser): I am assuming here that the domain crosses the
        # International Date Line.
        longitudes_deg_e = longitude_conv.convert_lng_positive_in_west(
            orig_era5_table_xarray.coords[era5_io.LONGITUDE_DIM].values
        )
        latitudes_deg_n = (
            orig_era5_table_xarray.coords[era5_io.LATITUDE_DIM].values
        )
        latitude_matrix_deg_n = grids.latlng_vectors_to_matrices(
            unique_latitudes_deg=latitudes_deg_n,
            unique_longitudes_deg=longitudes_deg_e
        )[0]

        data_matrix = orig_era5_table_xarray[era5_io.DATA_KEY].values
        field_names = (
            orig_era5_table_xarray.coords[era5_io.FIELD_DIM].values.tolist()
        )

        temp_index = field_names.index(era5_utils.TEMPERATURE_2METRE_NAME)
        temperature_matrix_deg_c = temperature_conv.kelvins_to_celsius(
            data_matrix[..., temp_index]
        )

        rh_index = field_names.index(era5_utils.RELATIVE_HUMIDITY_2METRE_NAME)
        rh_matrix_percent = UNITLESS_TO_PERCENT * data_matrix[..., rh_index]

        u_index = field_names.index(era5_utils.U_WIND_10METRE_NAME)
        v_index = field_names.index(era5_utils.V_WIND_10METRE_NAME)
        wind_speed_matrix_km_h01 = METRES_PER_SECOND_TO_KMH * numpy.sqrt(
            data_matrix[..., u_index] ** 2 + data_matrix[..., v_index] ** 2
        )

        precip_index = field_names.index(era5_utils.HOURLY_PRECIP_NAME)
        precip_matrix_mm = METRES_TO_MM * data_matrix[..., precip_index]

        data_matrix = numpy.stack((
            latitude_matrix_deg_n, temperature_matrix_deg_c, rh_matrix_percent,
            wind_speed_matrix_km_h01, precip_matrix_mm
        ), axis=0)

        num_fields = data_matrix.shape[0]
        coord_dict = {
            'band': numpy.linspace(1, num_fields, num=num_fields, dtype=int),
            'latitude_deg_n': latitudes_deg_n,
            'longitude_deg_e': longitudes_deg_e
        }

        new_era5_table_xarray = xarray.DataArray(
            data=data_matrix, coords=coord_dict
        )
        new_era5_table_xarray.rio.set_spatial_dims(
            x_dim='longitude_deg_e', y_dim='latitude_deg_n'
        )

        output_file_name = era5_io.find_file(
            directory_name=output_dir_name, valid_date_string=this_date_string,
            raise_error_if_missing=False
        )
        output_file_name = '{0:s}.tif'.format(
            os.path.splitext(output_file_name)[0]
        )

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        file_system_utils.mkdir_recursive_if_necessary(
            file_name=output_file_name
        )
        new_era5_table_xarray.rio.to_raster(output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        start_date_string=getattr(INPUT_ARG_OBJECT, START_DATE_ARG_NAME),
        end_date_string=getattr(INPUT_ARG_OBJECT, END_DATE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
