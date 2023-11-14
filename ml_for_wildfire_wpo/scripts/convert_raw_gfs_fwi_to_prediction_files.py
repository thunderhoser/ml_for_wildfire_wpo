"""Converts raw-GFS FWI forecasts to prediction files.

The inputs are daily GFS files, readable by `gfs_daily_io.read_file`.
The outputs are daily prediction files, readable by `prediction_io.read_file`.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml_for_wildfire_wpo.io import gfs_daily_io
from ml_for_wildfire_wpo.io import canadian_fwi_io
from ml_for_wildfire_wpo.io import era5_constant_io
from ml_for_wildfire_wpo.io import prediction_io
from ml_for_wildfire_wpo.utils import misc_utils
from ml_for_wildfire_wpo.utils import gfs_daily_utils
from ml_for_wildfire_wpo.utils import canadian_fwi_utils
from ml_for_wildfire_wpo.utils import era5_constant_utils
from ml_for_wildfire_wpo.machine_learning import neural_net

DATE_FORMAT = gfs_daily_io.DATE_FORMAT

DAYS_TO_SECONDS = 86400
DEGREES_TO_RADIANS = numpy.pi / 180.

TARGET_FIELD_NAMES = canadian_fwi_utils.ALL_FIELD_NAMES

DAILY_GFS_DIR_ARG_NAME = 'input_daily_gfs_dir_name'
CANADIAN_FWI_DIR_ARG_NAME = 'input_canadian_fwi_dir_name'
ERA5_CONSTANT_FILE_ARG_NAME = 'input_era5_constant_file_name'
TARGET_NORM_FILE_ARG_NAME = 'input_target_norm_file_name'
INIT_DATE_ARG_NAME = 'init_date_string'
LEAD_TIME_ARG_NAME = 'lead_time_days'
LATITUDE_LIMITS_ARG_NAME = 'latitude_limits_deg_n'
LONGITUDE_LIMITS_ARG_NAME = 'longitude_limits_deg_e'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

DAILY_GFS_DIR_HELP_STRING = (
    'Name of directory with daily GFS files, to be converted to prediction '
    'files.  Files in this directory will be located by '
    '`gfs_daily_io.find_file` and read by `gfs_daily_io.read_file`.'
)
CANADIAN_FWI_DIR_HELP_STRING = (
    'Name of directory with Canadian FWI data.  Target (actual) values will be '
    'read from this directory by `canadian_fwi_io.read_file`; the file needed '
    'will be found by `canadian_fwi_io.find_file`.'
)
ERA5_CONSTANT_FILE_HELP_STRING = (
    'Path to file with ERA5 constant fields.  This file will be read by '
    '`era5_constant_io.read_file` and used to create evaluation weights '
    '(cosine of latitude over land, 0 over ocean).'
)
TARGET_NORM_FILE_HELP_STRING = (
    'Path to file with normalization parameters for all target fields (FWIs).  '
    'This file will be read by `canadian_fwi_io.read_normalization_file` and '
    'used to determine the climatological target values, which are used to '
    'compute skill scores.'
)
INIT_DATE_HELP_STRING = (
    'Initialization date (format "yyyymmdd") for GFS model run.  Will assume '
    '0000 UTC on this date.'
)
LEAD_TIME_HELP_STRING = (
    'Lead time for FWI forecast.  Predictions for only this lead will be '
    'written to the output files.'
)
LATITUDE_LIMITS_HELP_STRING = (
    'Length-2 list with meridional limits (deg north) of bounding box for '
    'prediction domain.'
)
LONGITUDE_LIMITS_HELP_STRING = (
    'Same as {0:s} but for longitudes (deg east).'
).format(LATITUDE_LIMITS_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Raw-GFS-forecasts, for the given variable at '
    'the given lead time, will be written here by `prediction_io.write_file`, '
    'to exact locations determined by `prediction_io.find_file`.'
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
    '--' + ERA5_CONSTANT_FILE_ARG_NAME, type=str, required=True,
    help=ERA5_CONSTANT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_NORM_FILE_ARG_NAME, type=str, required=True,
    help=TARGET_NORM_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_DATE_ARG_NAME, type=str, required=True,
    help=INIT_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIME_ARG_NAME, type=int, required=True,
    help=LEAD_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LATITUDE_LIMITS_ARG_NAME, type=float, nargs=2,
    required=True, help=LATITUDE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LONGITUDE_LIMITS_ARG_NAME, type=float, nargs=2,
    required=True, help=LONGITUDE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(daily_gfs_dir_name, canadian_fwi_dir_name,
         era5_constant_file_name, target_norm_file_name,
         init_date_string, lead_time_days, latitude_limits_deg_n,
         longitude_limits_deg_e, output_dir_name):
    """Converts raw-GFS FWI forecasts to prediction files.

    This is effectively the main method.

    :param daily_gfs_dir_name: See documentation at top of file.
    :param canadian_fwi_dir_name: Same.
    :param era5_constant_file_name: Same.
    :param target_norm_file_name: Same.
    :param init_date_string: Same.
    :param lead_time_days: Same.
    :param latitude_limits_deg_n: Same.
    :param longitude_limits_deg_e: Same.
    :param output_dir_name: Same.
    """

    # Create prediction matrix.
    daily_gfs_file_name = gfs_daily_io.find_file(
        directory_name=daily_gfs_dir_name,
        init_date_string=init_date_string,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(daily_gfs_file_name))
    daily_gfs_table_xarray = gfs_daily_io.read_file(daily_gfs_file_name)
    dgfst = daily_gfs_table_xarray

    dgfst = gfs_daily_utils.subset_by_lead_time(
        daily_gfs_table_xarray=dgfst,
        lead_times_days=numpy.array([lead_time_days], dtype=int)
    )
    daily_gfs_row_indices = misc_utils.desired_latitudes_to_rows(
        grid_latitudes_deg_n=dgfst.coords[gfs_daily_utils.LATITUDE_DIM].values,
        start_latitude_deg_n=latitude_limits_deg_n[0],
        end_latitude_deg_n=latitude_limits_deg_n[1]
    )
    daily_gfs_column_indices = misc_utils.desired_longitudes_to_columns(
        grid_longitudes_deg_e=
        dgfst.coords[gfs_daily_utils.LONGITUDE_DIM].values,
        start_longitude_deg_e=longitude_limits_deg_e[0],
        end_longitude_deg_e=longitude_limits_deg_e[1]
    )
    dgfst = gfs_daily_utils.subset_by_row(
        daily_gfs_table_xarray=dgfst,
        desired_row_indices=daily_gfs_row_indices
    )
    dgfst = gfs_daily_utils.subset_by_column(
        daily_gfs_table_xarray=dgfst,
        desired_column_indices=daily_gfs_column_indices
    )

    daily_gfs_table_xarray = dgfst
    prediction_matrix = numpy.stack([
        gfs_daily_utils.get_field(
            daily_gfs_table_xarray=daily_gfs_table_xarray, field_name=f
        ) for f in TARGET_FIELD_NAMES
    ], axis=-1)

    # Create target matrix.
    init_date_unix_sec = time_conversion.string_to_unix_sec(
        init_date_string, DATE_FORMAT
    )
    valid_date_unix_sec = init_date_unix_sec + lead_time_days * DAYS_TO_SECONDS
    valid_date_string = time_conversion.unix_sec_to_string(
        valid_date_unix_sec, DATE_FORMAT
    )

    canadian_fwi_file_name = canadian_fwi_io.find_file(
        directory_name=canadian_fwi_dir_name,
        valid_date_string=valid_date_string,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(canadian_fwi_file_name))
    canadian_fwi_table_xarray = canadian_fwi_io.read_file(
        canadian_fwi_file_name
    )
    cfwit = canadian_fwi_table_xarray

    canadian_fwi_row_indices = misc_utils.desired_latitudes_to_rows(
        grid_latitudes_deg_n=
        cfwit.coords[canadian_fwi_utils.LATITUDE_DIM].values,
        start_latitude_deg_n=latitude_limits_deg_n[0],
        end_latitude_deg_n=latitude_limits_deg_n[1]
    )
    canadian_fwi_column_indices = misc_utils.desired_longitudes_to_columns(
        grid_longitudes_deg_e=
        cfwit.coords[canadian_fwi_utils.LONGITUDE_DIM].values,
        start_longitude_deg_e=longitude_limits_deg_e[0],
        end_longitude_deg_e=longitude_limits_deg_e[1]
    )
    cfwit = canadian_fwi_utils.subset_by_row(
        fwi_table_xarray=cfwit,
        desired_row_indices=canadian_fwi_row_indices
    )
    cfwit = canadian_fwi_utils.subset_by_column(
        fwi_table_xarray=cfwit,
        desired_column_indices=canadian_fwi_column_indices
    )

    canadian_fwi_table_xarray = cfwit
    target_matrix = numpy.stack([
        canadian_fwi_utils.get_field(
            fwi_table_xarray=canadian_fwi_table_xarray, field_name=f
        ) for f in TARGET_FIELD_NAMES
    ], axis=-1)

    # Create weight matrix.
    print('Reading data from: "{0:s}"...'.format(era5_constant_file_name))
    era5_constant_table_xarray = era5_constant_io.read_file(
        era5_constant_file_name
    )
    era5ct = era5_constant_table_xarray

    era5_row_indices = misc_utils.desired_latitudes_to_rows(
        grid_latitudes_deg_n=
        era5ct.coords[era5_constant_utils.LATITUDE_DIM].values,
        start_latitude_deg_n=latitude_limits_deg_n[0],
        end_latitude_deg_n=latitude_limits_deg_n[1]
    )
    era5_column_indices = misc_utils.desired_longitudes_to_columns(
        grid_longitudes_deg_e=
        era5ct.coords[era5_constant_utils.LONGITUDE_DIM].values,
        start_longitude_deg_e=longitude_limits_deg_e[0],
        end_longitude_deg_e=longitude_limits_deg_e[1]
    )
    era5ct = era5_constant_utils.subset_by_row(
        era5_constant_table_xarray=era5ct,
        desired_row_indices=era5_row_indices
    )
    era5ct = era5_constant_utils.subset_by_column(
        era5_constant_table_xarray=era5ct,
        desired_column_indices=era5_column_indices
    )

    era5_constant_table_xarray = era5ct
    land_mask_matrix = era5_constant_utils.get_field(
        era5_constant_table_xarray=era5_constant_table_xarray,
        field_name=era5_constant_utils.LAND_SEA_MASK_NAME
    )

    era5ct = era5_constant_table_xarray
    grid_latitudes_deg_n = (
        era5ct.coords[era5_constant_utils.LATITUDE_DIM].values
    )
    latitude_cosines = numpy.cos(DEGREES_TO_RADIANS * grid_latitudes_deg_n)
    latitude_cosine_matrix = numpy.repeat(
        numpy.expand_dims(latitude_cosines, axis=1),
        axis=1,
        repeats=len(era5ct.coords[era5_constant_utils.LONGITUDE_DIM].values)
    )

    weight_matrix = latitude_cosine_matrix * (land_mask_matrix > 0.05)

    # Create fake neural-net metafile.
    generator_option_dict = {
        neural_net.TARGET_FIELDS_KEY: TARGET_FIELD_NAMES,
        neural_net.TARGET_NORM_FILE_KEY: target_norm_file_name
    }

    fake_model_file_name = '{0:s}/fake_model_{1:s}/model.h5'.format(
        output_dir_name, init_date_string
    )
    fake_model_metafile_name = neural_net.find_metafile(
        model_file_name=fake_model_file_name,
        raise_error_if_missing=False
    )

    print('Writing fake neural-net metafile to: "{0:s}"...'.format(
        fake_model_metafile_name
    ))
    neural_net.write_metafile(
        pickle_file_name=fake_model_metafile_name,
        num_epochs=1000,
        num_training_batches_per_epoch=32,
        training_option_dict=generator_option_dict,
        num_validation_batches_per_epoch=32,
        validation_option_dict=generator_option_dict,
        loss_function_string='keras.losses.mse',
        plateau_patience_epochs=10,
        plateau_learning_rate_multiplier=0.6,
        early_stopping_patience_epochs=50
    )

    # Write prediction file.
    prediction_file_name = prediction_io.find_file(
        directory_name=output_dir_name,
        init_date_string=init_date_string,
        raise_error_if_missing=False
    )

    print('Writing prediction file: "{0:s}"...'.format(prediction_file_name))
    prediction_io.write_file(
        netcdf_file_name=prediction_file_name,
        target_matrix_with_weights=numpy.concatenate(
            (target_matrix, numpy.expand_dims(weight_matrix, axis=-1)),
            axis=-1
        ),
        prediction_matrix=prediction_matrix,
        grid_latitudes_deg_n=
        era5ct.coords[era5_constant_utils.LATITUDE_DIM].values,
        grid_longitudes_deg_e=
        era5ct.coords[era5_constant_utils.LONGITUDE_DIM].values,
        field_names=TARGET_FIELD_NAMES,
        init_date_string=init_date_string,
        model_file_name=fake_model_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        daily_gfs_dir_name=getattr(INPUT_ARG_OBJECT, DAILY_GFS_DIR_ARG_NAME),
        canadian_fwi_dir_name=getattr(
            INPUT_ARG_OBJECT, CANADIAN_FWI_DIR_ARG_NAME
        ),
        era5_constant_file_name=getattr(
            INPUT_ARG_OBJECT, ERA5_CONSTANT_FILE_ARG_NAME
        ),
        target_norm_file_name=getattr(
            INPUT_ARG_OBJECT, TARGET_NORM_FILE_ARG_NAME
        ),
        init_date_string=getattr(INPUT_ARG_OBJECT, INIT_DATE_ARG_NAME),
        lead_time_days=getattr(INPUT_ARG_OBJECT, LEAD_TIME_ARG_NAME),
        latitude_limits_deg_n=numpy.array(
            getattr(INPUT_ARG_OBJECT, LATITUDE_LIMITS_ARG_NAME), dtype=float
        ),
        longitude_limits_deg_e=numpy.array(
            getattr(INPUT_ARG_OBJECT, LONGITUDE_LIMITS_ARG_NAME), dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
