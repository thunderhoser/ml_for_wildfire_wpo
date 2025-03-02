"""Input/output methods for Shapley maps."""

import os
import numpy
import xarray
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.utils import canadian_fwi_utils
from ml_for_wildfire_wpo.machine_learning import neural_net

DATE_FORMAT = '%Y%m%d'

ROW_DIM = 'grid_row'
COLUMN_DIM = 'grid_column'
PRESSURE_LEVEL_DIM = 'pressure_level'
GFS_LEAD_TIME_DIM = 'gfs_lead_time'
GFS_3D_FIELD_DIM = 'gfs_3d_field'
GFS_2D_FIELD_DIM = 'gfs_2d_field'
ERA5_CONSTANT_FIELD_DIM = 'era5_constant_field'
LAGLEAD_TIME_DIM = 'laglead_time'
TARGET_FIELD_DIM = 'target_field_dim'

INIT_DATE_KEY = 'init_date_string'
BASELINE_INIT_DATES_KEY = 'baseline_init_date_strings'
REGION_MASK_FILE_KEY = 'region_mask_file_name'
TARGET_FIELD_KEY = 'target_field_name'
MODEL_LEAD_TIME_KEY = 'model_lead_time_days'
MODEL_FILE_KEY = 'model_file_name'

LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'
GFS_LEAD_TIME_KEY = 'gfs_lead_time_hours'
TARGET_LAGLEAD_TIME_KEY = 'target_laglead_time_hours'
SHAPLEY_FOR_3D_GFS_KEY = 'shapley_gfs_3d_inputs'
SHAPLEY_FOR_2D_GFS_KEY = 'shapley_gfs_2d_inputs'
SHAPLEY_FOR_ERA5_KEY = 'shapley_for_era5_inputs'
SHAPLEY_FOR_LAGLEAD_TARGETS_KEY = 'shapley_for_lagged_target_inputs'
SHAPLEY_FOR_PREDN_BASELINE_KEY = 'shapley_for_predn_baseline_inputs'

COMPOSITE_INIT_DATES_KEY = 'init_date_strings'
PREDICTOR_3D_GFS_KEY = 'predictor_gfs_3d'
PREDICTOR_2D_GFS_KEY = 'predictor_gfs_2d'
PREDICTOR_ERA5_KEY = 'predictor_era5'
PREDICTOR_LAGLEAD_TARGETS_KEY = 'predictor_lagged_target'
PREDICTOR_BASELINE_KEY = 'predictor_baseline'
TARGET_VALUE_KEY = 'target_value'


def find_file(directory_name, init_date_string, raise_error_if_missing=True):
    """Finds NetCDF file with Shapley maps for one forecast-init day (at 00Z).

    :param directory_name: Path to input directory.
    :param init_date_string: Initialization date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: shapley_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    _ = time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)
    error_checking.assert_is_boolean(raise_error_if_missing)

    shapley_file_name = '{0:s}/shapley_maps_{1:s}.nc'.format(
        directory_name, init_date_string
    )

    if raise_error_if_missing and not os.path.isfile(shapley_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            shapley_file_name
        )
        raise ValueError(error_string)

    return shapley_file_name


def write_file(
        netcdf_file_name, shapley_matrices,
        grid_latitudes_deg_n, grid_longitudes_deg_e,
        gfs_pred_lead_times_hours, target_laglead_times_hours,
        region_mask_file_name, target_field_name,
        model_input_layer_names, model_lead_time_days, model_file_name,
        init_date_string=None, baseline_init_date_strings=None,
        composite_init_date_strings=None, predictor_matrices=None,
        target_matrix=None):
    """Writes Shapley maps to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    If you're writing data for a single day (just a normal Shapley map), use the
    arguments "init_date_string" and "baseline_init_date_strings".

    If you're writing a composite Shapley map (averaged over many days), use the
    arguments "composite_init_date_strings", "predictor_matrices", and
    "target_matrix".

    :param netcdf_file_name: Path to output file.
    :param shapley_matrices: 1-D list of numpy arrays with Shapley values.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param gfs_pred_lead_times_hours: 1-D numpy array of lead times for
        GFS-based predictors.
    :param target_laglead_times_hours: 1-D numpy array of lag/lead times for
        target (FWI) fields used in predictors.
    :param region_mask_file_name: Path to file with region mask (readable by
        `region_mask_io.read_file`).  Shapley values should pertain only to the
        given target field averaged over this spatial region.
    :param target_field_name: Name of target field to which Shapley values
        pertain.
    :param model_input_layer_names: 1-D list with names of input layers to
        model.
    :param model_lead_time_days: Model lead time.
    :param model_file_name: Path to file with trained model.
    :param init_date_string: Forecast-init day (format "yyyymmdd").
    :param baseline_init_date_strings: 1-D list of forecast-init days used in
        DeepSHAP baseline (format "yyyymmdd").
    :param composite_init_date_strings: 1-D list of forecast-init days averaged
        into composite (format "yyyymmdd").
    :param predictor_matrices: 1-D list of numpy arrays with composite predictor
        values.  predictor_matrices must have the same length as
        shapley_matrices, and predictor_matirces[k] must have the same shape as
        shapley_matrices[k].
    :param target_matrix: numpy array with composite target values.
    """

    # Check input args.
    error_checking.assert_is_string_list(model_input_layer_names)
    for this_layer_name in model_input_layer_names:
        assert this_layer_name in neural_net.VALID_INPUT_LAYER_NAMES

    assert (
        neural_net.GFS_3D_LAYER_NAME in model_input_layer_names or
        neural_net.GFS_2D_LAYER_NAME in model_input_layer_names
    )
    assert len(model_input_layer_names) == len(shapley_matrices)

    if neural_net.GFS_3D_LAYER_NAME in model_input_layer_names:
        lyr_idx = model_input_layer_names.index(neural_net.GFS_3D_LAYER_NAME)
    else:
        lyr_idx = model_input_layer_names.index(neural_net.GFS_2D_LAYER_NAME)

    num_grid_rows = shapley_matrices[lyr_idx].shape[0]
    num_grid_columns = shapley_matrices[lyr_idx].shape[1]
    num_gfs_lead_times = shapley_matrices[lyr_idx].shape[-2]

    error_checking.assert_is_numpy_array(
        gfs_pred_lead_times_hours,
        exact_dimensions=numpy.array([num_gfs_lead_times], dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(gfs_pred_lead_times_hours, 0.)

    error_checking.assert_is_numpy_array(
        grid_latitudes_deg_n,
        exact_dimensions=numpy.array([num_grid_rows], dtype=int)
    )
    error_checking.assert_is_valid_lat_numpy_array(
        grid_latitudes_deg_n, allow_nan=False
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(grid_latitudes_deg_n), 0.
    )

    error_checking.assert_is_numpy_array(
        grid_longitudes_deg_e,
        exact_dimensions=numpy.array([num_grid_columns], dtype=int)
    )
    grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        grid_longitudes_deg_e, allow_nan=False
    )

    if not numpy.all(numpy.diff(grid_longitudes_deg_e) > 0):
        grid_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            grid_longitudes_deg_e
        )

    writing_composite = composite_init_date_strings is not None

    if writing_composite:
        init_date_string = None
        baseline_init_date_strings = None

        error_checking.assert_is_string_list(composite_init_date_strings)

        for this_date_string in composite_init_date_strings:
            _ = time_conversion.string_to_unix_sec(
                this_date_string, DATE_FORMAT
            )

        error_checking.assert_equals(
            len(predictor_matrices), len(shapley_matrices)
        )
        for k in range(len(predictor_matrices)):
            assert numpy.array_equal(
                predictor_matrices[k].shape, shapley_matrices[k].shape
            )
    else:
        _ = time_conversion.string_to_unix_sec(init_date_string, DATE_FORMAT)
        error_checking.assert_is_string_list(baseline_init_date_strings)

        for this_date_string in baseline_init_date_strings:
            _ = time_conversion.string_to_unix_sec(
                this_date_string, DATE_FORMAT
            )

    error_checking.assert_is_string(region_mask_file_name)
    canadian_fwi_utils.check_field_name(target_field_name)
    error_checking.assert_is_integer(model_lead_time_days)
    error_checking.assert_is_greater(model_lead_time_days, 1)
    error_checking.assert_is_string(model_file_name)

    # Do actual stuff.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF4'
    )

    dataset_object.setncattr(
        INIT_DATE_KEY,
        '' if init_date_string is None else init_date_string
    )
    dataset_object.setncattr(
        BASELINE_INIT_DATES_KEY,
        '' if baseline_init_date_strings is None
        else ' '.join(baseline_init_date_strings)
    )
    dataset_object.setncattr(
        COMPOSITE_INIT_DATES_KEY,
        '' if composite_init_date_strings is None
        else ' '.join(composite_init_date_strings)
    )
    dataset_object.setncattr(REGION_MASK_FILE_KEY, region_mask_file_name)
    dataset_object.setncattr(TARGET_FIELD_KEY, target_field_name)
    dataset_object.setncattr(MODEL_LEAD_TIME_KEY, model_lead_time_days)
    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)

    if writing_composite:
        dataset_object.createDimension(ROW_DIM, num_grid_rows)
        dataset_object.createDimension(COLUMN_DIM, num_grid_columns)
        dataset_object.createDimension(
            TARGET_FIELD_DIM, target_matrix.shape[2]
        )

        these_dim = (ROW_DIM, COLUMN_DIM, TARGET_FIELD_DIM)
        dataset_object.createVariable(
            TARGET_VALUE_KEY, datatype=numpy.float64, dimensions=these_dim
        )
        dataset_object.variables[TARGET_VALUE_KEY][:] = target_matrix

    num_input_layers = len(model_input_layer_names)

    for k in range(num_input_layers):
        if model_input_layer_names[k] == neural_net.LEAD_TIME_LAYER_NAME:
            continue

        if model_input_layer_names[k] == neural_net.GFS_3D_LAYER_NAME:
            if ROW_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(ROW_DIM, num_grid_rows)
            if COLUMN_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(COLUMN_DIM, num_grid_columns)
            if PRESSURE_LEVEL_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    PRESSURE_LEVEL_DIM, shapley_matrices[k].shape[2]
                )
            if GFS_LEAD_TIME_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    GFS_LEAD_TIME_DIM, shapley_matrices[k].shape[3]
                )
            if GFS_3D_FIELD_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    GFS_3D_FIELD_DIM, shapley_matrices[k].shape[4]
                )

            these_dim = (
                ROW_DIM, COLUMN_DIM, PRESSURE_LEVEL_DIM,
                GFS_LEAD_TIME_DIM, GFS_3D_FIELD_DIM
            )
            dataset_object.createVariable(
                SHAPLEY_FOR_3D_GFS_KEY,
                datatype=numpy.float64, dimensions=these_dim
            )
            dataset_object.variables[SHAPLEY_FOR_3D_GFS_KEY][:] = (
                shapley_matrices[k]
            )

            if writing_composite:
                dataset_object.createVariable(
                    PREDICTOR_3D_GFS_KEY,
                    datatype=numpy.float64, dimensions=these_dim
                )
                dataset_object.variables[PREDICTOR_3D_GFS_KEY][:] = (
                    predictor_matrices[k]
                )

        elif model_input_layer_names[k] == neural_net.GFS_2D_LAYER_NAME:
            if ROW_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(ROW_DIM, num_grid_rows)
            if COLUMN_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(COLUMN_DIM, num_grid_columns)
            if GFS_LEAD_TIME_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    GFS_LEAD_TIME_DIM, shapley_matrices[k].shape[2]
                )
            if GFS_2D_FIELD_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    GFS_2D_FIELD_DIM, shapley_matrices[k].shape[3]
                )

            these_dim = (
                ROW_DIM, COLUMN_DIM, GFS_LEAD_TIME_DIM, GFS_2D_FIELD_DIM
            )
            dataset_object.createVariable(
                SHAPLEY_FOR_2D_GFS_KEY,
                datatype=numpy.float64, dimensions=these_dim
            )
            dataset_object.variables[SHAPLEY_FOR_2D_GFS_KEY][:] = (
                shapley_matrices[k]
            )

            if writing_composite:
                dataset_object.createVariable(
                    PREDICTOR_2D_GFS_KEY,
                    datatype=numpy.float64, dimensions=these_dim
                )
                dataset_object.variables[PREDICTOR_2D_GFS_KEY][:] = (
                    predictor_matrices[k]
                )

        elif model_input_layer_names[k] == neural_net.LAGLEAD_TARGET_LAYER_NAME:
            num_target_laglead_times = shapley_matrices[k].shape[2]
            error_checking.assert_is_numpy_array(
                target_laglead_times_hours,
                exact_dimensions=
                numpy.array([num_target_laglead_times], dtype=int)
            )

            if ROW_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(ROW_DIM, num_grid_rows)
            if COLUMN_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(COLUMN_DIM, num_grid_columns)
            if LAGLEAD_TIME_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    LAGLEAD_TIME_DIM, shapley_matrices[k].shape[2]
                )
            if TARGET_FIELD_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    TARGET_FIELD_DIM, shapley_matrices[k].shape[3]
                )

            these_dim = (
                ROW_DIM, COLUMN_DIM, LAGLEAD_TIME_DIM, TARGET_FIELD_DIM
            )
            dataset_object.createVariable(
                SHAPLEY_FOR_LAGLEAD_TARGETS_KEY,
                datatype=numpy.float64, dimensions=these_dim
            )
            dataset_object.variables[SHAPLEY_FOR_LAGLEAD_TARGETS_KEY][:] = (
                shapley_matrices[k]
            )

            if writing_composite:
                dataset_object.createVariable(
                    PREDICTOR_LAGLEAD_TARGETS_KEY,
                    datatype=numpy.float64, dimensions=these_dim
                )
                dataset_object.variables[PREDICTOR_LAGLEAD_TARGETS_KEY][:] = (
                    predictor_matrices[k]
                )

        elif model_input_layer_names[k] == neural_net.ERA5_LAYER_NAME:
            if ROW_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(ROW_DIM, num_grid_rows)
            if COLUMN_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(COLUMN_DIM, num_grid_columns)
            if ERA5_CONSTANT_FIELD_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    ERA5_CONSTANT_FIELD_DIM, shapley_matrices[k].shape[2]
                )

            these_dim = (ROW_DIM, COLUMN_DIM, ERA5_CONSTANT_FIELD_DIM)
            dataset_object.createVariable(
                SHAPLEY_FOR_ERA5_KEY,
                datatype=numpy.float64, dimensions=these_dim
            )
            dataset_object.variables[SHAPLEY_FOR_ERA5_KEY][:] = (
                shapley_matrices[k]
            )

            if writing_composite:
                dataset_object.createVariable(
                    PREDICTOR_ERA5_KEY,
                    datatype=numpy.float64, dimensions=these_dim
                )
                dataset_object.variables[PREDICTOR_ERA5_KEY][:] = (
                    predictor_matrices[k]
                )

        else:
            if ROW_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(ROW_DIM, num_grid_rows)
            if COLUMN_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(COLUMN_DIM, num_grid_columns)
            if TARGET_FIELD_DIM not in dataset_object.dimensions:
                dataset_object.createDimension(
                    TARGET_FIELD_DIM, shapley_matrices[k].shape[2]
                )

            these_dim = (ROW_DIM, COLUMN_DIM, TARGET_FIELD_DIM)
            dataset_object.createVariable(
                SHAPLEY_FOR_PREDN_BASELINE_KEY,
                datatype=numpy.float64, dimensions=these_dim
            )
            dataset_object.variables[SHAPLEY_FOR_PREDN_BASELINE_KEY][:] = (
                shapley_matrices[k]
            )

            if writing_composite:
                dataset_object.createVariable(
                    PREDICTOR_BASELINE_KEY,
                    datatype=numpy.float64, dimensions=these_dim
                )
                dataset_object.variables[PREDICTOR_BASELINE_KEY][:] = (
                    predictor_matrices[k]
                )

    dataset_object.createVariable(
        LATITUDE_KEY, datatype=numpy.float64, dimensions=ROW_DIM
    )
    dataset_object.variables[LATITUDE_KEY][:] = grid_latitudes_deg_n

    dataset_object.createVariable(
        LONGITUDE_KEY, datatype=numpy.float64, dimensions=COLUMN_DIM
    )
    dataset_object.variables[LONGITUDE_KEY][:] = grid_longitudes_deg_e

    dataset_object.createVariable(
        GFS_LEAD_TIME_KEY, datatype=numpy.float64, dimensions=GFS_LEAD_TIME_DIM
    )
    dataset_object.variables[GFS_LEAD_TIME_KEY][:] = gfs_pred_lead_times_hours

    if neural_net.LAGLEAD_TARGET_LAYER_NAME in model_input_layer_names:
        dataset_object.createVariable(
            TARGET_LAGLEAD_TIME_KEY, datatype=numpy.float64,
            dimensions=LAGLEAD_TIME_DIM
        )
        dataset_object.variables[TARGET_LAGLEAD_TIME_KEY][:] = (
            target_laglead_times_hours
        )

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads Shapley maps from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: shapley_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    shapley_table_xarray = xarray.open_dataset(netcdf_file_name)

    if shapley_table_xarray.attrs[INIT_DATE_KEY] == '':
        shapley_table_xarray.attrs[INIT_DATE_KEY] = None

    if shapley_table_xarray.attrs[BASELINE_INIT_DATES_KEY] == '':
        shapley_table_xarray.attrs[BASELINE_INIT_DATES_KEY] = None
    else:
        shapley_table_xarray.attrs[BASELINE_INIT_DATES_KEY] = (
            shapley_table_xarray.attrs[BASELINE_INIT_DATES_KEY].split(' ')
        )

    if shapley_table_xarray.attrs[COMPOSITE_INIT_DATES_KEY] == '':
        shapley_table_xarray.attrs[COMPOSITE_INIT_DATES_KEY] = None
    else:
        shapley_table_xarray.attrs[COMPOSITE_INIT_DATES_KEY] = (
            shapley_table_xarray.attrs[COMPOSITE_INIT_DATES_KEY].split(' ')
        )

    return shapley_table_xarray
