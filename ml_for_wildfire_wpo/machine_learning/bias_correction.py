"""Training and inference code for bias-correction models.

This module includes both isotonic regression, for bias-correcting the ensemble
mean, and uncertainty calibration, for bias-correcting the ensemble spread.
"""

import time
from multiprocessing import Pool
import dill
import numpy
import xarray
from sklearn.isotonic import IsotonicRegression
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.io import prediction_io
from ml_for_wildfire_wpo.utils import canadian_fwi_utils

TOLERANCE = 1e-6
MASK_PIXEL_IF_WEIGHT_BELOW = 0.01
MASK_PIXEL_IF_ORIG_WEIGHT_BELOW = 0.001
NUM_SLICES_FOR_MULTIPROCESSING = 24
MAX_STDEV_INFLATION_FACTOR = 1000.

MODELS_KEY = 'model_object_matrix'
LATITUDES_KEY = 'latitude_matrix_deg_e'
LONGITUDES_KEY = 'longitude_matrix_deg_e'
FIELD_NAMES_KEY = 'field_names'
PIXEL_RADIUS_KEY = 'pixel_radius_metres'
WEIGHT_BY_INV_DIST_KEY = 'weight_pixels_by_inverse_dist'
WEIGHT_BY_INV_SQ_DIST_KEY = 'weight_pixels_by_inverse_sq_dist'
DO_UNCERTAINTY_CALIB_KEY = 'do_uncertainty_calibration'
DO_IR_BEFORE_UC_KEY = 'do_iso_reg_before_uncertainty_calib'

ALL_KEYS = [
    MODELS_KEY, LATITUDES_KEY, LONGITUDES_KEY, FIELD_NAMES_KEY,
    PIXEL_RADIUS_KEY, WEIGHT_BY_INV_DIST_KEY, WEIGHT_BY_INV_SQ_DIST_KEY,
    DO_UNCERTAINTY_CALIB_KEY, DO_IR_BEFORE_UC_KEY
]


def __get_slices_for_multiprocessing(num_grid_rows):
    """Returns slices for multiprocessing.

    Each "slice" consists of several grid rows.

    K = number of slices

    :param num_grid_rows: Total number of grid rows.
    :return: start_rows: length-K numpy array with index of each start row.
    :return: end_rows: length-K numpy array with index of each end row.
    """

    slice_indices_normalized = numpy.linspace(
        0, 1, num=NUM_SLICES_FOR_MULTIPROCESSING + 1, dtype=float
    )

    start_rows = numpy.round(
        num_grid_rows * slice_indices_normalized[:-1]
    ).astype(int)

    end_rows = numpy.round(
        num_grid_rows * slice_indices_normalized[1:]
    ).astype(int)

    return start_rows, end_rows


def _subset_predictions_by_location(
        prediction_tables_xarray, desired_latitude_deg_n,
        desired_longitude_deg_e, pixel_radius_metres,
        weight_pixels_by_inverse_dist,
        weight_pixels_by_inverse_sq_dist):
    """Subsets predictions by location.

    :param prediction_tables_xarray: See input documentation for
        `train_model_suite`.
    :param desired_latitude_deg_n: Desired latitude (deg north).
    :param desired_longitude_deg_e: Desired longitude (deg east).
    :param pixel_radius_metres: See input documentation for
        `train_model_suite`.
    :param weight_pixels_by_inverse_dist: Same.
    :param weight_pixels_by_inverse_sq_dist: Same.
    :return: new_prediction_tables_xarray: Same as input, but maybe with a
        smaller grid and maybe with different evaluation weights.
    """

    num_tables = len(prediction_tables_xarray)
    new_prediction_tables_xarray = [xarray.Dataset()] * num_tables

    good_rows = numpy.array([], dtype=int)
    good_columns = numpy.array([], dtype=int)

    for k in range(num_tables):
        if k == 0:
            (
                new_prediction_tables_xarray[k], keep_location_matrix
            ) = prediction_io.subset_by_location(
                prediction_table_xarray=prediction_tables_xarray[k],
                desired_latitude_deg_n=desired_latitude_deg_n,
                desired_longitude_deg_e=desired_longitude_deg_e,
                radius_metres=pixel_radius_metres,
                recompute_weights_by_inverse_dist=weight_pixels_by_inverse_dist,
                recompute_weights_by_inverse_sq_dist=
                weight_pixels_by_inverse_sq_dist
            )

            good_rows, good_columns = numpy.where(keep_location_matrix)
            good_rows = numpy.unique(good_rows)
            good_columns = numpy.unique(good_columns)
            continue

        new_prediction_tables_xarray[k] = prediction_tables_xarray[k].isel(
            {prediction_io.ROW_DIM: good_rows}
        )
        new_prediction_tables_xarray[k] = new_prediction_tables_xarray[k].isel(
            {prediction_io.COLUMN_DIM: good_columns}
        )
        new_prediction_tables_xarray[k] = new_prediction_tables_xarray[k].assign({
            prediction_io.WEIGHT_KEY: (
                new_prediction_tables_xarray[k][prediction_io.WEIGHT_KEY].dims,
                new_prediction_tables_xarray[0][prediction_io.WEIGHT_KEY].values
            )
        })

    return new_prediction_tables_xarray


def _train_one_model(prediction_tables_xarray):
    """Trains one model.

    One model corresponds to either [1] one field or [2] one field at one grid
    point.

    :param prediction_tables_xarray: Same as input to `train_model_suite`,
        except containing only the relevant data (i.e., the relevant fields and
        grid points).
    :return: model_object: Trained instance of
        `sklearn.isotonic.IsotonicRegression`.
    """

    eval_weight_matrix = (
        prediction_tables_xarray[0][prediction_io.WEIGHT_KEY].values
    )
    good_spatial_inds = numpy.where(
        eval_weight_matrix >= MASK_PIXEL_IF_WEIGHT_BELOW
    )
    if len(good_spatial_inds[0]) == 0:
        print('Num training pixels/samples = 0/0')
        return None

    predicted_values = numpy.concatenate([
        ptx[prediction_io.PREDICTION_KEY].values[..., 0, 0][good_spatial_inds]
        for ptx in prediction_tables_xarray
    ])
    target_values = numpy.concatenate([
        ptx[prediction_io.TARGET_KEY].values[..., 0][good_spatial_inds]
        for ptx in prediction_tables_xarray
    ])
    eval_weights = numpy.concatenate([
        ptx[prediction_io.WEIGHT_KEY].values[good_spatial_inds]
        for ptx in prediction_tables_xarray
    ])

    percentile_levels = numpy.linspace(0, 100, num=11, dtype=float)
    print((
        'Num training pixels/samples = {0:d}/{1:d}; '
        'percentiles {2:s} of sample weights = {3:s}'
    ).format(
        len(good_spatial_inds[0]),
        len(predicted_values),
        str(percentile_levels),
        str(numpy.percentile(eval_weights, percentile_levels))
    ))

    model_object = IsotonicRegression(
        increasing=True, out_of_bounds='clip', y_min=0.
    )
    model_object.fit(
        X=predicted_values, y=target_values, sample_weight=eval_weights
    )
    return model_object


def _train_one_model_per_pixel(
        prediction_tables_xarray, train_for_grid_rows, pixel_radius_metres,
        weight_pixels_by_inverse_dist, weight_pixels_by_inverse_sq_dist):
    """Trains one model per pixel, using multiprocessing.

    m = number of grid rows for which to train a model
    N = number of columns in full grid
    F = number of target fields

    :param prediction_tables_xarray: See documentation for `train_model_suite`.
    :param train_for_grid_rows: length-m numpy array of row indices.  Models
        will be trained only for these rows.
    :param pixel_radius_metres: See documentation for `train_model_suite`.
    :param weight_pixels_by_inverse_dist: Same.
    :param weight_pixels_by_inverse_sq_dist: Same.
    :return: model_object_matrix: m-by-N-by-F numpy array of trained
        bias-correction models (instances of
        `sklearn.isotonic.IsotonicRegression`).
    """

    ptx = prediction_tables_xarray[0]
    grid_latitudes_deg_n = ptx[prediction_io.LATITUDE_KEY].values
    grid_longitudes_deg_e = ptx[prediction_io.LONGITUDE_KEY].values
    orig_eval_weight_matrix = ptx[prediction_io.WEIGHT_KEY].values
    field_names = ptx[prediction_io.FIELD_NAME_KEY].values.tolist()

    latitude_matrix_deg_n, longitude_matrix_deg_e = (
        grids.latlng_vectors_to_matrices(
            unique_latitudes_deg=grid_latitudes_deg_n,
            unique_longitudes_deg=grid_longitudes_deg_e
        )
    )

    num_target_rows = len(train_for_grid_rows)
    num_columns = latitude_matrix_deg_n.shape[1]
    num_fields = len(field_names)
    model_object_matrix = numpy.full(
        (num_target_rows, num_columns, num_fields),
        '', dtype=object
    )

    for f in range(num_fields):
        for i_model in range(num_target_rows):
            i_pred = train_for_grid_rows[i_model]

            for j in range(num_columns):
                if (
                        orig_eval_weight_matrix[i_pred, j] <
                        MASK_PIXEL_IF_ORIG_WEIGHT_BELOW
                ):
                    model_object_matrix[i_model, j, f] = None
                    continue

                print((
                    'Training bias-correction model for '
                    '{0:d}th of {1:d} grid rows, '
                    '{2:d}th of {3:d} columns, '
                    '{4:d}th of {5:d} fields...'
                ).format(
                    i_model + 1, num_target_rows,
                    j + 1, num_columns,
                    f + 1, num_fields
                ))

                these_prediction_tables_xarray = [
                    ptx.isel({
                        prediction_io.FIELD_DIM: numpy.array([f], dtype=int)
                    })
                    for ptx in prediction_tables_xarray
                ]

                these_prediction_tables_xarray = (
                    _subset_predictions_by_location(
                        prediction_tables_xarray=these_prediction_tables_xarray,
                        desired_latitude_deg_n=latitude_matrix_deg_n[i_pred, j],
                        desired_longitude_deg_e=
                        longitude_matrix_deg_e[i_pred, j],
                        pixel_radius_metres=pixel_radius_metres,
                        weight_pixels_by_inverse_dist=
                        weight_pixels_by_inverse_dist,
                        weight_pixels_by_inverse_sq_dist=
                        weight_pixels_by_inverse_sq_dist
                    )
                )

                model_object_matrix[i_model, j, f] = _train_one_model(
                    these_prediction_tables_xarray
                )

    return model_object_matrix


def _apply_one_model_per_pixel(prediction_table_xarray, model_dict,
                               apply_to_grid_rows, verbose):
    """Applies one model per pixel, using multiprocessing.

    m = number of grid rows for which to train a model
    N = number of columns in full grid
    F = number of target fields
    S = ensemble size

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`, containing uncorrected predictions.
    :param model_dict: Dictionary created by `train_model_suite`.
    :param apply_to_grid_rows: length-m numpy array of row indices.  Models
        will be applied only to these rows.
    :param verbose: Boolean flag.
    :return: prediction_matrix: m-by-N-by-F-by-S numpy array of bias-corrected
        predictions.
    """

    field_names = model_dict[FIELD_NAMES_KEY]
    model_object_matrix = model_dict[MODELS_KEY]
    do_uncertainty_calibration = model_dict[DO_UNCERTAINTY_CALIB_KEY]

    num_fields = len(field_names)
    num_target_rows = len(apply_to_grid_rows)
    num_columns = model_object_matrix.shape[1]

    ptx = prediction_table_xarray
    prediction_matrix = (
        ptx[prediction_io.PREDICTION_KEY].values[:, apply_to_grid_rows, ...]
    )

    if do_uncertainty_calibration:
        mean_prediction_matrix = numpy.mean(prediction_matrix, axis=-1)
        prediction_stdev_matrix = numpy.std(
            prediction_matrix, axis=-1, ddof=1
        )
    else:
        mean_prediction_matrix = numpy.array([], dtype=float)
        prediction_stdev_matrix = numpy.array([], dtype=float)

    constrain_dsr = (
        canadian_fwi_utils.FWI_NAME in ptx[prediction_io.FIELD_NAME_KEY].values
        and
        canadian_fwi_utils.DSR_NAME in ptx[prediction_io.FIELD_NAME_KEY].values
    )

    for f_model in range(num_fields):
        if constrain_dsr and field_names[f_model] == canadian_fwi_utils.DSR_NAME:
            continue

        f_pred = numpy.where(
            ptx[prediction_io.FIELD_NAME_KEY].values == field_names[f_model]
        )[0][0]
        prediction_matrix_this_field = (
            ptx[prediction_io.PREDICTION_KEY].values[..., f_pred, :]
        )

        for i_pred in range(num_target_rows):
            i_model = apply_to_grid_rows[i_pred]

            for j in range(num_columns):
                if model_object_matrix[i_model, j, f_model] is None:
                    continue

                if verbose:
                    print((
                        'Applying bias-correction model for '
                        '{0:d}th of {1:d} grid rows, '
                        '{2:d}th of {3:d} columns, '
                        '{4:d}th of {5:d} fields...'
                    ).format(
                        i_pred + 1, num_target_rows,
                        j + 1, num_columns,
                        f_model + 1, num_fields
                    ))

                if do_uncertainty_calibration:
                    orig_stdev = prediction_stdev_matrix[i_pred, j, f_pred]
                    new_stdev = numpy.sqrt(
                        model_object_matrix[i_model, j, f_model].predict(
                            orig_stdev ** 2
                        )
                    )
                    stdev_inflation_factor = new_stdev / orig_stdev

                    if numpy.isnan(stdev_inflation_factor):
                        stdev_inflation_factor = 1.

                    stdev_inflation_factor = numpy.minimum(
                        stdev_inflation_factor, MAX_STDEV_INFLATION_FACTOR
                    )

                    prediction_matrix[i_pred, j, f_pred, :] = (
                        mean_prediction_matrix[i_pred, j, f_pred] +
                        stdev_inflation_factor * (
                            prediction_matrix[i_pred, j, f_pred, :] -
                            mean_prediction_matrix[i_pred, j, f_pred]
                        )
                    )
                else:
                    prediction_matrix[i_pred, j, f_pred, :] = (
                        model_object_matrix[i_model, j, f_model].predict(
                            prediction_matrix_this_field[i_pred, j, :]
                        )
                    )

    if constrain_dsr:
        fwi_index = numpy.where(
            ptx[prediction_io.FIELD_NAME_KEY].values ==
            canadian_fwi_utils.FWI_NAME
        )[0][0]

        dsr_index = numpy.where(
            ptx[prediction_io.FIELD_NAME_KEY].values ==
            canadian_fwi_utils.DSR_NAME
        )[0][0]

        prediction_matrix[..., dsr_index, :] = canadian_fwi_utils.fwi_to_dsr(
            prediction_matrix[..., fwi_index, :]
        )

    return prediction_matrix


def train_model_suite(
        prediction_tables_xarray, one_model_per_pixel,
        do_uncertainty_calibration,
        pixel_radius_metres=None,
        weight_pixels_by_inverse_dist=None,
        weight_pixels_by_inverse_sq_dist=None,
        do_iso_reg_before_uncertainty_calib=None,
        do_multiprocessing=True):
    """Trains one model suite.

    The model suite will contain either [1] one model per target field or
    [2] one model per target field per pixel.

    M = number of rows in model grid.  If `one_model_per_pixel == True`, this
    will be the number of rows in the physical grid; else, this will be 1.

    N = same but for columns
    F = number of target fields

    :param prediction_tables_xarray: 1-D list of xarray tables in format
        returned by `prediction_io.read_file`.
    :param one_model_per_pixel: Boolean flag.  If True, will train one model per
        target field per pixel.  If False, will just train one model per target
        field.
    :param do_uncertainty_calibration: Boolean flag.  If True, this method will
        [1] assume that every "prediction" in `prediction_tables_xarray` is the
        ensemble variance; [2] assume that every "target" in
        `prediction_tables_xarray` is the squared error of the ensemble mean;
        [3] train IR models to adjust the ensemble variance, i.e., to do
        uncertainty calibration.  If False, this method will do standard
        isotonic regression, correcting only the ensemble mean.
    :param pixel_radius_metres: [used only if `one_model_per_pixel == True`]
        When training the model for pixel P, will use all pixels within this
        radius.
    :param weight_pixels_by_inverse_dist:
        [used only if `one_model_per_pixel == True`]
        Boolean flag.  If True, when training the model for pixel P, will weight
        every other pixel by the inverse of its distance to P.
    :param weight_pixels_by_inverse_sq_dist:
        [used only if `one_model_per_pixel == True`]
        Boolean flag.  If True, when training the model for pixel P, will weight
        every other pixel by the inverse of its *squared* distance to P.
    :param do_iso_reg_before_uncertainty_calib:
        [used only if `do_uncertainty_calibration == True`]
        Boolean flag, indicating whether isotonic regression has been done
        before uncertainty calibration.
    :param do_multiprocessing: [used only if `one_model_per_pixel == True`]
        Boolean flag.  If True, will do multi-threaded processing to make this
        go faster.

    :return: model_dict: Dictionary with the following keys.
    model_dict["model_object_matrix"]: M-by-N-by-F numpy array of trained
        bias-correction models (instances of
        `sklearn.isotonic.IsotonicRegression`).
    model_dict["latitude_matrix_deg_e"]: M-by-N numpy array of latitudes (deg
        north).
    model_dict["longitude_matrix_deg_e"]: M-by-N numpy array of longitudes (deg
        east).
    model_dict["field_names"]: length-F list with names of target fields.
    model_dict["pixel_radius_metres"]: Same as input arg.
    model_dict["weight_pixels_by_inverse_dist"]: Same as input arg.
    model_dict["weight_pixels_by_inverse_sq_dist"]: Same as input arg.
    model_dict["do_uncertainty_calibration"]: Same as input arg.
    model_dict["do_iso_reg_before_uncertainty_calib"]: Same as input arg.
    """

    # TODO(thunderhoser): Add option to subset by season -- probably.
    #  I guess this depends on what my detailed evaluation finds.

    # Check input args.
    error_checking.assert_is_boolean(one_model_per_pixel)
    error_checking.assert_is_boolean(do_uncertainty_calibration)
    if not one_model_per_pixel:
        do_multiprocessing = False

    error_checking.assert_is_boolean(do_multiprocessing)

    grid_latitudes_deg_n = None
    grid_longitudes_deg_e = None
    orig_eval_weight_matrix = None
    field_names = []

    for k in range(len(prediction_tables_xarray)):
        # prediction_tables_xarray[k] = prediction_io.take_ensemble_mean(
        #     prediction_tables_xarray[k]
        # )
        ptx = prediction_tables_xarray[k]

        if grid_latitudes_deg_n is None:
            grid_latitudes_deg_n = ptx[prediction_io.LATITUDE_KEY].values
            grid_longitudes_deg_e = ptx[prediction_io.LONGITUDE_KEY].values
            orig_eval_weight_matrix = ptx[prediction_io.WEIGHT_KEY].values
            field_names = ptx[prediction_io.FIELD_NAME_KEY].values.tolist()

        assert numpy.allclose(
            grid_latitudes_deg_n, ptx[prediction_io.LATITUDE_KEY].values,
            atol=TOLERANCE
        )
        assert numpy.allclose(
            grid_longitudes_deg_e, ptx[prediction_io.LONGITUDE_KEY].values,
            atol=TOLERANCE
        )
        assert numpy.allclose(
            orig_eval_weight_matrix, ptx[prediction_io.WEIGHT_KEY].values,
            atol=TOLERANCE
        )
        assert field_names == ptx[prediction_io.FIELD_NAME_KEY].values.tolist()

    latitude_matrix_deg_n, longitude_matrix_deg_e = (
        grids.latlng_vectors_to_matrices(
            unique_latitudes_deg=grid_latitudes_deg_n,
            unique_longitudes_deg=grid_longitudes_deg_e
        )
    )

    if not one_model_per_pixel:
        pixel_radius_metres = None
        weight_pixels_by_inverse_dist = None
        weight_pixels_by_inverse_sq_dist = None
    if not do_uncertainty_calibration:
        do_iso_reg_before_uncertainty_calib = None

    # Do actual stuff.
    num_model_grid_rows = (
        latitude_matrix_deg_n.shape[0] if one_model_per_pixel else 1
    )
    num_model_grid_columns = (
        latitude_matrix_deg_n.shape[1] if one_model_per_pixel else 1
    )
    num_fields = len(field_names)

    model_latitude_matrix_deg_n = (
        latitude_matrix_deg_n if one_model_per_pixel
        else numpy.full((1, 1), numpy.nan)
    )
    model_longitude_matrix_deg_e = (
        longitude_matrix_deg_e if one_model_per_pixel
        else numpy.full((1, 1), numpy.nan)
    )
    model_object_matrix = numpy.full(
        (num_model_grid_rows, num_model_grid_columns, num_fields),
        '', dtype=object
    )

    if do_multiprocessing:
        start_rows, end_rows = __get_slices_for_multiprocessing(
            num_grid_rows=num_model_grid_rows
        )

        argument_list = []
        for s, e in zip(start_rows, end_rows):
            argument_list.append((
                prediction_tables_xarray,
                numpy.linspace(s, e - 1, num=e - s, dtype=int),
                pixel_radius_metres,
                weight_pixels_by_inverse_dist,
                weight_pixels_by_inverse_sq_dist
            ))

        with Pool() as pool_object:
            subarrays = pool_object.starmap(
                _train_one_model_per_pixel, argument_list
            )

            for k in range(len(start_rows)):
                s = start_rows[k]
                e = end_rows[k]
                model_object_matrix[s:e, ...] = subarrays[k]

        return {
            MODELS_KEY: model_object_matrix,
            LATITUDES_KEY: model_latitude_matrix_deg_n,
            LONGITUDES_KEY: model_longitude_matrix_deg_e,
            FIELD_NAMES_KEY: field_names,
            PIXEL_RADIUS_KEY: pixel_radius_metres,
            WEIGHT_BY_INV_DIST_KEY: weight_pixels_by_inverse_dist,
            WEIGHT_BY_INV_SQ_DIST_KEY: weight_pixels_by_inverse_sq_dist,
            DO_UNCERTAINTY_CALIB_KEY: do_uncertainty_calibration,
            DO_IR_BEFORE_UC_KEY: do_iso_reg_before_uncertainty_calib
        }

    for f in range(num_fields):
        for i in range(num_model_grid_rows):
            for j in range(num_model_grid_columns):
                if (
                        one_model_per_pixel and
                        orig_eval_weight_matrix[i, j] <
                        MASK_PIXEL_IF_ORIG_WEIGHT_BELOW
                ):
                    model_object_matrix[i, j, f] = None
                    continue

                print((
                    'Training bias-correction model for '
                    '{0:d}th of {1:d} grid rows, '
                    '{2:d}th of {3:d} columns, '
                    '{4:d}th of {5:d} fields...'
                ).format(
                    i + 1,
                    num_model_grid_rows,
                    j + 1,
                    num_model_grid_columns,
                    f + 1,
                    num_fields
                ))

                these_prediction_tables_xarray = [
                    ptx.isel({
                        prediction_io.FIELD_DIM: numpy.array([f], dtype=int)
                    })
                    for ptx in prediction_tables_xarray
                ]

                if one_model_per_pixel:
                    these_prediction_tables_xarray = (
                        _subset_predictions_by_location(
                            prediction_tables_xarray=
                            these_prediction_tables_xarray,
                            desired_latitude_deg_n=latitude_matrix_deg_n[i, j],
                            desired_longitude_deg_e=
                            longitude_matrix_deg_e[i, j],
                            pixel_radius_metres=pixel_radius_metres,
                            weight_pixels_by_inverse_dist=
                            weight_pixels_by_inverse_dist,
                            weight_pixels_by_inverse_sq_dist=
                            weight_pixels_by_inverse_sq_dist
                        )
                    )

                model_object_matrix[i, j, f] = _train_one_model(
                    these_prediction_tables_xarray
                )

    return {
        MODELS_KEY: model_object_matrix,
        LATITUDES_KEY: model_latitude_matrix_deg_n,
        LONGITUDES_KEY: model_longitude_matrix_deg_e,
        FIELD_NAMES_KEY: field_names,
        PIXEL_RADIUS_KEY: pixel_radius_metres,
        WEIGHT_BY_INV_DIST_KEY: weight_pixels_by_inverse_dist,
        WEIGHT_BY_INV_SQ_DIST_KEY: weight_pixels_by_inverse_sq_dist,
        DO_UNCERTAINTY_CALIB_KEY: do_uncertainty_calibration,
        DO_IR_BEFORE_UC_KEY: do_iso_reg_before_uncertainty_calib
    }


def apply_model_suite(prediction_table_xarray, model_dict, verbose,
                      do_multiprocessing=True):
    """Applies model suite to new data in inference mode.

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :param model_dict: Dictionary in format created by `train_model_suite`.
    :param verbose: Boolean flag.
    :param do_multiprocessing: [used only if there is one model per pixel]
        Boolean flag.  If True, will do multi-threaded processing to make this
        go faster.
    :return: prediction_table_xarray: Same as input but with new predictions.
    """

    exec_start_time_unix_sec = time.time()
    error_checking.assert_is_boolean(verbose)

    field_names = model_dict[FIELD_NAMES_KEY]
    model_latitude_matrix_deg_n = model_dict[LATITUDES_KEY]
    model_longitude_matrix_deg_e = model_dict[LONGITUDES_KEY]
    model_object_matrix = model_dict[MODELS_KEY]
    do_uncertainty_calibration = model_dict[DO_UNCERTAINTY_CALIB_KEY]

    one_model_per_pixel = model_latitude_matrix_deg_n.size > 1
    ptx = prediction_table_xarray

    if one_model_per_pixel:
        pred_latitude_matrix_deg_n, pred_longitude_matrix_deg_e = (
            grids.latlng_vectors_to_matrices(
                unique_latitudes_deg=ptx[prediction_io.LATITUDE_KEY].values,
                unique_longitudes_deg=ptx[prediction_io.LONGITUDE_KEY].values
            )
        )

        assert numpy.allclose(
            model_latitude_matrix_deg_n, pred_latitude_matrix_deg_n,
            atol=TOLERANCE
        )
        assert numpy.allclose(
            model_longitude_matrix_deg_e, pred_longitude_matrix_deg_e,
            atol=TOLERANCE
        )
        assert (
            set(field_names) ==
            set(ptx[prediction_io.FIELD_NAME_KEY].values.tolist())
        )
    else:
        do_multiprocessing = False

    error_checking.assert_is_boolean(do_multiprocessing)
    prediction_matrix = ptx[prediction_io.PREDICTION_KEY].values

    if do_multiprocessing:
        start_rows, end_rows = __get_slices_for_multiprocessing(
            num_grid_rows=model_latitude_matrix_deg_n.shape[0]
        )

        argument_list = []
        for s, e in zip(start_rows, end_rows):
            argument_list.append((
                prediction_table_xarray,
                model_dict,
                numpy.linspace(s, e - 1, num=e - s, dtype=int),
                verbose
            ))

        with Pool() as pool_object:
            subarrays = pool_object.starmap(
                _apply_one_model_per_pixel, argument_list
            )

            for k in range(len(start_rows)):
                s = start_rows[k]
                e = end_rows[k]
                prediction_matrix[s:e, ...] = subarrays[k]

        ptx = ptx.assign({
            prediction_io.PREDICTION_KEY: (
                ptx[prediction_io.PREDICTION_KEY].dims,
                prediction_matrix
            )
        })

        print('Applying bias-correction model took {0:.4f} seconds.'.format(
            time.time() - exec_start_time_unix_sec
        ))
        return ptx

    num_fields = len(field_names)
    num_model_grid_rows = model_latitude_matrix_deg_n.shape[0]
    num_model_grid_columns = model_latitude_matrix_deg_n.shape[1]

    if do_uncertainty_calibration:
        ensemble_size = prediction_matrix.shape[-1]
        assert ensemble_size > 1

        mean_prediction_matrix = numpy.mean(prediction_matrix, axis=-1)
        prediction_stdev_matrix = numpy.std(
            prediction_matrix, axis=-1, ddof=1
        )
    else:
        mean_prediction_matrix = numpy.array([], dtype=float)
        prediction_stdev_matrix = numpy.array([], dtype=float)

    constrain_dsr = (
        canadian_fwi_utils.FWI_NAME in ptx[prediction_io.FIELD_NAME_KEY].values
        and
        canadian_fwi_utils.DSR_NAME in ptx[prediction_io.FIELD_NAME_KEY].values
    )

    for f_model in range(num_fields):
        if constrain_dsr and field_names[f_model] == canadian_fwi_utils.DSR_NAME:
            continue

        f_pred = numpy.where(
            ptx[prediction_io.FIELD_NAME_KEY].values == field_names[f_model]
        )[0][0]
        prediction_matrix_this_field = (
            ptx[prediction_io.PREDICTION_KEY].values[..., f_pred, :]
        )

        for i in range(num_model_grid_rows):
            for j in range(num_model_grid_columns):
                if (
                        one_model_per_pixel and
                        model_object_matrix[i, j, f_model] is None
                ):
                    continue

                if verbose:
                    print((
                        'Applying bias-correction model for '
                        '{0:d}th of {1:d} grid rows, '
                        '{2:d}th of {3:d} columns, '
                        '{4:d}th of {5:d} fields...'
                    ).format(
                        i + 1,
                        num_model_grid_rows,
                        j + 1,
                        num_model_grid_columns,
                        f_model + 1,
                        num_fields
                    ))

                if do_uncertainty_calibration:
                    if one_model_per_pixel:
                        orig_stdev = prediction_stdev_matrix[i, j, f_pred]
                        new_stdev = numpy.sqrt(
                            model_object_matrix[i, j, f_model].predict(
                                orig_stdev ** 2
                            )
                        )
                        stdev_inflation_factor = new_stdev / orig_stdev

                        if numpy.isnan(stdev_inflation_factor):
                            stdev_inflation_factor = 1.

                        stdev_inflation_factor = numpy.minimum(
                            stdev_inflation_factor, MAX_STDEV_INFLATION_FACTOR
                        )

                        prediction_matrix[i, j, f_pred, :] = (
                            mean_prediction_matrix[i, j, f_pred] +
                            stdev_inflation_factor * (
                                prediction_matrix[i, j, f_pred, :] -
                                mean_prediction_matrix[i, j, f_pred]
                            )
                        )
                    else:
                        orig_stdev_matrix = prediction_stdev_matrix[..., f_pred]
                        these_dims = orig_stdev_matrix.shape

                        new_stdevs = numpy.sqrt(
                            model_object_matrix[i, j, f_model].predict(
                                numpy.ravel(orig_stdev_matrix ** 2)
                            )
                        )
                        new_stdev_matrix = numpy.reshape(new_stdevs, these_dims)
                        stdev_inflation_matrix = (
                            new_stdev_matrix / orig_stdev_matrix
                        )

                        stdev_inflation_matrix[
                            numpy.isnan(stdev_inflation_matrix)
                        ] = 1.
                        stdev_inflation_matrix = numpy.minimum(
                            stdev_inflation_matrix, MAX_STDEV_INFLATION_FACTOR
                        )

                        stdev_inflation_matrix = numpy.expand_dims(
                            stdev_inflation_matrix, axis=-1
                        )
                        this_mean_pred_matrix = numpy.expand_dims(
                            mean_prediction_matrix[..., f_pred], axis=-1
                        )

                        prediction_matrix[..., f_pred, :] = (
                            this_mean_pred_matrix +
                            stdev_inflation_matrix * (
                                prediction_matrix[..., f_pred, :] -
                                this_mean_pred_matrix
                            )
                        )

                    continue

                if one_model_per_pixel:
                    prediction_matrix[i, j, f_pred, :] = (
                        model_object_matrix[i, j, f_model].predict(
                            prediction_matrix_this_field[i, j, :]
                        )
                    )
                else:
                    these_dims = prediction_matrix_this_field.shape
                    new_predictions = model_object_matrix[i, j, f_model].predict(
                        numpy.ravel(prediction_matrix_this_field)
                    )
                    prediction_matrix[..., f_pred, :] = numpy.reshape(
                        new_predictions, these_dims
                    )

    if constrain_dsr:
        fwi_index = numpy.where(
            ptx[prediction_io.FIELD_NAME_KEY].values ==
            canadian_fwi_utils.FWI_NAME
        )[0][0]

        dsr_index = numpy.where(
            ptx[prediction_io.FIELD_NAME_KEY].values ==
            canadian_fwi_utils.DSR_NAME
        )[0][0]

        prediction_matrix[..., dsr_index, :] = canadian_fwi_utils.fwi_to_dsr(
            prediction_matrix[..., fwi_index, :]
        )

    ptx = ptx.assign({
        prediction_io.PREDICTION_KEY: (
            ptx[prediction_io.PREDICTION_KEY].dims,
            prediction_matrix
        )
    })

    print('Applying bias-correction model took {0:.4f} seconds.'.format(
        time.time() - exec_start_time_unix_sec
    ))
    return ptx


def write_file(dill_file_name, model_dict):
    """Writes suite of bias-correction models to Dill file.

    :param dill_file_name: Path to output file.
    :param model_dict: Dictionary in format created by `train_model_suite`.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    dill.dump(model_dict, dill_file_handle)
    dill_file_handle.close()


def read_file(dill_file_name):
    """Reads suite of bias-correction models from Dill file.

    :param dill_file_name: Path to input file.
    :return: model_dict: Dictionary in format created by `train_model_suite`.
    """

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    model_dict = dill.load(dill_file_handle)
    dill_file_handle.close()

    if DO_UNCERTAINTY_CALIB_KEY not in model_dict:
        model_dict[DO_UNCERTAINTY_CALIB_KEY] = False
    if DO_IR_BEFORE_UC_KEY not in model_dict:
        model_dict[DO_IR_BEFORE_UC_KEY] = False

    missing_keys = list(set(ALL_KEYS) - set(model_dict.keys()))
    if len(missing_keys) == 0:
        return model_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), dill_file_name)

    raise ValueError(error_string)
