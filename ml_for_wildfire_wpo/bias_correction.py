"""Training and inference code for bias-correction models.

This module includes both isotonic regression, for bias-correcting the ensemble
mean, and uncertainty calibration, for bias-correcting the ensemble spread.
"""

import os
import sys
import time
from multiprocessing import Pool
import dill
import numpy
import xarray
from sklearn.isotonic import IsotonicRegression

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import file_system_utils
import error_checking
import prediction_io
import canadian_fwi_utils
import bias_clustering

TOLERANCE = 1e-6
MASK_PIXEL_IF_WEIGHT_BELOW = 0.01
MASK_PIXEL_IF_ORIG_WEIGHT_BELOW = 0.001
NUM_SLICES_FOR_MULTIPROCESSING = 24
MAX_STDEV_INFLATION_FACTOR = 1000.

GRID_LATITUDES_KEY = 'grid_latitudes_deg_n'
GRID_LONGITUDES_KEY = 'grid_longitudes_deg_e'
CLUSTER_ID_MATRIX_KEY = 'cluster_id_matrix'
FIELD_NAMES_KEY = 'field_names'
MODEL_DICT_KEY = 'field_and_cluster_to_model'
PIXEL_RADIUS_KEY = 'pixel_radius_metres'
WEIGHT_BY_INV_DIST_KEY = 'weight_pixels_by_inverse_dist'
WEIGHT_BY_INV_SQ_DIST_KEY = 'weight_pixels_by_inverse_sq_dist'
DO_UNCERTAINTY_CALIB_KEY = 'do_uncertainty_calibration'
DO_IR_BEFORE_UC_KEY = 'do_iso_reg_before_uncertainty_calib'

ALL_KEYS = [
    GRID_LATITUDES_KEY, GRID_LONGITUDES_KEY, CLUSTER_ID_MATRIX_KEY,
    FIELD_NAMES_KEY, MODEL_DICT_KEY,
    PIXEL_RADIUS_KEY, WEIGHT_BY_INV_DIST_KEY, WEIGHT_BY_INV_SQ_DIST_KEY,
    DO_UNCERTAINTY_CALIB_KEY, DO_IR_BEFORE_UC_KEY
]


def __subset_grid_for_multiprocessing(num_grid_rows):
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


def __subset_clusters_for_multiprocessing(cluster_ids):
    """Returns slices for multiprocessing.

    Each "slice" consists of several clusters.

    K = number of slices

    :param cluster_ids: 1-D numpy array of cluster IDs (positive integers).
    :return: slice_to_cluster_ids: Dictionary, where each key is a slice index
        (non-negative integer) and the corresponding value is a list of cluster
        IDs.
    """

    shuffled_cluster_ids = cluster_ids + 0
    numpy.random.shuffle(shuffled_cluster_ids)
    num_clusters = len(shuffled_cluster_ids)

    slice_indices_normalized = numpy.linspace(
        0, 1, num=NUM_SLICES_FOR_MULTIPROCESSING + 1, dtype=float
    )

    start_indices = numpy.round(
        num_clusters * slice_indices_normalized[:-1]
    ).astype(int)

    end_indices = numpy.round(
        num_clusters * slice_indices_normalized[1:]
    ).astype(int)

    slice_to_cluster_ids = dict()

    for i in range(len(start_indices)):
        slice_to_cluster_ids[i] = (
            shuffled_cluster_ids[start_indices[i]:end_indices[i]]
        )

    return slice_to_cluster_ids


def _subset_predictions_to_pixel(
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


def _subset_predictions_to_cluster(
        prediction_tables_xarray, cluster_table_xarray, desired_cluster_id):
    """Subsets predictions by location.

    :param prediction_tables_xarray: See input documentation for
        `train_model_suite`.
    :param cluster_table_xarray: Same.
    :param desired_cluster_id: Will subset predictions to cluster k, where
        k = `desired_cluster_id`.
    :return: new_prediction_tables_xarray: Same as input, except with a smaller
        grid and different evaluation weights.
    """

    # This method automatically handles NaN predictions, because only grid
    # points without NaN predictions (i.e., where NaN is impossible) are
    # assigned to a real cluster (with index > 0).
    assert desired_cluster_id > 0

    cluster_id_matrix = (
        cluster_table_xarray[bias_clustering.CLUSTER_ID_KEY].values[..., 0]
    )
    good_rows, good_columns = numpy.where(
        cluster_id_matrix == desired_cluster_id
    )
    good_rows = numpy.unique(good_rows)
    good_columns = numpy.unique(good_columns)

    first_ptx = prediction_tables_xarray[0]
    eval_weight_submatrix = (
        first_ptx[prediction_io.WEIGHT_KEY].values[good_rows, :][:, good_columns]
    )
    cluster_id_submatrix = cluster_id_matrix[good_rows, :][:, good_columns]
    eval_weight_submatrix[cluster_id_submatrix != desired_cluster_id] = 0.

    num_tables = len(prediction_tables_xarray)
    new_prediction_tables_xarray = [xarray.Dataset()] * num_tables

    for k in range(num_tables):
        new_prediction_tables_xarray[k] = prediction_tables_xarray[k].isel(
            {prediction_io.ROW_DIM: good_rows}
        )
        new_prediction_tables_xarray[k] = new_prediction_tables_xarray[k].isel(
            {prediction_io.COLUMN_DIM: good_columns}
        )
        new_prediction_tables_xarray[k] = new_prediction_tables_xarray[k].assign({
            prediction_io.WEIGHT_KEY: (
                new_prediction_tables_xarray[k][prediction_io.WEIGHT_KEY].dims,
                eval_weight_submatrix
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
                    _subset_predictions_to_pixel(
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


def _train_one_model_per_cluster(prediction_tables_xarray, cluster_table_xarray,
                                 train_for_cluster_ids, field_name):
    """Trains one model per cluster, using multiprocessing.

    :param prediction_tables_xarray: See documentation for `train_model_suite`.
    :param cluster_table_xarray: Same.
    :param train_for_cluster_ids: 1-D numpy array of cluster IDs for which to
        train a model.
    :return: field_and_cluster_to_model: See output doc for `train_model_suite`.
        This will be a subset of that final dictionary.
    """

    field_and_cluster_to_model = dict()
    num_clusters = len(train_for_cluster_ids)

    for k in range(num_clusters):
        print((
            'Training bias-correction model for {0:d}th of {1:d} clusters...'
        ).format(
            k + 1, num_clusters
        ))

        these_prediction_tables_xarray = _subset_predictions_to_cluster(
            prediction_tables_xarray=prediction_tables_xarray,
            cluster_table_xarray=cluster_table_xarray,
            desired_cluster_id=train_for_cluster_ids[k]
        )

        this_key = (field_name, train_for_cluster_ids[k])
        field_and_cluster_to_model[this_key] = _train_one_model(
            these_prediction_tables_xarray
        )

    return field_and_cluster_to_model


def train_model_suite(
        prediction_tables_xarray, do_uncertainty_calibration,
        cluster_table_xarray=None,
        pixel_radius_metres=None,
        weight_pixels_by_inverse_dist=None,
        weight_pixels_by_inverse_sq_dist=None,
        do_iso_reg_before_uncertainty_calib=None,
        do_multiprocessing=True):
    """Trains one suite of bias-correction models.

    The model suite may have any of the following setups:

    - One model per target field
    - One model per target field per spatial pixel
    - One model per target field per spatial cluster

    M = number of rows in grid
    N = number of columns in grid
    F = number of target fields

    :param prediction_tables_xarray: 1-D list of xarray tables in format
        returned by `prediction_io.read_file`.
    :param do_uncertainty_calibration: Boolean flag.  If True, this method will
        [1] assume that every "prediction" in `prediction_tables_xarray` is the
        ensemble variance; [2] assume that every "target" in
        `prediction_tables_xarray` is the squared error of the ensemble mean;
        [3] train IR models to adjust the ensemble variance, i.e., to do
        uncertainty calibration.  If False, this method will do standard
        isotonic regression, correcting only the ensemble mean.
    :param cluster_table_xarray: xarray table in format returned by
        `bias_clustering.read_file`.  If you do *not* want one model per field
        per spatial cluster, make this None.
    :param pixel_radius_metres: When training the model for pixel P, will use
        all pixels within this radius of P.  If you do *not* want one model per
        field per spatial pixel, make this None.
    :param weight_pixels_by_inverse_dist: Boolean flag.  If True, when training
        the model for pixel P, will weight every other pixel by the inverse of
        its distance to P.  If you do *not* want one model per field per spatial
        pixel, make this None.
    :param weight_pixels_by_inverse_sq_dist: Boolean flag.  If True, when
        training the model for pixel P, will weight every other pixel by the
        inverse of its squared distance to P.  If you do *not* want one model
        per field per spatial pixel, make this None.
    :param do_iso_reg_before_uncertainty_calib:
        [used only if `do_uncertainty_calibration == True`]
        Boolean flag, indicating whether isotonic regression has been done
        before uncertainty calibration.
    :param do_multiprocessing: [used only if `one_model_per_pixel == True`]
        Boolean flag.  If True, will do multi-threaded processing to speed up
        training.

    :return: model_dict: Dictionary with the following keys.
    model_dict["grid_latitudes_deg_n"]: length-M numpy array of latitudes (deg
        north).
    model_dict["grid_longitudes_deg_e"]: length-N numpy array of longitudes (deg
        east).
    model_dict["cluster_id_matrix"]: M-by-N-by-F numpy array of cluster IDs.
    model_dict["field_names"]: length-F list with names of target fields.
    model_dict["field_and_cluster_to_model"]: Double-indexed dictionary.  Each
        key is (target_field_name, cluster_id), and the corresponding value is a
        trained bias-correction model (instance of
        `sklearn.isotonic.IsotonicRegression`).
    model_dict["pixel_radius_metres"]: Same as input arg.
    model_dict["weight_pixels_by_inverse_dist"]: Same as input arg.
    model_dict["weight_pixels_by_inverse_sq_dist"]: Same as input arg.
    model_dict["do_uncertainty_calibration"]: Same as input arg.
    model_dict["do_iso_reg_before_uncertainty_calib"]: Same as input arg.
    """

    # TODO(thunderhoser): This method should now handle all 3 types of model
    # suites -- but I need to verify!  And then I also need to modify the main
    # inference method.

    # Check input args.
    one_model_per_cluster = cluster_table_xarray is not None
    if one_model_per_cluster:
        pixel_radius_metres = None
        weight_pixels_by_inverse_dist = None
        weight_pixels_by_inverse_sq_dist = None

    one_model_per_pixel = (
        pixel_radius_metres is not None
        or weight_pixels_by_inverse_dist is not None
        or weight_pixels_by_inverse_sq_dist is not None
    )

    if not one_model_per_pixel or one_model_per_cluster:
        do_multiprocessing = False

    error_checking.assert_is_boolean(do_uncertainty_calibration)
    if not do_uncertainty_calibration:
        do_iso_reg_before_uncertainty_calib = None

    error_checking.assert_is_boolean(do_multiprocessing)
    error_checking.assert_is_boolean(do_iso_reg_before_uncertainty_calib)

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

    num_grid_rows = len(grid_latitudes_deg_n)
    num_grid_columns = len(grid_longitudes_deg_e)
    num_fields = len(field_names)

    if one_model_per_pixel:
        model_object_matrix = numpy.full(
            (num_grid_rows, num_grid_columns, num_fields), '', dtype=object
        )

        if do_multiprocessing:
            start_rows, end_rows = __subset_grid_for_multiprocessing(
                num_grid_rows=num_grid_rows
            )
        else:
            start_rows = numpy.array([0], dtype=int)
            end_rows = numpy.array([num_grid_rows], dtype=int)

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

        num_grid_points = num_grid_rows * num_grid_columns
        unique_cluster_ids = numpy.linspace(
            1, num_grid_points, num=num_grid_points, dtype=int
        )
        cluster_id_matrix = numpy.reshape(
            unique_cluster_ids, (num_grid_rows, num_grid_columns)
        )
        cluster_id_matrix = numpy.expand_dims(cluster_id_matrix, axis=-1)
        cluster_id_matrix = numpy.repeat(
            cluster_id_matrix, axis=-1, repeats=num_fields
        )

        field_and_cluster_to_model = dict()

        for i in range(num_grid_rows):
            for j in range(num_grid_columns):
                for f in range(num_fields):
                    this_key = (field_names[f], cluster_id_matrix[i, j])
                    field_and_cluster_to_model[this_key] = (
                        model_object_matrix[i, j, f]
                    )

        return {
            GRID_LATITUDES_KEY: grid_latitudes_deg_n,
            GRID_LONGITUDES_KEY: grid_longitudes_deg_e,
            FIELD_NAMES_KEY: field_names,
            CLUSTER_ID_MATRIX_KEY:
                cluster_table_xarray[bias_clustering.CLUSTER_ID_KEY].values,
            MODEL_DICT_KEY: field_and_cluster_to_model,
            PIXEL_RADIUS_KEY: pixel_radius_metres,
            WEIGHT_BY_INV_DIST_KEY: weight_pixels_by_inverse_dist,
            WEIGHT_BY_INV_SQ_DIST_KEY: weight_pixels_by_inverse_sq_dist,
            DO_UNCERTAINTY_CALIB_KEY: do_uncertainty_calibration,
            DO_IR_BEFORE_UC_KEY: do_iso_reg_before_uncertainty_calib
        }

    if one_model_per_cluster:
        sort_indices = numpy.array([
            numpy.squeeze(numpy.where(
                cluster_table_xarray[bias_clustering.FIELD_NAME_KEY].values == f
            )[0]) for f in field_names
        ], dtype=int)

        cluster_table_xarray = cluster_table_xarray.isel(
            {bias_clustering.FIELD_DIM: sort_indices}
        )

        field_and_cluster_to_model = dict()

        for f in range(num_fields):
            cluster_table_xarray_1field = cluster_table_xarray.isel({
                bias_clustering.FIELD_DIM: numpy.array([f], dtype=int)
            })

            prediction_tables_xarray_1field = [
                ptx.isel({
                    prediction_io.FIELD_DIM: numpy.array([f], dtype=int)
                })
                for ptx in prediction_tables_xarray
            ]

            ctx = cluster_table_xarray_1field
            unique_cluster_ids = numpy.unique(
                ctx[bias_clustering.CLUSTER_ID_KEY].values[..., 0]
            )
            unique_cluster_ids = unique_cluster_ids[unique_cluster_ids > 0]

            if do_multiprocessing:
                slice_to_cluster_ids = __subset_clusters_for_multiprocessing(
                    cluster_ids=unique_cluster_ids
                )
            else:
                slice_to_cluster_ids = {0: unique_cluster_ids}

            argument_list = []
            for this_slice in slice_to_cluster_ids:
                argument_list.append((
                    prediction_tables_xarray_1field,
                    cluster_table_xarray_1field,
                    slice_to_cluster_ids[this_slice]
                ))

            with Pool() as pool_object:
                subdicts = pool_object.starmap(
                    _train_one_model_per_cluster, argument_list
                )

                for k in range(len(subdicts)):
                    field_and_cluster_to_model.update(subdicts[k])

        for this_key in field_and_cluster_to_model:
            assert field_and_cluster_to_model[this_key] is not None

        return {
            GRID_LATITUDES_KEY: grid_latitudes_deg_n,
            GRID_LONGITUDES_KEY: grid_longitudes_deg_e,
            FIELD_NAMES_KEY: field_names,
            CLUSTER_ID_MATRIX_KEY:
                cluster_table_xarray[bias_clustering.CLUSTER_ID_KEY].values,
            MODEL_DICT_KEY: field_and_cluster_to_model,
            PIXEL_RADIUS_KEY: pixel_radius_metres,
            WEIGHT_BY_INV_DIST_KEY: weight_pixels_by_inverse_dist,
            WEIGHT_BY_INV_SQ_DIST_KEY: weight_pixels_by_inverse_sq_dist,
            DO_UNCERTAINTY_CALIB_KEY: do_uncertainty_calibration,
            DO_IR_BEFORE_UC_KEY: do_iso_reg_before_uncertainty_calib
        }

    cluster_id_matrix = numpy.full(
        (num_grid_rows, num_grid_columns, num_fields), 1, dtype=int
    )
    field_and_cluster_to_model = dict()

    for f in range(num_fields):
        print((
            'Training bias-correction model for {0:d}th of {1:d} fields...'
        ).format(
            f + 1, num_fields
        ))

        these_prediction_tables_xarray = [
            ptx.isel({
                prediction_io.FIELD_DIM: numpy.array([f], dtype=int)
            })
            for ptx in prediction_tables_xarray
        ]

        this_model_object = _train_one_model(these_prediction_tables_xarray)
        field_and_cluster_to_model[(field_names[f], 1)] = this_model_object

    return {
        GRID_LATITUDES_KEY: grid_latitudes_deg_n,
        GRID_LONGITUDES_KEY: grid_longitudes_deg_e,
        FIELD_NAMES_KEY: field_names,
        CLUSTER_ID_MATRIX_KEY: cluster_id_matrix,
        MODEL_DICT_KEY: field_and_cluster_to_model,
        PIXEL_RADIUS_KEY: pixel_radius_metres,
        WEIGHT_BY_INV_DIST_KEY: weight_pixels_by_inverse_dist,
        WEIGHT_BY_INV_SQ_DIST_KEY: weight_pixels_by_inverse_sq_dist,
        DO_UNCERTAINTY_CALIB_KEY: do_uncertainty_calibration,
        DO_IR_BEFORE_UC_KEY: do_iso_reg_before_uncertainty_calib
    }


def apply_model_suite(prediction_table_xarray, model_dict, verbose):
    """Applies model suite to new data in inference mode.

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :param model_dict: Dictionary in format created by `train_model_suite`.
    :param verbose: Boolean flag.
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

    prediction_matrix = ptx[prediction_io.PREDICTION_KEY].values
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
                                numpy.array([orig_stdev], dtype=float) ** 2
                            )
                        )[0]
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

    prediction_matrix = numpy.maximum(prediction_matrix, 0.)

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

    missing_keys = list(set(ALL_KEYS) - set(model_dict.keys()))
    if len(missing_keys) == 0:
        return model_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), dill_file_name)

    raise ValueError(error_string)
