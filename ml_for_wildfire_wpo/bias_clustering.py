"""Creates spatially connected clusters with similar model bias."""

import os
import sys
import warnings
from multiprocessing import Pool
import numpy
import xarray
import netCDF4
from scipy.ndimage import label

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import longitude_conversion as lng_conversion
import number_rounding
import file_system_utils
import error_checking
import misc_utils

TOLERANCE = 1e-6
NUM_SLICES_FOR_MULTIPROCESSING = 24

ROW_DIM = 'grid_row'
COLUMN_DIM = 'grid_column'
FIELD_DIM = 'field'
FIELD_CHAR_DIM = 'field_char'

CLUSTER_ID_KEY = 'cluster_id'
LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'
FIELD_NAME_KEY = 'field_name'


def __get_slices_for_multiprocessing(bin_ids):
    """Returns slices for multiprocessing.

    Each "slice" consists of several bins.

    K = number of slices

    :param bin_ids: 1-D numpy array of bin IDs (positive integers).
    :return: slice_to_bin_ids: Dictionary, where each key is a slice index
        (non-negative integer) and the corresponding value is a list of bin IDs.
    """

    shuffled_bin_ids = bin_ids + 0
    numpy.random.shuffle(shuffled_bin_ids)
    num_bins = len(shuffled_bin_ids)

    slice_indices_normalized = numpy.linspace(
        0, 1, num=NUM_SLICES_FOR_MULTIPROCESSING + 1, dtype=float
    )

    start_indices = numpy.round(
        num_bins * slice_indices_normalized[:-1]
    ).astype(int)

    end_indices = numpy.round(
        num_bins * slice_indices_normalized[1:]
    ).astype(int)

    slice_to_bin_ids = dict()
    for i in range(len(start_indices)):
        slice_to_bin_ids[i] = shuffled_bin_ids[start_indices[i]:end_indices[i]]

    return slice_to_bin_ids


def __find_clusters_one_scale(
        bin_id_matrix, unique_bin_ids_to_process,
        cluster_id_matrix, last_cluster_id,
        buffer_distance_px, min_cluster_size):
    """Finds clusters at one spatial scale.

    M = number of rows in grid
    N = number of columns in grid

    :param bin_id_matrix: M-by-N numpy array of bin IDs (positive integers).
        For pixels where the bias is NaN, a bin cannot be assigned and the value
        in this array will be -1.
    :param unique_bin_ids_to_process: 1-D numpy array with bin IDs to process
        right now.
    :param cluster_id_matrix: M-by-N numpy array of current cluster IDs, which
        will be updated by this method.
    :param last_cluster_id: Last cluster ID assigned.  This method will only
        increase the cluster ID.
    :param buffer_distance_px: See documentation for `find_clusters`.
    :param min_cluster_size: Same.
    :return: cluster_id_matrix: Updated version of input.
    :return: last_cluster_id: Updated version of input.
    """

    for i in range(len(unique_bin_ids_to_process)):
        print('Have processed {0:d} of {1:d} bins...'.format(
            i, len(unique_bin_ids_to_process)
        ))

        ith_bin_flag_matrix = (
            bin_id_matrix == unique_bin_ids_to_process[i]
        ).astype(int)

        if buffer_distance_px > TOLERANCE:
            ith_bin_flag_matrix = misc_utils.dilate_binary_matrix(
                binary_matrix=ith_bin_flag_matrix,
                buffer_distance_px=buffer_distance_px
            )

        this_cluster_id_matrix = label(
            input=ith_bin_flag_matrix,
            structure=numpy.full((3, 3), 1, dtype=int)
        )[0]

        these_unique_cluster_ids = numpy.unique(this_cluster_id_matrix)
        these_unique_cluster_ids = these_unique_cluster_ids[
            these_unique_cluster_ids > 0
        ]

        for this_cluster_id in these_unique_cluster_ids:
            this_cluster_mask = numpy.logical_and(
                this_cluster_id_matrix == this_cluster_id,
                bin_id_matrix == unique_bin_ids_to_process[i]
            )

            this_cluster_size = numpy.sum(this_cluster_mask)
            if this_cluster_size < min_cluster_size:
                continue

            last_cluster_id += 1
            cluster_id_matrix[this_cluster_mask] = last_cluster_id

    return cluster_id_matrix, last_cluster_id


def __salvage_clusters(
        bin_id_matrix, unique_bin_ids_to_process,
        cluster_id_matrix, last_cluster_id, buffer_distance_px):
    """Salvages clusters.

    :param bin_id_matrix: See documentation for `__find_clusters_one_scale`.
    :param unique_bin_ids_to_process: Same.
    :param cluster_id_matrix: Same.
    :param last_cluster_id: Same.
    :param buffer_distance_px: Same.
    :return: cluster_id_matrix: Updated version of input.
    :return: last_cluster_id: Updated version of input.
    :return: num_pixels_salvaged: Number of pixels salvaged into a cluster.
    """

    num_pixels_salvaged = 0

    for i in range(len(unique_bin_ids_to_process)):
        print('Have processed {0:d} of {1:d} bins...'.format(
            i, len(unique_bin_ids_to_process)
        ))

        ith_bin_flag_matrix = (
            bin_id_matrix == unique_bin_ids_to_process[i]
        ).astype(int)

        if buffer_distance_px > TOLERANCE:
            ith_bin_flag_matrix = misc_utils.dilate_binary_matrix(
                binary_matrix=ith_bin_flag_matrix,
                buffer_distance_px=buffer_distance_px
            )

        this_cluster_id_matrix = label(
            input=ith_bin_flag_matrix,
            structure=numpy.full((3, 3), 1, dtype=int)
        )[0]

        these_unique_cluster_ids = numpy.unique(this_cluster_id_matrix)
        these_unique_cluster_ids = these_unique_cluster_ids[
            these_unique_cluster_ids > 0
        ]

        for this_cluster_id in these_unique_cluster_ids:
            this_cluster_mask = numpy.logical_and(
                this_cluster_id_matrix == this_cluster_id,
                bin_id_matrix == unique_bin_ids_to_process[i]
            )
            this_cluster_mask = numpy.logical_and(
                this_cluster_mask,
                cluster_id_matrix == 0
            )
            if numpy.sum(this_cluster_mask) == 0:
                continue

            last_cluster_id += 1
            cluster_id_matrix[this_cluster_mask] = last_cluster_id
            num_pixels_salvaged += numpy.sum(this_cluster_mask)

    return cluster_id_matrix, last_cluster_id, num_pixels_salvaged


def _report_cluster_membership(cluster_id_matrix):
    """Reports cluster membership.

    :param cluster_id_matrix: 2-D numpy array of integer IDs.
    """

    _, unique_cluster_counts = numpy.unique(
        cluster_id_matrix[cluster_id_matrix != 0],
        return_counts=True
    )
    unique_cluster_counts = numpy.sort(unique_cluster_counts)

    for i in range(len(unique_cluster_counts)):
        print('Number of pixels in {0:d}th cluster = {1:d}'.format(
            i + 1,
            unique_cluster_counts[i]
        ))


def _discretize_biases(bias_matrix, discretization_interval):
    """Discretizes biases.

    M = number of rows in grid
    N = number of columns in grid

    :param bias_matrix: See documentation for `find_clusters`.
    :param discretization_interval: Same.
    :return: bin_id_matrix: M-by-N numpy array of bin IDs (positive integers).
        For pixels where the bias is NaN, a bin cannot be assigned and the value
        in this array will be -1.
    """

    first_bin_edge = number_rounding.floor_to_nearest(
        numpy.nanmin(bias_matrix),
        discretization_interval
    )
    last_bin_edge = number_rounding.ceiling_to_nearest(
        numpy.nanmax(bias_matrix),
        discretization_interval
    )
    num_bin_edges = 1 + int(numpy.round(
        (last_bin_edge - first_bin_edge) / discretization_interval
    ))
    bin_edges = numpy.linspace(
        first_bin_edge, last_bin_edge, num=num_bin_edges, dtype=float
    )

    bin_id_matrix = numpy.digitize(x=bias_matrix, bins=bin_edges, right=False)
    bin_id_matrix[numpy.isnan(bias_matrix)] = -1

    return bin_id_matrix


def find_clusters_one_field(
        bias_matrix, min_cluster_size, bias_discretization_intervals,
        buffer_distance_px, do_multiprocessing=True):
    """Finds clusters for one target field.

    M = number of rows in grid
    N = number of columns in grid

    :param bias_matrix: M-by-N numpy array of biases.
    :param min_cluster_size: Minimum cluster size (number of pixels).
    :param bias_discretization_intervals: 1-D list of bias-discretization
        intervals, from the smallest scale to the largest.
    :param buffer_distance_px: Buffer distance (number of pixels).  A non-zero
        buffer distance allows a cluster to be a non-simply-connected spatial
        region.
    :param do_multiprocessing: Boolean flag.
    :return: cluster_id_matrix: M-by-N numpy array of cluster IDs (positive
        integers).  For pixels where the bias is NaN, a cluster cannot be
        assigned and the value in this array will be -1.
    """

    error_checking.assert_is_numpy_array(bias_matrix, num_dimensions=2)
    error_checking.assert_is_integer(min_cluster_size)
    error_checking.assert_is_geq(min_cluster_size, 1)
    error_checking.assert_is_geq(buffer_distance_px, 0.)
    error_checking.assert_is_boolean(do_multiprocessing)
    error_checking.assert_is_numpy_array(
        bias_discretization_intervals, num_dimensions=1
    )
    error_checking.assert_is_greater_numpy_array(
        bias_discretization_intervals, 0.
    )
    bias_discretization_intervals = numpy.unique(bias_discretization_intervals)

    num_scales = len(bias_discretization_intervals)
    num_rows = bias_matrix.shape[0]
    num_columns = bias_matrix.shape[1]

    cluster_id_matrix = numpy.full((num_rows, num_columns), 0, dtype=int)
    cluster_id_matrix[numpy.isnan(bias_matrix)] = -1
    last_cluster_id = 0
    orig_bias_matrix = bias_matrix + 0.

    for k in range(num_scales):
        print((
            'Finding clusters for bias-discretization interval = {0:f}...'
        ).format(
            bias_discretization_intervals[k]
        ))

        bias_matrix[cluster_id_matrix > 0] = numpy.nan

        bin_id_matrix = _discretize_biases(
            bias_matrix=bias_matrix,
            discretization_interval=bias_discretization_intervals[k]
        )
        unique_bin_ids = numpy.unique(bin_id_matrix[bin_id_matrix != -1])

        if do_multiprocessing:
            slice_to_bin_ids = __get_slices_for_multiprocessing(
                bin_ids=unique_bin_ids
            )

            argument_list = []
            for this_slice in slice_to_bin_ids:
                last_cluster_id += int(1e5)
                argument_list.append((
                    bin_id_matrix,
                    slice_to_bin_ids[this_slice],
                    cluster_id_matrix,
                    last_cluster_id,
                    buffer_distance_px,
                    min_cluster_size
                ))

            with Pool() as pool_object:
                cluster_id_matrices, last_cluster_ids = zip(
                    *pool_object.starmap(
                        __find_clusters_one_scale, argument_list
                    )
                )

                last_cluster_id = max([l for l in last_cluster_ids])

                for k in range(len(cluster_id_matrices)):
                    cluster_id_matrix = numpy.maximum(
                        cluster_id_matrix, cluster_id_matrices[k]
                    )
        else:
            cluster_id_matrix, last_cluster_id = __find_clusters_one_scale(
                bin_id_matrix=bin_id_matrix,
                unique_bin_ids_to_process=unique_bin_ids,
                cluster_id_matrix=cluster_id_matrix,
                last_cluster_id=last_cluster_id,
                buffer_distance_px=buffer_distance_px,
                min_cluster_size=min_cluster_size
            )

    if not numpy.any(cluster_id_matrix == 0):
        _report_cluster_membership(cluster_id_matrix)
        return cluster_id_matrix

    print('Finding clusters for bias-discretization interval = {0:f}...'.format(
        bias_discretization_intervals[0]
    ))

    bin_id_matrix = _discretize_biases(
        bias_matrix=orig_bias_matrix,
        discretization_interval=bias_discretization_intervals[0]
    )
    unique_bin_ids = numpy.unique(bin_id_matrix[bin_id_matrix != -1])

    if do_multiprocessing:
        slice_to_bin_ids = __get_slices_for_multiprocessing(
            bin_ids=unique_bin_ids
        )

        argument_list = []
        for this_slice in slice_to_bin_ids:
            last_cluster_id += int(1e5)
            argument_list.append((
                bin_id_matrix,
                slice_to_bin_ids[this_slice],
                cluster_id_matrix,
                last_cluster_id,
                buffer_distance_px
            ))

        with Pool() as pool_object:
            cluster_id_matrices, last_cluster_ids, salvaged_pixel_counts = zip(
                *pool_object.starmap(__salvage_clusters, argument_list)
            )

            last_cluster_id = max([l for l in last_cluster_ids])
            num_pixels_salvaged = numpy.sum(salvaged_pixel_counts)

            for k in range(len(cluster_id_matrices)):
                cluster_id_matrix = numpy.maximum(
                    cluster_id_matrix, cluster_id_matrices[k]
                )
    else:
        cluster_id_matrix, last_cluster_id, num_pixels_salvaged = (
            __salvage_clusters(
                bin_id_matrix=bin_id_matrix,
                unique_bin_ids_to_process=unique_bin_ids,
                cluster_id_matrix=cluster_id_matrix,
                last_cluster_id=last_cluster_id,
                buffer_distance_px=buffer_distance_px
            )
        )

    print('Number of pixels salvaged from smallest scale = {0:d}'.format(
        num_pixels_salvaged
    ))

    if numpy.any(cluster_id_matrix == 0):
        warning_string = (
            'POTENTIAL MAJOR ERROR: {0:d} pixels have cluster ID of 0!'
        ).format(numpy.sum(cluster_id_matrix == 0))

        warnings.warn(warning_string)

    _report_cluster_membership(cluster_id_matrix)
    return cluster_id_matrix


def find_clusters_one_field_backwards(
        bias_matrix, min_cluster_size, bias_discretization_intervals,
        buffer_distance_px, do_multiprocessing=True):
    """Finds clusters for one target field.

    M = number of rows in grid
    N = number of columns in grid

    :param bias_matrix: M-by-N numpy array of biases.
    :param min_cluster_size: Minimum cluster size (number of pixels).
    :param bias_discretization_intervals: 1-D list of bias-discretization
        intervals, from the smallest scale to the largest.
    :param buffer_distance_px: Buffer distance (number of pixels).  A non-zero
        buffer distance allows a cluster to be a non-simply-connected spatial
        region.
    :param do_multiprocessing: Boolean flag.
    :return: cluster_id_matrix: M-by-N numpy array of cluster IDs (positive
        integers).  For pixels where the bias is NaN, a cluster cannot be
        assigned and the value in this array will be -1.
    """

    error_checking.assert_is_numpy_array(bias_matrix, num_dimensions=2)
    error_checking.assert_is_integer(min_cluster_size)
    error_checking.assert_is_geq(min_cluster_size, 1)
    error_checking.assert_is_geq(buffer_distance_px, 0.)
    error_checking.assert_is_boolean(do_multiprocessing)
    error_checking.assert_is_numpy_array(
        bias_discretization_intervals, num_dimensions=1
    )
    error_checking.assert_is_greater_numpy_array(
        bias_discretization_intervals, 0.
    )
    bias_discretization_intervals = numpy.unique(
        bias_discretization_intervals
    )[::-1]

    num_scales = len(bias_discretization_intervals)
    num_rows = bias_matrix.shape[0]
    num_columns = bias_matrix.shape[1]

    cluster_id_matrix = numpy.full((num_rows, num_columns), 0, dtype=int)
    cluster_id_matrix[numpy.isnan(bias_matrix)] = -1
    last_cluster_id = 0

    for k in range(num_scales):
        print((
            'Finding clusters for bias-discretization interval = {0:f}...'
        ).format(
            bias_discretization_intervals[k]
        ))

        bin_id_matrix = _discretize_biases(
            bias_matrix=bias_matrix,
            discretization_interval=bias_discretization_intervals[k]
        )
        unique_bin_ids = numpy.unique(bin_id_matrix[bin_id_matrix != -1])

        if do_multiprocessing:
            slice_to_bin_ids = __get_slices_for_multiprocessing(
                bin_ids=unique_bin_ids
            )

            argument_list = []
            for this_slice in slice_to_bin_ids:
                last_cluster_id += int(1e5)
                argument_list.append((
                    bin_id_matrix,
                    slice_to_bin_ids[this_slice],
                    cluster_id_matrix,
                    last_cluster_id,
                    buffer_distance_px,
                    min_cluster_size
                ))

            with Pool() as pool_object:
                cluster_id_matrices, last_cluster_ids = zip(
                    *pool_object.starmap(
                        __find_clusters_one_scale, argument_list
                    )
                )

                last_cluster_id = max([l for l in last_cluster_ids])

                for k in range(len(cluster_id_matrices)):
                    cluster_id_matrix = numpy.maximum(
                        cluster_id_matrix, cluster_id_matrices[k]
                    )
        else:
            cluster_id_matrix, last_cluster_id = __find_clusters_one_scale(
                bin_id_matrix=bin_id_matrix,
                unique_bin_ids_to_process=unique_bin_ids,
                cluster_id_matrix=cluster_id_matrix,
                last_cluster_id=last_cluster_id,
                buffer_distance_px=buffer_distance_px,
                min_cluster_size=min_cluster_size
            )

    if not numpy.any(cluster_id_matrix == 0):
        _report_cluster_membership(cluster_id_matrix)
        return cluster_id_matrix

    print('Finding clusters for bias-discretization interval = {0:f}...'.format(
        bias_discretization_intervals[0]
    ))

    bin_id_matrix = _discretize_biases(
        bias_matrix=bias_matrix,
        discretization_interval=bias_discretization_intervals[0]
    )
    unique_bin_ids = numpy.unique(bin_id_matrix[bin_id_matrix != -1])

    if do_multiprocessing:
        slice_to_bin_ids = __get_slices_for_multiprocessing(
            bin_ids=unique_bin_ids
        )

        argument_list = []
        for this_slice in slice_to_bin_ids:
            last_cluster_id += int(1e5)
            argument_list.append((
                bin_id_matrix,
                slice_to_bin_ids[this_slice],
                cluster_id_matrix,
                last_cluster_id,
                buffer_distance_px
            ))

        with Pool() as pool_object:
            cluster_id_matrices, last_cluster_ids, salvaged_pixel_counts = zip(
                *pool_object.starmap(__salvage_clusters, argument_list)
            )

            last_cluster_id = max([l for l in last_cluster_ids])
            num_pixels_salvaged = numpy.sum(salvaged_pixel_counts)

            for k in range(len(cluster_id_matrices)):
                cluster_id_matrix = numpy.maximum(
                    cluster_id_matrix, cluster_id_matrices[k]
                )
    else:
        cluster_id_matrix, last_cluster_id, num_pixels_salvaged = (
            __salvage_clusters(
                bin_id_matrix=bin_id_matrix,
                unique_bin_ids_to_process=unique_bin_ids,
                cluster_id_matrix=cluster_id_matrix,
                last_cluster_id=last_cluster_id,
                buffer_distance_px=buffer_distance_px
            )
        )

    print('Number of pixels salvaged from largest scale = {0:d}'.format(
        num_pixels_salvaged
    ))

    if numpy.any(cluster_id_matrix == 0):
        warning_string = (
            'POTENTIAL MAJOR ERROR: {0:d} pixels have cluster ID of 0!'
        ).format(numpy.sum(cluster_id_matrix == 0))

        warnings.warn(warning_string)

    _report_cluster_membership(cluster_id_matrix)
    return cluster_id_matrix


def write_file(
        netcdf_file_name, cluster_id_matrix,
        grid_latitudes_deg_n, grid_longitudes_deg_e, field_names):
    """Writes bias-clustering to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid
    F = number of target fields

    :param netcdf_file_name: Path to output file.
    :param cluster_id_matrix: M-by-N-by-F numpy array of cluster IDs (all
        positive integers or -1).
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param field_names: length-F list of field names.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(cluster_id_matrix, num_dimensions=3)
    error_checking.assert_is_integer_numpy_array(cluster_id_matrix)
    assert numpy.all(numpy.logical_or(
        cluster_id_matrix == -1,
        cluster_id_matrix > 0
    ))

    num_rows = cluster_id_matrix.shape[0]
    num_columns = cluster_id_matrix.shape[1]
    num_fields = cluster_id_matrix.shape[2]

    error_checking.assert_is_numpy_array(
        grid_latitudes_deg_n,
        exact_dimensions=numpy.array([num_rows], dtype=int)
    )
    error_checking.assert_is_valid_lat_numpy_array(
        grid_latitudes_deg_n, allow_nan=False
    )

    error_checking.assert_is_numpy_array(
        grid_longitudes_deg_e,
        exact_dimensions=numpy.array([num_columns], dtype=int)
    )
    grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        grid_longitudes_deg_e, allow_nan=False
    )

    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(field_names),
        exact_dimensions=numpy.array([num_fields], dtype=int)
    )

    # Do actual stuff.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF4_CLASSIC'
    )

    num_field_chars = max([len(f) for f in field_names])

    dataset_object.createDimension(ROW_DIM, num_rows)
    dataset_object.createDimension(COLUMN_DIM, num_columns)
    dataset_object.createDimension(FIELD_DIM, num_fields)
    dataset_object.createDimension(FIELD_CHAR_DIM, num_field_chars)

    these_dim = (ROW_DIM, COLUMN_DIM, FIELD_DIM)
    dataset_object.createVariable(
        CLUSTER_ID_KEY, datatype=numpy.int32, dimensions=these_dim
    )
    dataset_object.variables[CLUSTER_ID_KEY][:] = cluster_id_matrix

    these_dim = (ROW_DIM,)
    dataset_object.createVariable(
        LATITUDE_KEY, datatype=numpy.float64, dimensions=these_dim
    )
    dataset_object.variables[LATITUDE_KEY][:] = grid_latitudes_deg_n

    these_dim = (COLUMN_DIM,)
    dataset_object.createVariable(
        LONGITUDE_KEY, datatype=numpy.float64, dimensions=these_dim
    )
    dataset_object.variables[LONGITUDE_KEY][:] = grid_longitudes_deg_e

    this_string_format = 'S{0:d}'.format(num_field_chars)
    field_names_char_array = netCDF4.stringtochar(numpy.array(
        field_names, dtype=this_string_format
    ))

    dataset_object.createVariable(
        FIELD_NAME_KEY, datatype='S1', dimensions=(FIELD_DIM, FIELD_CHAR_DIM)
    )
    dataset_object.variables[FIELD_NAME_KEY][:] = numpy.array(
        field_names_char_array
    )

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads predictions from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: cluster_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    prediction_table_xarray = xarray.open_dataset(netcdf_file_name)
    target_field_names = [
        f.decode('utf-8') for f in
        prediction_table_xarray[FIELD_NAME_KEY].values
    ]

    prediction_table_xarray = prediction_table_xarray.assign({
        FIELD_NAME_KEY: (
            prediction_table_xarray[FIELD_NAME_KEY].dims,
            target_field_names
        )
    })

    num_fields = len(target_field_names)
    field_indices = numpy.linspace(0, num_fields - 1, num=num_fields, dtype=int)
    return prediction_table_xarray.assign_coords({
        FIELD_DIM: field_indices
    })
