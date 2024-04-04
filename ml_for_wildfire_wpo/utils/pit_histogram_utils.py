"""Methods for computing PIT (probability integral transform) histogram."""

import numpy
import xarray
from scipy.stats import percentileofscore
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_wildfire_wpo.utils import regression_evaluation as regression_eval

TOLERANCE = 1e-6

LOW_PIT_THRESHOLD = 1. / 3
HIGH_PIT_THRESHOLD = 2. / 3

FIELD_DIM = 'field'
BIN_CENTER_DIM = 'bin_center'
BIN_EDGE_DIM = 'bin_edge'

PIT_DEVIATION_KEY = 'pit_deviation'
PERFECT_PIT_DEVIATION_KEY = 'perfect_pit_deviation'
LOW_BIN_BIAS_KEY = 'low_bin_pit_bias'
MIDDLE_BIN_BIAS_KEY = 'middle_bin_pit_bias'
HIGH_BIN_BIAS_KEY = 'high_bin_pit_bias'
EXAMPLE_COUNT_KEY = 'example_count'

MODEL_FILE_KEY = 'model_file_name'
PREDICTION_FILES_KEY = 'prediction_file_names'


def _get_low_mid_hi_bins(bin_edges):
    """Returns indices for low-PIT, medium-PIT, and high-PIT bins.

    B = number of bins

    :param bin_edges: length-(B + 1) numpy array of bin edges, sorted in
        ascending order.
    :return: low_bin_indices: 1-D numpy array with array indices for low-PIT
        bins.
    :return: middle_bin_indices: 1-D numpy array with array indices for
        medium-PIT bins.
    :return: high_bin_indices: 1-D numpy array with array indices for high-PIT
        bins.
    """

    num_bins = len(bin_edges) - 1

    these_diffs = bin_edges - LOW_PIT_THRESHOLD
    these_diffs[these_diffs > TOLERANCE] = numpy.inf
    max_index_for_low_bins = numpy.argmin(numpy.absolute(these_diffs)) - 1
    max_index_for_low_bins = max([max_index_for_low_bins, 0])

    low_bin_indices = numpy.linspace(
        0, max_index_for_low_bins, num=max_index_for_low_bins + 1, dtype=int
    )

    these_diffs = HIGH_PIT_THRESHOLD - bin_edges
    these_diffs[these_diffs > TOLERANCE] = numpy.inf
    min_index_for_high_bins = numpy.argmin(numpy.absolute(these_diffs))
    min_index_for_high_bins = min([min_index_for_high_bins, num_bins - 1])

    high_bin_indices = numpy.linspace(
        min_index_for_high_bins, num_bins - 1,
        num=num_bins - min_index_for_high_bins, dtype=int
    )

    middle_bin_indices = numpy.linspace(
        0, num_bins - 1, num=num_bins, dtype=int
    )
    middle_bin_indices = numpy.array(list(
        set(middle_bin_indices.tolist())
        - set(low_bin_indices.tolist())
        - set(high_bin_indices.tolist())
    ))

    return low_bin_indices, middle_bin_indices, high_bin_indices


def _compute_pit_histogram_1field(
        target_matrix, prediction_matrix, result_table_xarray, field_index):
    """Computes PIT histogram for one target field.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    S = number of ensemble members

    :param target_matrix: E-by-M-by-N numpy array of target values.
    :param prediction_matrix: E-by-M-by-N-by-S numpy array of predictions.
    :param result_table_xarray: Same as output from `run_discard_test`.  Results
        from this method will be stored in the table.
    :param field_index: Field index.  If `field_index == j`, this means we are
        working on the [j]th target field in the table.
    :return: result_table_xarray: Same as input, except populated with results
        for the given target field.
    """

    j = field_index

    # Compute PIT values.
    num_scalar_examples = numpy.prod(target_matrix.shape)
    ensemble_size = prediction_matrix.shape[-1]

    target_values_1d = numpy.ravel(target_matrix)
    prediction_matrix_2d = numpy.reshape(
        prediction_matrix, (num_scalar_examples, ensemble_size)
    )
    pit_values = numpy.full(num_scalar_examples, numpy.nan)

    for i in range(num_scalar_examples):
        if numpy.mod(i, 10000) == 0:
            print((
                'Have computed PIT value for {0:d} of {1:d} scalar examples...'
            ).format(
                i, num_scalar_examples
            ))

        pit_values[i] = 0.01 * percentileofscore(
            a=prediction_matrix_2d[i, :], score=target_values_1d[i], kind='mean'
        )

    print('Computed PIT value for all {0:d} scalar examples!'.format(
        num_scalar_examples
    ))

    # Compute histogram.
    num_bins = len(result_table_xarray.coords[BIN_CENTER_DIM].values)

    bin_edges = numpy.linspace(0, 1, num=num_bins + 1, dtype=float)
    example_to_bin = numpy.digitize(
        x=pit_values, bins=bin_edges, right=False
    ) - 1
    example_to_bin = numpy.maximum(example_to_bin, 0)
    example_to_bin = numpy.minimum(example_to_bin, num_bins - 1)

    used_bin_indices, used_bin_counts = numpy.unique(
        example_to_bin, return_counts=True
    )
    bin_counts = numpy.full(num_bins, 0, dtype=int)
    bin_counts[used_bin_indices] = used_bin_counts

    result_table_xarray[EXAMPLE_COUNT_KEY].values[j, :] = bin_counts

    bin_frequencies = bin_counts.astype(float) / num_scalar_examples
    perfect_bin_frequency = 1. / num_bins

    result_table_xarray[PIT_DEVIATION_KEY].values[j] = numpy.sqrt(numpy.mean(
        (bin_frequencies - perfect_bin_frequency) ** 2
    ))
    result_table_xarray[PERFECT_PIT_DEVIATION_KEY].values[j] = numpy.sqrt(
        (1. - perfect_bin_frequency) /
        (num_scalar_examples * num_bins)
    )

    low_bin_indices, middle_bin_indices, high_bin_indices = (
        _get_low_mid_hi_bins(bin_edges)
    )

    result_table_xarray[LOW_BIN_BIAS_KEY].values[j] = numpy.mean(
        bin_frequencies[low_bin_indices] - perfect_bin_frequency
    )
    result_table_xarray[MIDDLE_BIN_BIAS_KEY].values[j] = numpy.mean(
        bin_frequencies[middle_bin_indices] - perfect_bin_frequency
    )
    result_table_xarray[HIGH_BIN_BIAS_KEY].values[j] = numpy.mean(
        bin_frequencies[high_bin_indices] - perfect_bin_frequency
    )

    return result_table_xarray


def compute_pit_histograms(prediction_file_names, target_field_names, num_bins):
    """Computes the PIT histogram independently for each target field.

    T = number of target fields
    B = number of bins for PIT values

    :param prediction_file_names: 1-D list of paths to prediction files.  Each
        file will be read by `prediction_io.read_file`.
    :param target_field_names: length-T list of field names.
    :param num_bins: Number of bins (B in the above discussion).
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    # Check input args.
    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_string_list(target_field_names)
    error_checking.assert_is_integer(num_bins)
    error_checking.assert_is_geq(num_bins, 10)
    error_checking.assert_is_leq(num_bins, 1000)

    # Read the data.
    (
        target_matrix, prediction_matrix, weight_matrix, model_file_name
    ) = regression_eval.read_inputs(
        prediction_file_names=prediction_file_names,
        target_field_names=target_field_names
    )

    # TODO(thunderhoser): This is a HACK.  I should use the weight matrix to
    # actually weight the various scores.
    target_matrix[weight_matrix < 0.05] = 0.
    prediction_matrix[weight_matrix < 0.05] = 0.

    # Set up the output table.
    num_target_fields = len(target_field_names)
    these_dimensions = (num_target_fields,)
    these_dim_keys = (FIELD_DIM,)
    main_data_dict = {
        PIT_DEVIATION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        PERFECT_PIT_DEVIATION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        LOW_BIN_BIAS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MIDDLE_BIN_BIAS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        HIGH_BIN_BIAS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }

    these_dimensions = (num_target_fields, num_bins)
    these_dim_keys = (FIELD_DIM, BIN_CENTER_DIM)
    main_data_dict.update({
        EXAMPLE_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, -1, dtype=int)
        )
    })

    bin_edges = numpy.linspace(0, 1, num=num_bins + 1, dtype=float)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    metadata_dict = {
        FIELD_DIM: target_field_names,
        BIN_CENTER_DIM: bin_centers
    }

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    result_table_xarray.attrs[PREDICTION_FILES_KEY] = ' '.join([
        '{0:s}'.format(f) for f in prediction_file_names
    ])

    # Do actual stuff.
    for j in range(num_target_fields):
        result_table_xarray = _compute_pit_histogram_1field(
            target_matrix=target_matrix[..., j],
            prediction_matrix=prediction_matrix[..., j, :],
            result_table_xarray=result_table_xarray,
            field_index=j
        )

    return result_table_xarray


def write_results(result_table_xarray, netcdf_file_name):
    """Writes PIT histogram for each target variable to NetCDF file.

    :param result_table_xarray: xarray table in format returned by
        `get_results_all_vars`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_results(netcdf_file_name):
    """Reads PIT histogram for each target variable from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)
