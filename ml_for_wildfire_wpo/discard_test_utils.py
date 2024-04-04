"""Methods for conducting discard test."""

import os
import sys
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import regression_evaluation as regression_eval

FIELD_DIM = 'field'
DISCARD_FRACTION_DIM = 'discard_fraction'

MONO_FRACTION_KEY = 'monotonicity_fraction'
DISCARD_IMPROVEMENT_KEY = 'discard_improvement'

DETERMINISTIC_ERROR_KEY = 'deterministic_error'
MEAN_DETERMINISTIC_PRED_KEY = 'mean_deterministic_prediction'
MEAN_TARGET_KEY = 'mean_target_value'
MEAN_UNCERTAINTY_KEY = 'mean_uncertainty'

MODEL_FILE_KEY = 'model_file_name'
PREDICTION_FILES_KEY = 'prediction_file_names'

ERROR_FUNCTION_KEY = 'error_function_name'
UNCERTAINTY_FUNCTION_KEY = 'uncertainty_function_name'


def _run_discard_test_1field(
        target_matrix, prediction_matrix, result_table_xarray, field_index,
        discard_fractions, uncertainty_function, error_function,
        is_error_pos_oriented):
    """Runs discard test for one target field.

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
    :param discard_fractions: See doc for `run_discard_test`.
    :param uncertainty_function: Same.
    :param error_function: Same.
    :param is_error_pos_oriented: Same.
    :return: result_table_xarray: Same as input, except populated with results
        for the given target field.
    """

    j = field_index
    num_fractions = len(discard_fractions)
    rtx = result_table_xarray

    print(prediction_matrix[..., j, :].shape)
    uncertainty_matrix = uncertainty_function(prediction_matrix[..., j, :])
    print(uncertainty_matrix.shape)
    print(numpy.max(uncertainty_matrix))
    deterministic_pred_matrix = numpy.mean(prediction_matrix, axis=-1)
    mask_matrix = numpy.full(uncertainty_matrix.shape, True, dtype=bool)

    for k in range(num_fractions):
        this_percentile_level = 100 * (1 - discard_fractions[k])
        this_inverted_mask_matrix = (
            uncertainty_matrix >
            numpy.percentile(uncertainty_matrix, this_percentile_level)
        )
        mask_matrix[this_inverted_mask_matrix == True] = False
        print(numpy.mean(mask_matrix))

        rtx[MEAN_UNCERTAINTY_KEY].values[j, k] = numpy.mean(
            uncertainty_matrix[mask_matrix == True]
        )
        rtx[MEAN_DETERMINISTIC_PRED_KEY].values[j, k] = numpy.mean(
            deterministic_pred_matrix[mask_matrix == True]
        )
        rtx[MEAN_TARGET_KEY].values[j, k] = numpy.mean(
            target_matrix[mask_matrix == True]
        )
        rtx[DETERMINISTIC_ERROR_KEY].values[j, k] = error_function(
            target_matrix, deterministic_pred_matrix, mask_matrix
        )

    error_values = rtx[DETERMINISTIC_ERROR_KEY].values[j]

    if is_error_pos_oriented:
        rtx[MONO_FRACTION_KEY].values[j] = numpy.mean(
            numpy.diff(error_values) > 0
        )
        rtx[DISCARD_IMPROVEMENT_KEY].values[j] = numpy.mean(
            numpy.diff(error_values)
        )
    else:
        rtx[MONO_FRACTION_KEY].values[j] = numpy.mean(
            numpy.diff(error_values) < 0
        )
        rtx[DISCARD_IMPROVEMENT_KEY].values[j] = numpy.mean(
            -1 * numpy.diff(error_values)
        )

    result_table_xarray = rtx
    return result_table_xarray


def get_stdev_uncertainty_func_1field():
    """Creates standard-deviation-based uncertainty function for one field.

    For use as input to `run_discard_test`.

    :return: uncertainty_function: Function handle.
    """

    def uncertainty_function(prediction_matrix):
        """Computes stdev of prediction distribution for each example/pixel.

        E = number of examples
        M = number of rows in grid
        N = number of columns in grid
        S = number of ensemble members

        :param prediction_matrix: E-by-M-by-N-by-S numpy array of predictions.
        :return: stdev_matrix: E-by-M-by-N numpy array of standard deviations.
        """

        return numpy.std(prediction_matrix, axis=-1, ddof=1)

    return uncertainty_function


def get_rmse_error_func_1field():
    """Creates RMSE-based error function for one field.

    For use as input to `run_discard_test`.

    :return: error_function: Function handle.
    """

    def error_function(target_matrix, deterministic_pred_matrix, mask_matrix):
        """Computes RMSE of deterministic prediction for each example/pixel.

        E = number of examples
        M = number of rows in grid
        N = number of columns in grid

        :param target_matrix: E-by-M-by-N numpy array of target values.
        :param deterministic_pred_matrix: E-by-M-by-N numpy array of
            deterministic predictions.
        :param mask_matrix: E-by-M-by-N numpy array of Boolean flags, with True
            indicating points to use in the evaluation and False indicating
            points to ignore (because they have already been discarded).
        :return: rmse_matrix: E-by-M-by-N numpy array of RMSE values.
        """

        squared_errors = (
            target_matrix[mask_matrix == True] -
            deterministic_pred_matrix[mask_matrix == True]
        ) ** 2
        return numpy.sqrt(numpy.mean(squared_errors))

    return error_function


def run_discard_test(
        prediction_file_names, target_field_names, discard_fractions,
        error_function, error_function_string,
        uncertainty_function, uncertainty_function_string,
        is_error_pos_oriented):
    """Runs the discard test independently for each target field.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields
    S = number of ensemble members
    F = number of discard fractions

    :param prediction_file_names: 1-D list of paths to prediction files.  Each
        file will be read by `prediction_io.read_file`.
    :param target_field_names: length-T list of field names.
    :param discard_fractions: length-(F - 1) numpy array of discard fractions,
        ranging from (0, 1).  This method will use 0 as the lowest discard
        fraction.

    :param error_function: Function with the following inputs and outputs...
    Input: target_matrix: E-by-M-by-N numpy array of target values.
    Input: deterministic_pred_matrix: E-by-M-by-N numpy array of deterministic
        predictions.
    Input: mask_matrix: E-by-M-by-N numpy array of Boolean flags, with True
        indicating points to use in the evaluation and False indicating points
        to ignore (because they have already been discarded).
    Output: error_value: Scalar value of error metric.

    :param error_function_string: String description of error function (used for
        metadata).

    :param uncertainty_function: Function with the following inputs and
        outputs...
    Input: prediction_matrix: E-by-M-by-N-by-S numpy array of predictions.
    Output: uncertainty_matrix: E-by-M-by-N numpy array with values of
        uncertainty metric.  The metric must be oriented so that higher value =
        more uncertainty.

    :param uncertainty_function_string: String description of uncertainty
        function (used for metadata).
    :param is_error_pos_oriented: Boolean flag.  If True (False), error function
        is positively (negatively) oriented.
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    # Check input args.
    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_string_list(target_field_names)
    error_checking.assert_is_boolean(is_error_pos_oriented)
    error_checking.assert_is_string(error_function_string)
    error_checking.assert_is_string(uncertainty_function_string)

    error_checking.assert_is_numpy_array(discard_fractions, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(discard_fractions, 0.)
    error_checking.assert_is_less_than_numpy_array(discard_fractions, 1.)

    discard_fractions = numpy.concatenate((
        numpy.array([0.]),
        discard_fractions
    ))

    num_fractions = len(discard_fractions)
    assert num_fractions >= 2

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
        MONO_FRACTION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        DISCARD_IMPROVEMENT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }

    these_dimensions = (num_target_fields, num_fractions)
    these_dim_keys = (FIELD_DIM, DISCARD_FRACTION_DIM)

    main_data_dict.update({
        DETERMINISTIC_ERROR_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MEAN_DETERMINISTIC_PRED_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MEAN_TARGET_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MEAN_UNCERTAINTY_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
    })

    metadata_dict = {
        FIELD_DIM: target_field_names,
        DISCARD_FRACTION_DIM: discard_fractions
    }

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    result_table_xarray.attrs[PREDICTION_FILES_KEY] = ' '.join([
        '{0:s}'.format(f) for f in prediction_file_names
    ])
    result_table_xarray.attrs[ERROR_FUNCTION_KEY] = error_function_string
    result_table_xarray.attrs[UNCERTAINTY_FUNCTION_KEY] = (
        uncertainty_function_string
    )

    # Do actual stuff.
    for j in range(num_target_fields):
        result_table_xarray = _run_discard_test_1field(
            target_matrix=target_matrix[..., j],
            prediction_matrix=prediction_matrix[..., j, :],
            result_table_xarray=result_table_xarray,
            field_index=j,
            discard_fractions=discard_fractions,
            uncertainty_function=uncertainty_function,
            error_function=error_function,
            is_error_pos_oriented=is_error_pos_oriented
        )

    return result_table_xarray


def write_results(result_table_xarray, netcdf_file_name):
    """Writes discard-test results to NetCDF file.

    :param result_table_xarray: xarray table in format returned by
        `run_discard_test`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_results(netcdf_file_name):
    """Reads discard-test results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)
