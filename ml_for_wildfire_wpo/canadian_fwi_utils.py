"""Helper methods for Canadian fire-weather indices."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking

LATITUDE_DIM = 'latitude_deg_n'
LONGITUDE_DIM = 'longitude_deg_e'
FIELD_DIM = 'field_name'
QUANTILE_LEVEL_DIM = 'quantile_level'

DATA_KEY = 'data'

MEAN_VALUE_KEY = 'mean_value'
MEAN_SQUARED_VALUE_KEY = 'mean_squared_value'
STDEV_KEY = 'standard_deviation'
QUANTILE_KEY = 'quantile'

FFMC_NAME = 'fine_fuel_moisture_code'
DMC_NAME = 'duff_moisture_code'
DC_NAME = 'drought_code'
ISI_NAME = 'initial_spread_index'
BUI_NAME = 'buildup_index'
FWI_NAME = 'fire_weather_index'
DSR_NAME = 'daily_severity_rating'

ALL_FIELD_NAMES = [
    FFMC_NAME, DMC_NAME, DC_NAME, ISI_NAME, BUI_NAME, FWI_NAME, DSR_NAME
]


def _natural_log(input_array):
    """Computes natural logarithm.

    :param input_array: numpy array.
    :return: logarithm_array: numpy array with the same shape as `input_array`.
    """

    return numpy.log(numpy.maximum(input_array, 1e-6))


def check_field_name(field_name):
    """Ensures validity of field name.

    :param field_name: Name of weather field.
    :raises: ValueError: if `field_name not in ALL_FIELD_NAMES`.
    """

    error_checking.assert_is_string(field_name)
    if field_name in ALL_FIELD_NAMES:
        return

    error_string = (
        'Field "{0:s}" is not in the list of recognized fields (below):\n{1:s}'
    ).format(
        field_name, str(ALL_FIELD_NAMES)
    )

    raise ValueError(error_string)


def get_field(fwi_table_xarray, field_name):
    """Extracts one field from xarray table.

    M = number of rows in grid
    N = number of columns in grid

    :param fwi_table_xarray: xarray table with Canadian FWI variables.
    :param field_name: Field name.
    :return: data_matrix: M-by-N numpy array of data values.
    """

    error_checking.assert_is_string(field_name)

    k = numpy.where(
        fwi_table_xarray.coords[FIELD_DIM].values == field_name
    )[0][0]

    return fwi_table_xarray[DATA_KEY].values[..., k]


def subset_by_row(fwi_table_xarray, desired_row_indices):
    """Subsets xarray table by grid row.

    :param fwi_table_xarray: xarray table with Canadian FWI variables.
    :param desired_row_indices: 1-D numpy array with indices of desired rows.
    :return: fwi_table_xarray: Same as input but maybe with fewer
        rows.
    """

    error_checking.assert_is_numpy_array(desired_row_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_row_indices)
    error_checking.assert_is_geq_numpy_array(desired_row_indices, 0)

    return fwi_table_xarray.isel({LATITUDE_DIM: desired_row_indices})


def subset_by_column(fwi_table_xarray, desired_column_indices):
    """Subsets xarray table by grid column.

    :param fwi_table_xarray: xarray table with Canadian FWI variables.
    :param desired_column_indices: 1-D numpy array with indices of desired
        columns.
    :return: fwi_table_xarray: Same as input but maybe with fewer
        columns.
    """

    error_checking.assert_is_numpy_array(
        desired_column_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)

    return fwi_table_xarray.isel(
        {LONGITUDE_DIM: desired_column_indices}
    )


def dmc_and_dc_to_bui(dmc_value_or_array, dc_value_or_array):
    """Converts duff moisture code (DMC) and drought code (DC) to BUI.

    BUI = build-up index

    :param dmc_value_or_array: DMC (either a single value or a numpy array).
    :param dc_value_or_array: DC (either a single value or a numpy array).
    :return: bui_value_or_array: BUI (same shape as both inputs).
    """

    found_arrays = isinstance(dmc_value_or_array, numpy.ndarray)
    if not found_arrays:
        dmc_value_or_array = numpy.array([dmc_value_or_array], dtype=float)
        dc_value_or_array = numpy.array([dc_value_or_array], dtype=float)

    error_checking.assert_is_numpy_array(
        dc_value_or_array,
        exact_dimensions=numpy.array(dmc_value_or_array.shape, dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(dmc_value_or_array, 0.)
    error_checking.assert_is_geq_numpy_array(dc_value_or_array, 0.)

    bui_value_or_array = numpy.full(dmc_value_or_array.shape, numpy.nan)
    dmc = dmc_value_or_array
    dc = dc_value_or_array

    idx = dmc_value_or_array <= 0.4 * dc_value_or_array
    bui_value_or_array[idx] = (0.8 * dmc[idx] * dc[idx]) / (dmc[idx] + 0.4 * dc[idx])

    idx = numpy.invert(idx)
    bui_value_or_array[idx] = dmc[idx] - (1. - 0.8 * dc[idx] / (dmc[idx] + 0.4 * dc[idx])) / (0.92 + numpy.power(0.0114 * dmc[idx], 1.7))
    bui_value_or_array[numpy.isnan(bui_value_or_array)] = 0.
    bui_value_or_array = numpy.maximum(bui_value_or_array, 0.)

    if found_arrays:
        return bui_value_or_array

    return bui_value_or_array[0]


def isi_and_bui_to_fwi(isi_value_or_array, bui_value_or_array):
    """Converts initial-spread index (ISI) and build-up index (BUI) to FWI.

    FWI = fire-weather index

    :param isi_value_or_array: ISI (either a single value or a numpy array).
    :param bui_value_or_array: BUI (either a single value or a numpy array).
    :return: fwi_value_or_array: FWI (same shape as both inputs).
    """

    found_arrays = isinstance(isi_value_or_array, numpy.ndarray)
    if not found_arrays:
        isi_value_or_array = numpy.array([isi_value_or_array], dtype=float)
        bui_value_or_array = numpy.array([bui_value_or_array], dtype=float)

    error_checking.assert_is_numpy_array(
        bui_value_or_array,
        exact_dimensions=numpy.array(isi_value_or_array.shape, dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(isi_value_or_array, 0.)
    error_checking.assert_is_geq_numpy_array(bui_value_or_array, 0.)

    dmfunc = numpy.full(isi_value_or_array.shape, numpy.nan)
    isi = isi_value_or_array
    bui = bui_value_or_array

    idx = bui_value_or_array <= 80.
    dmfunc[idx] = 0.626 * numpy.power(bui[idx], 0.809) + 2

    idx = numpy.invert(idx)
    dmfunc[idx] = 1000. / (25 + 108.64 * numpy.exp(-0.023 * bui[idx]))
    prelim_fwi = 0.1 * isi * dmfunc

    fwi_value_or_array = numpy.full(prelim_fwi.shape, numpy.nan)

    idx = prelim_fwi > 1.
    fwi_value_or_array[idx] = numpy.exp(
        2.72 *
        numpy.power(0.434 * _natural_log(prelim_fwi[idx]), 0.647)
    )

    idx = numpy.invert(idx)
    fwi_value_or_array[idx] = prelim_fwi[idx]
    fwi_value_or_array[numpy.isnan(fwi_value_or_array)] = 0.
    fwi_value_or_array = numpy.maximum(fwi_value_or_array, 0.)

    if found_arrays:
        return fwi_value_or_array

    return fwi_value_or_array[0]


def fwi_to_dsr(fwi_value_or_array):
    """Converts fire-weather index (FWI) to daily severity rating (DSR).

    :param fwi_value_or_array: FWI (either a single value or a numpy array).
    :return: dsr_value_or_array: DSR (same shape as the input).
    """

    found_arrays = isinstance(fwi_value_or_array, numpy.ndarray)
    if not found_arrays:
        fwi_value_or_array = numpy.array([fwi_value_or_array], dtype=float)

    error_checking.assert_is_geq_numpy_array(fwi_value_or_array, 0.)

    dsr_value_or_array = 0.0272 * numpy.power(fwi_value_or_array, 1.77)
    if found_arrays:
        return dsr_value_or_array

    return dsr_value_or_array[0]
