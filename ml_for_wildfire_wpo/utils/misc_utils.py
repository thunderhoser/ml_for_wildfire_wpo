"""Miscellaneous helper methods."""

import numpy
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6


def desired_latitudes_to_rows(
        grid_latitudes_deg_n, start_latitude_deg_n, end_latitude_deg_n):
    """Converts desired latitudes to desired grid rows.

    :param grid_latitudes_deg_n: 1-D numpy array of latitudes (deg north) in
        full grid.
    :param start_latitude_deg_n: Latitude at start of desired range (deg north).
    :param end_latitude_deg_n: Latitude at end of desired range (deg north).
    :return: desired_row_indices: 1-D numpy array with indices of desired rows.
    """

    error_checking.assert_is_numpy_array(
        grid_latitudes_deg_n, num_dimensions=1
    )
    error_checking.assert_is_valid_lat_numpy_array(
        grid_latitudes_deg_n, allow_nan=False
    )

    meridional_grid_spacings_deg = numpy.diff(grid_latitudes_deg_n)
    error_checking.assert_is_greater_numpy_array(
        meridional_grid_spacings_deg, 0
    )

    min_meridional_spacing_deg = numpy.min(meridional_grid_spacings_deg)
    max_meridional_spacing_deg = numpy.max(meridional_grid_spacings_deg)
    error_checking.assert_is_less_than(
        max_meridional_spacing_deg - min_meridional_spacing_deg, TOLERANCE
    )

    meridional_grid_spacing_deg = numpy.mean(meridional_grid_spacings_deg)
    start_latitude_deg_n = number_rounding.floor_to_nearest(
        start_latitude_deg_n, meridional_grid_spacing_deg
    )
    end_latitude_deg_n = number_rounding.ceiling_to_nearest(
        end_latitude_deg_n, meridional_grid_spacing_deg
    )
    error_checking.assert_is_greater(
        numpy.absolute(start_latitude_deg_n - end_latitude_deg_n),
        TOLERANCE
    )

    error_checking.assert_is_valid_latitude(start_latitude_deg_n)
    error_checking.assert_is_valid_latitude(end_latitude_deg_n)

    num_latitudes = 1 + int(numpy.absolute(numpy.round(
        (end_latitude_deg_n - start_latitude_deg_n) /
        meridional_grid_spacing_deg
    )))
    desired_latitudes_deg_n = numpy.linspace(
        start_latitude_deg_n, end_latitude_deg_n, num=num_latitudes
    )
    desired_row_indices = numpy.array([
        numpy.where(numpy.absolute(grid_latitudes_deg_n - d) <= TOLERANCE)[0][0]
        for d in desired_latitudes_deg_n
    ], dtype=int)

    return desired_row_indices


def desired_longitudes_to_columns(
        grid_longitudes_deg_e, start_longitude_deg_e, end_longitude_deg_e):
    """Converts desired longitudes to desired grid columns.

    :param grid_longitudes_deg_e: 1-D numpy array of longitudes (deg east) in
        full grid.  This may be in either format (positive or negative values in
        western hemisphere).
    :param start_longitude_deg_e: Longitude at start of desired range.  This may
        be in either format.
    :param end_longitude_deg_e: Longitude at end of desired range.  This may
        be in either format.
    :return: desired_column_indices: 1-D numpy array with indices of desired
        columns.
    """

    error_checking.assert_is_numpy_array(
        grid_longitudes_deg_e, num_dimensions=1
    )
    grid_longitudes_positive_in_west_deg_e = (
        lng_conversion.convert_lng_positive_in_west(
            grid_longitudes_deg_e, allow_nan=False
        )
    )
    grid_longitudes_negative_in_west_deg_e = (
        lng_conversion.convert_lng_negative_in_west(
            grid_longitudes_deg_e, allow_nan=False
        )
    )

    try:
        zonal_grid_spacings_deg = numpy.diff(
            grid_longitudes_positive_in_west_deg_e
        )
        error_checking.assert_is_greater_numpy_array(zonal_grid_spacings_deg, 0)

        min_zonal_spacing_deg = numpy.min(zonal_grid_spacings_deg)
        max_zonal_spacing_deg = numpy.max(zonal_grid_spacings_deg)
        error_checking.assert_is_less_than(
            max_zonal_spacing_deg - min_zonal_spacing_deg, TOLERANCE
        )
    except:
        zonal_grid_spacings_deg = numpy.diff(
            grid_longitudes_negative_in_west_deg_e
        )
        error_checking.assert_is_greater_numpy_array(zonal_grid_spacings_deg, 0)

        min_zonal_spacing_deg = numpy.min(zonal_grid_spacings_deg)
        max_zonal_spacing_deg = numpy.max(zonal_grid_spacings_deg)
        error_checking.assert_is_less_than(
            max_zonal_spacing_deg - min_zonal_spacing_deg, TOLERANCE
        )

    zonal_grid_spacing_deg = numpy.mean(zonal_grid_spacings_deg)
    start_longitude_deg_e = number_rounding.floor_to_nearest(
        start_longitude_deg_e, zonal_grid_spacing_deg
    )
    end_longitude_deg_e = number_rounding.ceiling_to_nearest(
        end_longitude_deg_e, zonal_grid_spacing_deg
    )
    error_checking.assert_is_greater(
        numpy.absolute(start_longitude_deg_e - end_longitude_deg_e),
        TOLERANCE
    )

    start_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        start_longitude_deg_e, allow_nan=False
    )
    end_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        end_longitude_deg_e, allow_nan=False
    )

    if end_longitude_deg_e > start_longitude_deg_e:
        are_longitudes_positive_in_west = True
    else:
        start_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            start_longitude_deg_e, allow_nan=False
        )
        end_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            end_longitude_deg_e, allow_nan=False
        )
        are_longitudes_positive_in_west = False

    if are_longitudes_positive_in_west:
        grid_longitudes_deg_e = grid_longitudes_positive_in_west_deg_e
    else:
        grid_longitudes_deg_e = grid_longitudes_negative_in_west_deg_e

    num_longitudes = 1 + int(numpy.absolute(numpy.round(
        (end_longitude_deg_e - start_longitude_deg_e) / zonal_grid_spacing_deg
    )))

    if end_longitude_deg_e > start_longitude_deg_e:
        desired_longitudes_deg_e = numpy.linspace(
            start_longitude_deg_e, end_longitude_deg_e, num=num_longitudes
        )

        desired_column_indices = numpy.array([
            numpy.where(
                numpy.absolute(grid_longitudes_deg_e - d) <= TOLERANCE
            )[0][0]
            for d in desired_longitudes_deg_e
        ], dtype=int)
    else:
        undesired_longitudes_deg_e = numpy.linspace(
            end_longitude_deg_e, start_longitude_deg_e, num=num_longitudes
        )[1:-1]

        undesired_column_indices = numpy.array([
            numpy.where(
                numpy.absolute(grid_longitudes_deg_e - d) <= TOLERANCE
            )[0][0]
            for d in undesired_longitudes_deg_e
        ], dtype=int)

        all_column_indices = numpy.linspace(
            0, len(grid_longitudes_deg_e) - 1, num=len(grid_longitudes_deg_e),
            dtype=int
        )
        desired_column_indices = (
            set(all_column_indices.tolist()) -
            set(undesired_column_indices.tolist())
        )
        desired_column_indices = numpy.array(
            list(desired_column_indices), dtype=int
        )

        break_index = 1 + numpy.where(
            numpy.diff(desired_column_indices) > 1
        )[0][0]

        desired_column_indices = numpy.concatenate((
            desired_column_indices[break_index:],
            desired_column_indices[:break_index]
        ))

    return desired_column_indices


def split_array_by_nan(input_array):
    """Splits numpy array into list of contiguous subarrays without NaN.

    :param input_array: 1-D numpy array.
    :return: list_of_arrays: 1-D list of 1-D numpy arrays.  Each numpy array is
        without NaN.
    """

    error_checking.assert_is_real_numpy_array(input_array)
    error_checking.assert_is_numpy_array(input_array, num_dimensions=1)

    return [
        input_array[i] for i in
        numpy.ma.clump_unmasked(numpy.ma.masked_invalid(input_array))
    ]
