"""Finds limits of each subdomain in GFS grid."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from ml_for_wildfire_wpo.io import border_io
from ml_for_wildfire_wpo.io import raw_gfs_io
from ml_for_wildfire_wpo.io import region_mask_io
from ml_for_wildfire_wpo.io import era5_constant_io
from ml_for_wildfire_wpo.utils import era5_constant_utils
from ml_for_wildfire_wpo.plotting import gfs_plotting
from ml_for_wildfire_wpo.plotting import plotting_utils
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
ERA5_CONSTANT_FILE_NAME = (
    '/home/ralager/ml_for_wildfire_wpo_stuff/processed_era5_constants/'
    'era5_constants.nc'
)
OUTPUT_DIR_NAME = '/home/ralager/ml_for_wildfire_wpo_stuff/region_masks'

FIGURE_RESOLUTION_DPI = 300


def _plot_subdomains(
        latitude_limit_matrix_deg_n, longitude_limit_matrix_deg_e,
        output_file_name):
    """Plots subdomains.

    D = number of subdomains

    :param latitude_limit_matrix_deg_n: D-by-2 numpy array of latitude limits
        (deg north).
    :param longitude_limit_matrix_deg_e: D-by-2 numpy array of longitude limits
        (deg east).
    :param output_file_name: Path to output file.
    """

    print('Reading data from: "{0:s}"...'.format(ERA5_CONSTANT_FILE_NAME))
    era5_constant_table_xarray = era5_constant_io.read_file(
        ERA5_CONSTANT_FILE_NAME
    )
    geopotential_matrix_m2_s02 = era5_constant_utils.get_field(
        era5_constant_table_xarray=era5_constant_table_xarray,
        field_name=era5_constant_utils.GEOPOTENTIAL_NAME
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    colour_norm_object = pyplot.Normalize(
        vmin=numpy.min(geopotential_matrix_m2_s02),
        vmax=numpy.max(geopotential_matrix_m2_s02)
    )

    era5ct = era5_constant_table_xarray
    grid_latitudes_deg_n = (
        era5ct.coords[era5_constant_utils.LATITUDE_DIM].values
    )
    grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        era5ct.coords[era5_constant_utils.LONGITUDE_DIM].values
    )

    is_longitude_positive_in_west = gfs_plotting.plot_field(
        data_matrix=geopotential_matrix_m2_s02,
        grid_latitudes_deg_n=
        era5ct.coords[era5_constant_utils.LATITUDE_DIM].values,
        grid_longitudes_deg_e=lng_conversion.convert_lng_positive_in_west(
            era5ct.coords[era5_constant_utils.LONGITUDE_DIM].values
        ),
        colour_map_object=pyplot.get_cmap('viridis'),
        colour_norm_object=colour_norm_object,
        axes_object=axes_object,
        plot_colour_bar=True,
        plot_in_log2_scale=False
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    if is_longitude_positive_in_west:
        border_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            border_longitudes_deg_e
        )
        grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            grid_longitudes_deg_e
        )
        longitude_limit_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
            longitude_limit_matrix_deg_e
        )
    else:
        border_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            border_longitudes_deg_e
        )
        grid_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            grid_longitudes_deg_e
        )
        longitude_limit_matrix_deg_e = lng_conversion.convert_lng_negative_in_west(
            longitude_limit_matrix_deg_e
        )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object,
        line_colour=numpy.full(3, 0.)
    )
    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=grid_latitudes_deg_n,
        plot_longitudes_deg_e=grid_longitudes_deg_e,
        axes_object=axes_object,
        meridian_spacing_deg=20.,
        parallel_spacing_deg=10.
    )

    num_subdomains = latitude_limit_matrix_deg_n.shape[0]
    for i in range(num_subdomains):
        x_indices = numpy.array([0, 1, 1, 0, 0], dtype=int)
        y_indices = numpy.array([0, 0, 1, 1, 0], dtype=int)
        axes_object.plot(
            longitude_limit_matrix_deg_e[i, x_indices],
            latitude_limit_matrix_deg_n[i, y_indices],
            color='r',
            linewidth=4
        )

    axes_object.set_xlim(
        numpy.min(grid_longitudes_deg_e),
        numpy.max(grid_longitudes_deg_e)
    )
    axes_object.set_ylim(
        numpy.min(grid_latitudes_deg_n),
        numpy.max(grid_latitudes_deg_n)
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run():
    """Finds limits of each subdomain in GFS grid.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    gfs_latitude_matrix_deg_n, gfs_longitude_matrix_deg_e = (
        grids.latlng_vectors_to_matrices(
            unique_latitudes_deg=raw_gfs_io.GRID_LATITUDES_DEG_N,
            unique_longitudes_deg=
            raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
        )
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=42., end_latitude_deg_n=49.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=-125., end_longitude_deg_e=-111.
    )
    print((
        'Subdomain limits for Pacific northwest = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/pacific_northwest_mask.nc'.format(OUTPUT_DIR_NAME)
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=36., end_latitude_deg_n=42.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=-125., end_longitude_deg_e=-111.
    )
    print((
        'Subdomain limits for Pacific central-west = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/pacific_central_west_mask.nc'.format(
        OUTPUT_DIR_NAME
    )
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=30., end_latitude_deg_n=36.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=-125., end_longitude_deg_e=-111.
    )
    print((
        'Subdomain limits for Pacific southwest = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/pacific_southwest_mask.nc'.format(OUTPUT_DIR_NAME)
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=42., end_latitude_deg_n=49.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=-111., end_longitude_deg_e=-104.
    )
    print((
        'Subdomain limits for mountain northwest = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/mountain_northwest_mask.nc'.format(
        OUTPUT_DIR_NAME
    )
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=36., end_latitude_deg_n=42.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=-111., end_longitude_deg_e=-103.
    )
    print((
        'Subdomain limits for mountain central-west = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/mountain_central_west_mask.nc'.format(
        OUTPUT_DIR_NAME
    )
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=30., end_latitude_deg_n=36.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=-111., end_longitude_deg_e=-103.
    )
    print((
        'Subdomain limits for mountain southwest = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/mountain_southwest_mask.nc'.format(
        OUTPUT_DIR_NAME
    )
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=42., end_latitude_deg_n=49.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=-104., end_longitude_deg_e=-80.
    )
    print((
        'Subdomain limits for northern plains = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/northern_plains_mask.nc'.format(OUTPUT_DIR_NAME)
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=36., end_latitude_deg_n=42.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=-103., end_longitude_deg_e=-87.
    )
    print((
        'Subdomain limits for central plains = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/central_plains_mask.nc'.format(OUTPUT_DIR_NAME)
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=25., end_latitude_deg_n=36.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=-103., end_longitude_deg_e=-87.
    )
    print((
        'Subdomain limits for southern plains = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/southern_plains_mask.nc'.format(OUTPUT_DIR_NAME)
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=42., end_latitude_deg_n=49.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=-80., end_longitude_deg_e=-66.
    )
    print((
        'Subdomain limits for northeast = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/northeast_mask.nc'.format(OUTPUT_DIR_NAME)
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=36., end_latitude_deg_n=42.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=-87., end_longitude_deg_e=-69.
    )
    print((
        'Subdomain limits for central-east = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/central_east_mask.nc'.format(OUTPUT_DIR_NAME)
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=24., end_latitude_deg_n=36.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=-87., end_longitude_deg_e=-75.
    )
    print((
        'Subdomain limits for southeast = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/southeast_mask.nc'.format(OUTPUT_DIR_NAME)
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=51., end_latitude_deg_n=72.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=172., end_longitude_deg_e=-129.
    )
    print((
        'Subdomain limits for Alaska = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/alaska_mask.nc'.format(OUTPUT_DIR_NAME)
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=51., end_latitude_deg_n=72.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=-168.75, end_longitude_deg_e=-129.
    )
    print((
        'Subdomain limits for small Alaska = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/small_alaska_mask.nc'.format(OUTPUT_DIR_NAME)
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    row_indices = raw_gfs_io.desired_latitudes_to_rows(
        start_latitude_deg_n=18., end_latitude_deg_n=29.
    )
    column_indices = raw_gfs_io.desired_longitudes_to_columns(
        start_longitude_deg_e=-179., end_longitude_deg_e=-154.
    )
    print((
        'Subdomain limits for Hawaii = {0:.2f} to {1:.2f} deg N, '
        '{2:.2f} to {3:.2f} deg E, {4:d} rows x {5:d} columns'
    ).format(
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[0]],
        raw_gfs_io.GRID_LATITUDES_DEG_N[row_indices[-1]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[0]],
        raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E[column_indices[-1]],
        len(row_indices),
        len(column_indices)
    ))

    mask_matrix = numpy.full(gfs_latitude_matrix_deg_n.shape, 0, dtype=bool)
    mask_matrix[numpy.ix_(row_indices, column_indices)] = True

    output_file_name = '{0:s}/hawaii_mask.nc'.format(OUTPUT_DIR_NAME)
    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    region_mask_io.write_file(
        netcdf_file_name=output_file_name,
        mask_matrix=mask_matrix,
        grid_latitudes_deg_n=raw_gfs_io.GRID_LATITUDES_DEG_N,
        grid_longitudes_deg_e=raw_gfs_io.GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )

    latitude_limit_matrix_deg_n = numpy.array([
        [42, 49],
        [36, 42],
        [30, 36],
        [42, 49],
        [36, 42],
        [30, 36],
        [42, 49],
        [36, 42],
        [25, 36],
        [42, 49],
        [36, 42],
        [24, 36],
        [51, 72],
        [18, 29]
    ], dtype=float)

    longitude_limit_matrix_deg_e = numpy.array([
        [-125, -111],
        [-125, -111],
        [-125, -111],
        [-111, -104],
        [-111, -103],
        [-111, -103],
        [-104, -80],
        [-103, -87],
        [-103, -87],
        [-80, -66],
        [-87, -69],
        [-87, -75],
        [172, -129],
        [-179, -154]
    ], dtype=float)

    _plot_subdomains(
        latitude_limit_matrix_deg_n=latitude_limit_matrix_deg_n,
        longitude_limit_matrix_deg_e=longitude_limit_matrix_deg_e,
        output_file_name=(
            '/home/ralager/ml_for_wildfire_wpo_stuff/processed_era5_constants/'
            'subdomains.jpg'
        )
    )


if __name__ == '__main__':
    _run()
