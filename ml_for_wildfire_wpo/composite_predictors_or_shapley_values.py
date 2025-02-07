"""Composites predictor fields and/or Shapley fields over many days.

This script has two modes:

#1 Compositing only predictor fields.  In this case the arguments
   "input_model_file_name" and "model_lead_time_days" are needed -- while
   "input_shapley_dir_name" is NOT needed.

#2 Compositing both predictor and Shapley fields.  In this case the arguments
   "input_model_file_name" and "model_lead_time_days" are NOT needed -- but
   "input_shapley_dir_name" is needed.
"""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prob_matched_means as pmm
import shapley_io
import composite_io
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SHAPLEY_KEYS = [
    shapley_io.SHAPLEY_FOR_3D_GFS_KEY, shapley_io.SHAPLEY_FOR_2D_GFS_KEY,
    shapley_io.SHAPLEY_FOR_ERA5_KEY, shapley_io.SHAPLEY_FOR_LAGLEAD_TARGETS_KEY,
    shapley_io.SHAPLEY_FOR_PREDN_BASELINE_KEY
]

GFS_DIRECTORY_ARG_NAME = 'input_gfs_directory_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
GFS_FCST_TARGET_DIR_ARG_NAME = 'input_gfs_fcst_target_dir_name'
INIT_DATES_ARG_NAME = 'init_date_strings'
MODEL_FILE_ARG_NAME = 'input_model_file_name'
MODEL_LEAD_TIME_ARG_NAME = 'model_lead_time_days'
SHAPLEY_DIR_ARG_NAME = 'input_shapley_dir_name'
USE_PMM_ARG_NAME = 'use_pmm'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

GFS_DIRECTORY_HELP_STRING = (
    'Path to directory with GFS data (predictors).  Files therein will be '
    'found by `gfs_io.find_file` and read by `gfs_io.read_file`.'
)
TARGET_DIR_HELP_STRING = (
    'Path to directory with target fields.  Files therein will be found by '
    '`canadian_fwo_io.find_file` and read by `canadian_fwo_io.read_file`.'
)
GFS_FCST_TARGET_DIR_HELP_STRING = (
    'Path to directory with raw-GFS-forecast target fields.  Files therein '
    'will be found by `gfs_daily_io.find_file` and read by '
    '`gfs_daily_io.read_file`.'
)
INIT_DATES_HELP_STRING = (
    'List of forecast-init days (format "yyyymmdd") over which to composite.'
)
MODEL_FILE_HELP_STRING = (
    'Path to trained model (will be read by `neural_net.read_model`).`  If you '
    'are compositing only predictor fields, this argument is needed.  If you '
    'are compositing both predictor fields and Shapley fields, leave this '
    'argument alone and use `{0:s}`.'
).format(
    SHAPLEY_DIR_ARG_NAME
)
MODEL_LEAD_TIME_HELP_STRING = (
    'Model lead time.  If you are compositing only predictor fields, this '
    'argument is needed.  If you are compositing both predictor fields and '
    'Shapley fields, leave this argument alone and use `{0:s}`.'
).format(
    SHAPLEY_DIR_ARG_NAME
)
SHAPLEY_DIR_HELP_STRING = (
    'Path to directory with Shapley fields (files therein will be found by '
    '`shapley_io.find_file` and read by `shapley_io.read_file`).  If you are '
    'compositing only predictor fields, leave this argument alone.'
)
USE_PMM_HELP_STRING = (
    'Boolean flag.  If 1 (0), the composite will use a probability-matched '
    '(arithmetic) mean.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by either '
    '`composite_io.write_file` (if this script is running in mode #1) or '
    '`shapley_io.write_composite_file` (if this script is running in mode #2).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + GFS_DIRECTORY_ARG_NAME, type=str, required=True,
    help=GFS_DIRECTORY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GFS_FCST_TARGET_DIR_ARG_NAME, type=str, required=True,
    help=GFS_FCST_TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_DATES_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=INIT_DATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=False, default='',
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_LEAD_TIME_ARG_NAME, type=int, required=False, default=-1,
    help=MODEL_LEAD_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHAPLEY_DIR_ARG_NAME, type=str, required=False, default='',
    help=SHAPLEY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_PMM_ARG_NAME, type=int, required=True,
    help=USE_PMM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _composite_one_matrix(data_matrix, use_pmm):
    """Composites one data matrix.

    E = number of examples (time steps)
    M = number of rows in spatial grid
    N = number of columns in spatial grid

    :param data_matrix: numpy array with data to composite.  The first axis must
        have length E; the second axis must have length M; and the third axis
        must have length N.
    :param use_pmm: Boolean flag.  If True (False), will take
        probability-matched (arithmetic) mean.
    :return: composite_data_matrix: numpy array with averaged data.  This array
        has the same shape as `data_matrix` but without the first axis.
    """

    orig_dimensions = data_matrix.shape
    assert len(orig_dimensions) >= 4
    data_matrix = numpy.reshape(data_matrix, data_matrix.shape[:3] + (-1,))

    if use_pmm:
        composite_data_matrix = pmm.run_pmm_many_variables(
            input_matrix=data_matrix, max_percentile_level=99.
        )
    else:
        composite_data_matrix = numpy.nanmean(data_matrix, axis=0)

    return numpy.reshape(composite_data_matrix, orig_dimensions[1:])


def _run(gfs_directory_name, target_dir_name, gfs_forecast_target_dir_name,
         init_date_strings, model_file_name, model_lead_time_days,
         shapley_dir_name, use_pmm, output_file_name):
    """Composites predictor fields and/or Shapley fields over many days.

    This is effectively the main method.

    :param gfs_directory_name: See documentation at top of this script.
    :param target_dir_name: Same.
    :param gfs_forecast_target_dir_name: Same.
    :param init_date_strings: Same.
    :param model_file_name: Same.
    :param model_lead_time_days: Same.
    :param shapley_dir_name: Same.
    :param use_pmm: Same.
    :param output_file_name: Same.
    """

    # TODO(thunderhoser): I should probably eventually worry about NaN's in PMM.

    # Read and composite Shapley fields, if necessary.
    include_shapley = shapley_dir_name != ''
    shapley_table_xarray = None

    if include_shapley:
        shapley_matrices = []
        model_file_name = ''
        model_lead_time_days = -1

        num_dates = len(init_date_strings)
        good_date_indices = []

        for i in range(num_dates):
            this_file_name = shapley_io.find_file(
                directory_name=shapley_dir_name,
                init_date_string=init_date_strings[i],
                raise_error_if_missing=False
            )

            if not os.path.isfile(this_file_name):
                continue

            good_date_indices.append(i)

            print('Reading data from: "{0:s}"...'.format(this_file_name))
            shapley_table_xarray = shapley_io.read_file(this_file_name)
            stx = shapley_table_xarray

            this_model_file_name = stx.attrs[shapley_io.MODEL_FILE_KEY]
            this_model_lead_time_days = stx.attrs[
                shapley_io.MODEL_LEAD_TIME_KEY
            ]

            if model_file_name == '':
                model_file_name = copy.deepcopy(this_model_file_name)
                model_lead_time_days = this_model_lead_time_days + 0

            assert model_file_name == this_model_file_name
            assert model_lead_time_days == this_model_lead_time_days

            these_shapley_matrices = [
                stx[k].values if k in stx.data_vars else None
                for k in SHAPLEY_KEYS
            ]
            these_shapley_matrices = [
                m for m in these_shapley_matrices if m is not None
            ]

            if len(shapley_matrices) == 0:
                for this_shapley_matrix in these_shapley_matrices:
                    these_dim = ((num_dates,) + this_shapley_matrix.shape)
                    shapley_matrices.append(
                        numpy.full(these_dim, numpy.nan)
                    )

            for j in range(len(these_shapley_matrices)):
                shapley_matrices[j][i, ...] = these_shapley_matrices[j] + 0.

        good_date_indices = numpy.array(good_date_indices, dtype=int)
        shapley_matrices = [m[good_date_indices, ...] for m in shapley_matrices]
        init_date_strings = [init_date_strings[k] for k in good_date_indices]
    else:
        shapley_matrices = None

    # Read model.
    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)
    # model_object = neural_net.read_model_for_shapley(model_file_name)

    # Read model metadata.
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )
    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    vod = model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    patch_size_deg = vod[neural_net.OUTER_PATCH_SIZE_DEG_KEY]
    if patch_size_deg is not None:
        raise ValueError('This script does not yet work for patch-trained NNs.')

    vod[neural_net.GFS_DIRECTORY_KEY] = gfs_directory_name
    vod[neural_net.TARGET_DIRECTORY_KEY] = target_dir_name
    vod[neural_net.GFS_FORECAST_TARGET_DIR_KEY] = gfs_forecast_target_dir_name
    vod[neural_net.USE_LEAD_TIME_AS_PRED_KEY] = False
    vod[neural_net.GFS_NORM_FILE_KEY] = None
    vod[neural_net.TARGET_NORM_FILE_KEY] = None
    vod[neural_net.ERA5_NORM_FILE_KEY] = None
    num_target_fields = len(vod[neural_net.TARGET_FIELDS_KEY])
    validation_option_dict = vod

    # Read predictor data to composite.
    predictor_matrices = []
    target_matrix = numpy.array([])

    num_dates = len(init_date_strings)
    good_date_indices = []
    data_dict = {}
    print(SEPARATOR_STRING)

    for i in range(num_dates):
        try:
            data_dict = neural_net.create_data(
                option_dict=validation_option_dict,
                init_date_string=init_date_strings[i],
                model_lead_time_days=model_lead_time_days
            )
            print(SEPARATOR_STRING)
        except:
            print(SEPARATOR_STRING)
            continue

        good_date_indices.append(i)

        these_predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]
        this_target_matrix = data_dict[neural_net.TARGETS_AND_WEIGHTS_KEY][
            ..., :num_target_fields
        ]

        if len(predictor_matrices) == 0:
            for this_predictor_matrix in these_predictor_matrices:
                these_dim = ((num_dates,) + this_predictor_matrix.shape[1:])
                predictor_matrices.append(
                    numpy.full(these_dim, numpy.nan)
                )

            these_dim = ((num_dates,) + this_target_matrix.shape[1:])
            target_matrix = numpy.full(these_dim, numpy.nan)

        for j in range(len(these_predictor_matrices)):
            predictor_matrices[j][i, ...] = (
                these_predictor_matrices[j][0, ...]
            )

        target_matrix[i, ...] = this_target_matrix[0, ...]

    good_date_indices = numpy.array(good_date_indices, dtype=int)
    predictor_matrices = [m[good_date_indices, ...] for m in predictor_matrices]
    init_date_strings = [init_date_strings[k] for k in good_date_indices]

    composite_predictor_matrices = [
        _composite_one_matrix(data_matrix=pm, use_pmm=use_pmm)
        for pm in predictor_matrices
    ]
    del predictor_matrices

    composite_target_matrix = _composite_one_matrix(
        data_matrix=target_matrix, use_pmm=use_pmm
    )
    del target_matrix

    if include_shapley:
        shapley_matrices = [m[good_date_indices, ...] for m in shapley_matrices]
        composite_shapley_matrices = [
            _composite_one_matrix(data_matrix=sm, use_pmm=use_pmm)
            for sm in shapley_matrices
        ]
        del shapley_matrices

        stx = shapley_table_xarray

        print('Writing results to: "{0:s}"...'.format(output_file_name))
        shapley_io.write_file(
            netcdf_file_name=output_file_name,
            shapley_matrices=composite_shapley_matrices,
            grid_latitudes_deg_n=data_dict[neural_net.GRID_LATITUDES_KEY],
            grid_longitudes_deg_e=data_dict[neural_net.GRID_LONGITUDES_KEY],
            composite_init_date_strings=init_date_strings,
            region_mask_file_name=stx.attrs[shapley_io.REGION_MASK_FILE_KEY],
            target_field_name=stx.attrs[shapley_io.TARGET_FIELD_KEY],
            model_input_layer_names=
            [l.name.split(':')[0] for l in model_object.input],
            model_lead_time_days=model_lead_time_days,
            model_file_name=model_file_name,
            predictor_matrices=composite_predictor_matrices,
            target_matrix=composite_target_matrix
        )

        return

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    composite_io.write_file(
        netcdf_file_name=output_file_name,
        grid_latitudes_deg_n=data_dict[neural_net.GRID_LATITUDES_KEY],
        grid_longitudes_deg_e=data_dict[neural_net.GRID_LONGITUDES_KEY],
        init_date_strings=init_date_strings,
        model_input_layer_names=
        [l.name.split(':')[0] for l in model_object.input],
        model_lead_time_days=model_lead_time_days,
        model_file_name=model_file_name,
        composite_predictor_matrices=composite_predictor_matrices,
        composite_target_matrix=composite_target_matrix
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        gfs_directory_name=getattr(INPUT_ARG_OBJECT, GFS_DIRECTORY_ARG_NAME),
        target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        gfs_forecast_target_dir_name=getattr(
            INPUT_ARG_OBJECT, GFS_FCST_TARGET_DIR_ARG_NAME
        ),
        init_date_strings=getattr(INPUT_ARG_OBJECT, INIT_DATES_ARG_NAME),
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        model_lead_time_days=getattr(
            INPUT_ARG_OBJECT, MODEL_LEAD_TIME_ARG_NAME
        ),
        shapley_dir_name=getattr(INPUT_ARG_OBJECT, SHAPLEY_DIR_ARG_NAME),
        use_pmm=bool(getattr(INPUT_ARG_OBJECT, USE_PMM_ARG_NAME)),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
