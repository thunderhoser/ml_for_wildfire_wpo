"""Tests new generator."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gfs_utils
import era5_constant_utils
import canadian_fwi_utils
import neural_net


def _run():
    """Tests new generator.

    This is effectively the main method.
    """

    option_dict = {
        neural_net.INNER_LATITUDE_LIMITS_KEY: numpy.array([17, 73], dtype=float),
        neural_net.INNER_LONGITUDE_LIMITS_KEY: numpy.array([171, -65], dtype=float),
        neural_net.OUTER_LATITUDE_BUFFER_KEY: 5.,
        neural_net.OUTER_LONGITUDE_BUFFER_KEY: 5.,
        neural_net.INIT_DATE_LIMITS_KEY: ['20210104', '20210204'],
        neural_net.GFS_PREDICTOR_FIELDS_KEY: gfs_utils.ALL_FIELD_NAMES,
        neural_net.GFS_PRESSURE_LEVELS_KEY: numpy.array([500, 800], dtype=int),
        neural_net.GFS_PREDICTOR_LEADS_KEY: numpy.array([0, 6, 12, 18, 24], dtype=int),
        neural_net.GFS_DIRECTORY_KEY: '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/gfs_data/processed/merged',
        neural_net.ERA5_CONSTANT_PREDICTOR_FIELDS_KEY: era5_constant_utils.ALL_FIELD_NAMES,
        neural_net.ERA5_CONSTANT_FILE_KEY: '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/era5_constants.nc',
        neural_net.TARGET_FIELD_KEY: canadian_fwi_utils.FFMC_NAME,
        neural_net.TARGET_LEAD_TIME_KEY: 2,
        neural_net.TARGET_LAG_TIMES_KEY: numpy.array([1, 2, 3], dtype=int),
        neural_net.TARGET_CUTOFFS_KEY: None,
        neural_net.TARGET_DIRECTORY_KEY: '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/canadian_fwi',
        neural_net.BATCH_SIZE_KEY: 4,
        neural_net.SENTINEL_VALUE_KEY: -999.
    }

    generator_object = neural_net.data_generator(option_dict)
    for _ in range(3):
        next(generator_object)


if __name__ == '__main__':
    _run()