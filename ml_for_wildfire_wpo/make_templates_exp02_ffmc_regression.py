"""Makes templates for Experiment 2 (FFMC regression)."""

import os
import sys
import copy
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import custom_losses
import neural_net
import chiu_net_architecture as chiu_net_arch
import architecture_utils
import file_system_utils

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_models/'
    'experiment02_ffmc_regression/templates'
)

LAYERS_PER_BLOCK_COUNTS_AXIS1 = numpy.array([1, 2], dtype=int)
FIRST_LAYER_FILTER_COUNTS_AXIS2 = numpy.array([10, 15, 20, 25, 30], dtype=int)
GFS_LEAD_TIME_COUNTS_AXIS3 = numpy.array([2, 3, 4, 5, 6, 7, 9, 11], dtype=int)

DEFAULT_OPTION_DICT = {
    # chiu_net_arch.GFS_3D_DIMENSIONS_KEY:
    #     numpy.array([265, 537, 3, 5, 5], dtype=int),
    # chiu_net_arch.GFS_2D_DIMENSIONS_KEY:
    #     numpy.array([265, 537, 5, 7], dtype=int),
    chiu_net_arch.ERA5_CONST_DIMENSIONS_KEY: None,
    chiu_net_arch.LAGTGT_DIMENSIONS_KEY:
        numpy.array([265, 537, 6, 1], dtype=int),
    chiu_net_arch.GFS_FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    chiu_net_arch.GFS_FC_MODULE_DROPOUT_RATES_KEY:
        numpy.array([0.]),
    chiu_net_arch.GFS_FC_MODULE_USE_3D_CONV: True,
    chiu_net_arch.LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    chiu_net_arch.LAGTGT_FC_MODULE_DROPOUT_RATES_KEY:
        numpy.array([0.]),
    chiu_net_arch.LAGTGT_FC_MODULE_USE_3D_CONV: True,
    chiu_net_arch.NUM_LEVELS_KEY: 6,
    # chiu_net_arch.GFS_ENCODER_NUM_CONV_LAYERS_KEY:
    #     numpy.full(7, 2, dtype=int),
    # chiu_net_arch.GFS_ENCODER_NUM_CHANNELS_KEY:
    #     numpy.array([16, 20, 24, 28, 32, 36, 40], dtype=int),
    chiu_net_arch.GFS_ENCODER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    chiu_net_arch.LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY:
        numpy.full(7, 2, dtype=int),
    # chiu_net_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY:
    #     numpy.array([4, 6, 8, 10, 12, 14, 16], dtype=int),
    chiu_net_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY:
        numpy.full(7, 0.),
    chiu_net_arch.DECODER_NUM_CONV_LAYERS_KEY:
        numpy.full(6, 2, dtype=int),
    # chiu_net_arch.DECODER_NUM_CHANNELS_KEY:
    #     numpy.array([20, 26, 32, 38, 44, 50], dtype=int),
    chiu_net_arch.UPSAMPLING_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    chiu_net_arch.SKIP_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    chiu_net_arch.INCLUDE_PENULTIMATE_KEY: False,
    chiu_net_arch.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    chiu_net_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    chiu_net_arch.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    chiu_net_arch.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    chiu_net_arch.L1_WEIGHT_KEY: 0.,
    chiu_net_arch.L2_WEIGHT_KEY: 1e-6,
    chiu_net_arch.USE_BATCH_NORM_KEY: True
}


def _run():
    """Makes templates for Experiment 2 (FFMC regression).

    This is effectively the main method.
    """

    for i in range(len(LAYERS_PER_BLOCK_COUNTS_AXIS1)):
        for j in range(len(FIRST_LAYER_FILTER_COUNTS_AXIS2)):
            for k in range(len(GFS_LEAD_TIME_COUNTS_AXIS3)):
                option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

                option_dict[chiu_net_arch.GFS_3D_DIMENSIONS_KEY] = numpy.array(
                    [265, 537, 3, GFS_LEAD_TIME_COUNTS_AXIS3[k], 5], dtype=int
                )

                option_dict[chiu_net_arch.GFS_2D_DIMENSIONS_KEY] = numpy.array(
                    [265, 537, GFS_LEAD_TIME_COUNTS_AXIS3[k], 7], dtype=int
                )

                option_dict[chiu_net_arch.GFS_ENCODER_NUM_CONV_LAYERS_KEY] = (
                    numpy.full(7, LAYERS_PER_BLOCK_COUNTS_AXIS1[i], dtype=int)
                )

                option_dict[chiu_net_arch.GFS_ENCODER_NUM_CHANNELS_KEY] = (
                    FIRST_LAYER_FILTER_COUNTS_AXIS2[j] *
                    numpy.linspace(1, 7, num=7, dtype=int)
                )

                option_dict[chiu_net_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY] = (
                    int(numpy.round(0.25 * FIRST_LAYER_FILTER_COUNTS_AXIS2[j]))
                    * numpy.linspace(1, 7, num=7, dtype=int)
                )

                option_dict[chiu_net_arch.DECODER_NUM_CHANNELS_KEY] = (
                    option_dict[chiu_net_arch.GFS_ENCODER_NUM_CHANNELS_KEY] +
                    option_dict[chiu_net_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY]
                )[:-1]

                model_object = chiu_net_arch.create_model(
                    option_dict=option_dict,
                    loss_function=custom_losses.dual_weighted_mse(),
                    metric_list=[]
                )

                output_file_name = (
                    '{0:s}/num-conv-layers-per-block={1:d}_'
                    'num-first-layer-filters={2:02d}_'
                    'num-gfs-lead-times={3:02d}/model.h5'
                ).format(
                    OUTPUT_DIR_NAME,
                    LAYERS_PER_BLOCK_COUNTS_AXIS1[i],
                    FIRST_LAYER_FILTER_COUNTS_AXIS2[j],
                    GFS_LEAD_TIME_COUNTS_AXIS3[k]
                )

                file_system_utils.mkdir_recursive_if_necessary(
                    file_name=output_file_name
                )

                print('Writing model to: "{0:s}"...'.format(output_file_name))
                model_object.save(
                    filepath=output_file_name, overwrite=True,
                    include_optimizer=True
                )

                metafile_name = neural_net.find_metafile(
                    model_file_name=output_file_name,
                    raise_error_if_missing=False
                )
                option_dict[neural_net.LOSS_FUNCTION_KEY] = (
                    'custom_losses.dual_weighted_mse()'
                )

                neural_net.write_metafile(
                    pickle_file_name=metafile_name,
                    num_epochs=100,
                    num_training_batches_per_epoch=32,
                    training_option_dict={},
                    num_validation_batches_per_epoch=16,
                    validation_option_dict={},
                    loss_function_string='custom_losses.dual_weighted_mse()',
                    plateau_patience_epochs=10,
                    plateau_learning_rate_multiplier=0.6,
                    early_stopping_patience_epochs=50
                )


if __name__ == '__main__':
    _run()
