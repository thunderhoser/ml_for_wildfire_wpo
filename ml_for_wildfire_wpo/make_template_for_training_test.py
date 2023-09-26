"""Makes neural-net template for training test."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import custom_losses
import neural_net
import chiu_net_architecture
import architecture_utils
import file_system_utils

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_models/'
    'training_test/template'
)


def _run():
    """Makes neural-net template for training test.

    This is effectively the main method.
    """

    option_dict = {
        chiu_net_architecture.GFS_3D_DIMENSIONS_KEY:
            numpy.array([265, 537, 3, 5, 5], dtype=int),
        chiu_net_architecture.GFS_2D_DIMENSIONS_KEY:
            numpy.array([265, 537, 5, 7], dtype=int),
        chiu_net_architecture.ERA5_CONST_DIMENSIONS_KEY: None,
        chiu_net_architecture.LAGGED_TARGET_DIMENSIONS_KEY: None,
        chiu_net_architecture.NUM_FC_CONV_LAYERS_KEY: 1,
        chiu_net_architecture.FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
        chiu_net_architecture.USE_3D_CONV_IN_FC_KEY: True,
        chiu_net_architecture.NUM_LEVELS_KEY: 6,
        chiu_net_architecture.CONV_LAYER_COUNTS_KEY:
            numpy.full(7, 2, dtype=int),
        chiu_net_architecture.CHANNEL_COUNTS_KEY:
            numpy.array([32, 48, 64, 96, 128, 192, 256], dtype=int),
        chiu_net_architecture.ENCODER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
        chiu_net_architecture.DECODER_DROPOUT_RATES_KEY: numpy.full(6, 0.),
        chiu_net_architecture.SKIP_DROPOUT_RATES_KEY: numpy.full(6, 0.),
        chiu_net_architecture.INCLUDE_PENULTIMATE_KEY: False,
        chiu_net_architecture.INNER_ACTIV_FUNCTION_KEY:
            architecture_utils.RELU_FUNCTION_STRING,
        chiu_net_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
        chiu_net_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
            architecture_utils.SIGMOID_FUNCTION_STRING,
        chiu_net_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
        chiu_net_architecture.L1_WEIGHT_KEY: 0.,
        chiu_net_architecture.L2_WEIGHT_KEY: 1e-6,
        chiu_net_architecture.USE_BATCH_NORM_KEY: True
    }

    model_object = chiu_net_architecture.create_model(
        option_dict=option_dict,
        loss_function=custom_losses.mean_squared_error(),
        metric_list=[]
    )

    output_file_name = '{0:s}/model.h5'.format(OUTPUT_DIR_NAME)
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_file_name
    )

    print('Writing model to: "{0:s}"...'.format(output_file_name))
    model_object.save(
        filepath=output_file_name, overwrite=True,
        include_optimizer=True
    )

    metafile_name = neural_net.find_metafile(
        model_file_name=output_file_name, raise_error_if_missing=False
    )
    option_dict[neural_net.LOSS_FUNCTION_KEY] = (
        'custom_losses.mean_squared_error()'
    )

    neural_net.write_metafile(
        pickle_file_name=metafile_name,
        num_epochs=100,
        num_training_batches_per_epoch=32,
        training_option_dict={},
        num_validation_batches_per_epoch=16,
        validation_option_dict={},
        loss_function_string='custom_losses.mean_squared_error()',
        plateau_patience_epochs=10,
        plateau_learning_rate_multiplier=0.6,
        early_stopping_patience_epochs=50
    )


if __name__ == '__main__':
    _run()
