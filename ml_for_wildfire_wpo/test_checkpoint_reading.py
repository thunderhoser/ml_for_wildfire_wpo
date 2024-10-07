"""USE ONCE AND DESTROY."""

import os
import sys
import re
import keras
import tensorflow

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import neural_net
import custom_losses
import custom_metrics

tensorflow.compat.v1.disable_v2_behavior()

MODEL_FILE_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_models/experiment17/training-strategy=20-30-daily_spectral-complexity=080_use-residual-blocks=0/model.weights.h5'

metafile_name = neural_net.find_metafile(
    model_file_name=MODEL_FILE_NAME, raise_error_if_missing=True
)
metadata_dict = neural_net.read_metafile(metafile_name)

chiu_net_pp_architecture_dict = metadata_dict[neural_net.CHIU_NET_PP_ARCHITECTURE_KEY]
if chiu_net_pp_architecture_dict is not None:
    import \
        chiu_net_pp_architecture

    arch_dict = chiu_net_pp_architecture_dict
    if chiu_net_pp_architecture.USE_LEAD_TIME_AS_PRED_KEY not in arch_dict:
        arch_dict[chiu_net_pp_architecture.USE_LEAD_TIME_AS_PRED_KEY] = False

    arch_dict[chiu_net_pp_architecture.LOSS_FUNCTION_KEY] = 'mse'

    for this_key in [
        chiu_net_pp_architecture.OPTIMIZER_FUNCTION_KEY
    ]:
        try:
            arch_dict[this_key] = eval(arch_dict[this_key])
        except:
            arch_dict[this_key] = re.sub(
                r"gradient_accumulation_steps=\d+", "", arch_dict[this_key]
            )
            arch_dict[this_key] = eval(arch_dict[this_key])

    for this_key in [chiu_net_pp_architecture.METRIC_FUNCTIONS_KEY]:
        for k in range(len(arch_dict[this_key])):
            arch_dict[this_key][k] = eval(arch_dict[this_key][k])

    model_object = chiu_net_pp_architecture.create_model(arch_dict)
    # model_object.load_weights(hdf5_file_name)
    print(model_object.get_layer(name='output_conv0').get_weights())

    checkpoint_file_name = MODEL_FILE_NAME.replace('model.weights.h5', 'model_checkpoint_new')
    checkpoint_file_name = checkpoint_file_name.replace('weights.weights.h5', 'model_checkpoint_new')
    print(checkpoint_file_name)

    import tensorflow

    checkpoint_object = tensorflow.train.Checkpoint(model=model_object)
    checkpoint_manager = tensorflow.train.CheckpointManager(
        checkpoint=checkpoint_object,
        directory=os.path.split(checkpoint_file_name)[0],
        max_to_keep=2,
        checkpoint_name='model_checkpoint_new'
    )
    checkpoint_manager.restore_or_initialize()
    print(checkpoint_manager)
    print(checkpoint_manager.checkpoints)
    print(checkpoint_manager.latest_checkpoint)

    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if latest_checkpoint:
        checkpoint_object.restore(latest_checkpoint).expect_partial()
    else:
        print("No checkpoint found.")

    # model_object.summary()
    print('\n\n\n')
    print(model_object.get_layer(name='output_conv0').get_weights())
