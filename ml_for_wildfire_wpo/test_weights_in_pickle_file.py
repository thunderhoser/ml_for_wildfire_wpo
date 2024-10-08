"""USE ONCE AND DESTROY."""

import os
import sys
import re
import pickle
import keras
import numpy
import tensorflow

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import neural_net
import custom_losses
import custom_metrics

tensorflow.compat.v1.disable_v2_behavior()

MODEL_FILE_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_models/'
    'experiment17/'
    'training-strategy=20-30-daily_spectral-complexity=080_use-residual-blocks=0/'
    'model.weights.p'
)

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

    orig_model_object = chiu_net_pp_architecture.create_model(arch_dict)
    model_object = chiu_net_pp_architecture.create_model(arch_dict)

    pickle_file_handle = open(MODEL_FILE_NAME, 'rb')
    model_weights_array_list = pickle.load(pickle_file_handle)
    pickle_file_handle.close()
    model_object.set_weights(model_weights_array_list)

    layer_names = [layer.name for layer in model_object.layers]

    for this_layer_name in layer_names:
        orig_weights_array_list = orig_model_object.get_layer(
            name=this_layer_name
        ).get_weights()

        new_weights_array_list = model_object.get_layer(
            name=this_layer_name
        ).get_weights()

        for k in range(len(orig_weights_array_list)):
            if numpy.allclose(
                    orig_weights_array_list[k], new_weights_array_list[k],
                    atol=1e-6
            ):
                continue

            print('Some weight tensors in layer "{0:s}" did not change!'.format(
                this_layer_name
            ))

            print(this_layer_name)
