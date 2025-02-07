"""Converts model weights from HDF5 file to Pickle file."""

import os
import sys
import pickle
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import neural_net

INPUT_FILE_ARG_NAME = 'input_hdf5_file_name'
OUTPUT_FILE_ARG_NAME = 'output_pickle_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing model weights in standard HDF5 format '
    'produced by Keras.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Model weights will be pickled to this location.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_name, output_file_name):
    """Converts model weights from HDF5 file to Pickle file.

    This is effectively the main method.

    :param input_file_name: See documentation at top of this script.
    :param output_file_name: Same.
    """

    print('Reading model weights from: "{0:s}"...'.format(input_file_name))
    model_object = neural_net.read_model(input_file_name)
    model_object.summary()

    input_dir_name = os.path.split(input_file_name)[0]
    output_dir_name = os.path.split(output_file_name)[0]
    assert input_dir_name == output_dir_name

    print('Writing model weights to: "{0:s}"...'.format(output_file_name))
    output_file_handle = open(output_file_name, 'wb')
    pickle.dump(model_object.get_weights(), output_file_handle)
    output_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
