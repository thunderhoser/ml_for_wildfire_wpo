"""Unit tests for custom_losses.py."""

import unittest
import numpy
import tensorflow
from keras import backend as K
from ml_for_wildfire_wpo.machine_learning import custom_losses

TOLERANCE = 1e-6

# The following constants are used to test dual_weighted_mse with one channel.
THIS_TARGET_MATRIX = numpy.array([
    [1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19, 20, 21],
    [22, 23, 24, 25, 26, 27, 28]
], dtype=float)

THIS_PREDICTION_MATRIX = numpy.array([
    [4, 4, 4, 4, 4, 4, 4],
    [11, 11, 11, 11, 11, 11, 11],
    [18, 18, 18, 18, 18, 18, 18],
    [25, 25, 25, 25, 25, 25, 25]
], dtype=float)

THIS_TARGET_MATRIX = numpy.stack(
    [0.5 * THIS_TARGET_MATRIX, THIS_TARGET_MATRIX, 2 * THIS_TARGET_MATRIX],
    axis=0
)

THIS_PREDICTION_MATRIX = numpy.stack([
    0.5 * THIS_PREDICTION_MATRIX - 1,
    THIS_PREDICTION_MATRIX,
    2 * THIS_PREDICTION_MATRIX + 1
], axis=0)

THIS_TARGET_MATRIX = numpy.stack(
    [THIS_TARGET_MATRIX, numpy.ones(THIS_TARGET_MATRIX.shape)], axis=-1
)
THIS_PREDICTION_MATRIX = numpy.expand_dims(THIS_PREDICTION_MATRIX, axis=-1)
THIS_PREDICTION_MATRIX = numpy.expand_dims(THIS_PREDICTION_MATRIX, axis=-1)

TARGET_TENSOR_1CHANNEL = tensorflow.constant(
    THIS_TARGET_MATRIX, dtype=tensorflow.float32
)
PREDICTION_TENSOR_1CHANNEL = tensorflow.constant(
    THIS_PREDICTION_MATRIX, dtype=tensorflow.float32
)

THIS_DUAL_WEIGHT_MATRIX_EXAMPLE1 = numpy.array([
    [1, 1, 1.5, 2, 2.5, 3, 3.5],
    [4.5, 4.5, 5, 5.5, 6, 6.5, 7],
    [8, 8, 8.5, 9, 9.5, 10, 10.5],
    [11.5, 11.5, 12, 12.5, 13, 13.5, 14]
], dtype=float)

THIS_DUAL_WEIGHT_MATRIX_EXAMPLE2 = numpy.array([
    [4, 4, 4, 4, 5, 6, 7],
    [11, 11, 11, 11, 12, 13, 14],
    [18, 18, 18, 18, 19, 20, 21],
    [25, 25, 25, 25, 26, 27, 28]
], dtype=float)

THIS_DUAL_WEIGHT_MATRIX_EXAMPLE3 = numpy.array([
    [9, 9, 9, 9, 10, 12, 14],
    [23, 23, 23, 23, 24, 26, 28],
    [37, 37, 37, 37, 38, 40, 42],
    [51, 51, 51, 51, 52, 54, 56]
], dtype=float)

THIS_DUAL_WEIGHT_MATRIX = numpy.stack([
    THIS_DUAL_WEIGHT_MATRIX_EXAMPLE1,
    THIS_DUAL_WEIGHT_MATRIX_EXAMPLE2,
    THIS_DUAL_WEIGHT_MATRIX_EXAMPLE3
], axis=0)

THIS_DUAL_WEIGHT_MATRIX = numpy.expand_dims(THIS_DUAL_WEIGHT_MATRIX, axis=-1)

THIS_ERROR_MATRIX = (
    THIS_DUAL_WEIGHT_MATRIX *
    (THIS_TARGET_MATRIX[..., :-1] - THIS_PREDICTION_MATRIX[..., 0]) ** 2
)
DWMSE_1CHANNEL = numpy.mean(THIS_ERROR_MATRIX)

# The following constants are used to test dual_weighted_mse with two channels.
CHANNEL_WEIGHTS_2CHANNELS = numpy.stack([10, 1.5])

THIS_TARGET_MATRIX_CHANNEL1 = numpy.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
], dtype=float)

THIS_PREDICTION_MATRIX_CHANNEL1 = numpy.array([
    [2, 2, 2],
    [5, 5, 5],
    [8, 8, 8],
    [11, 11, 11]
], dtype=float)

THIS_TARGET_MATRIX_CHANNEL2 = numpy.array([
    [30, 20, 10],
    [60, 50, 40],
    [90, 80, 70],
    [120, 110, 100]
], dtype=float)

THIS_PREDICTION_MATRIX_CHANNEL2 = numpy.array([
    [20, 20, 20],
    [50, 50, 50],
    [80, 80, 80],
    [110, 110, 110]
], dtype=float)

THIS_TARGET_MATRIX = numpy.stack(
    [THIS_TARGET_MATRIX_CHANNEL1, THIS_TARGET_MATRIX_CHANNEL2], axis=-1
)
THIS_PREDICTION_MATRIX = numpy.stack(
    [THIS_PREDICTION_MATRIX_CHANNEL1, THIS_PREDICTION_MATRIX_CHANNEL2], axis=-1
)

THIS_TARGET_MATRIX = numpy.stack(
    [0.5 * THIS_TARGET_MATRIX, THIS_TARGET_MATRIX], axis=0
)
THIS_PREDICTION_MATRIX = numpy.stack([
    0.5 * THIS_PREDICTION_MATRIX - 1,
    THIS_PREDICTION_MATRIX
], axis=0)

THIS_TARGET_MATRIX = numpy.concatenate(
    [THIS_TARGET_MATRIX, numpy.ones(THIS_TARGET_MATRIX[..., [0]].shape)],
    axis=-1
)
THIS_PREDICTION_MATRIX = numpy.expand_dims(THIS_PREDICTION_MATRIX, axis=-1)

TARGET_TENSOR_2CHANNELS = tensorflow.constant(
    THIS_TARGET_MATRIX, dtype=tensorflow.float32
)
PREDICTION_TENSOR_2CHANNELS = tensorflow.constant(
    THIS_PREDICTION_MATRIX, dtype=tensorflow.float32
)

THIS_DUAL_WEIGHT_MATRIX_EXAMPLE2_CHANNEL1 = numpy.array([
    [2, 2, 3],
    [5, 5, 6],
    [8, 8, 9],
    [11, 11, 12]
], dtype=float)

THIS_DUAL_WEIGHT_MATRIX_EXAMPLE2_CHANNEL2 = numpy.array([
    [30, 20, 20],
    [60, 50, 50],
    [90, 80, 80],
    [120, 110, 110]
], dtype=float)

THIS_DUAL_WEIGHT_MATRIX_EXAMPLE1_CHANNEL1 = numpy.array([
    [0.5, 1, 1.5],
    [2, 2.5, 3],
    [3.5, 4, 4.5],
    [5, 5.5, 6]
], dtype=float)

THIS_DUAL_WEIGHT_MATRIX_EXAMPLE1_CHANNEL2 = numpy.array([
    [15, 10, 9],
    [30, 25, 24],
    [45, 40, 39],
    [60, 55, 54]
], dtype=float)

THIS_TOTAL_WEIGHT_MATRIX_CHANNEL1 = 10 * numpy.stack([
    THIS_DUAL_WEIGHT_MATRIX_EXAMPLE1_CHANNEL1,
    THIS_DUAL_WEIGHT_MATRIX_EXAMPLE2_CHANNEL1
], axis=0)

THIS_TOTAL_WEIGHT_MATRIX_CHANNEL2 = 1.5 * numpy.stack([
    THIS_DUAL_WEIGHT_MATRIX_EXAMPLE1_CHANNEL2,
    THIS_DUAL_WEIGHT_MATRIX_EXAMPLE2_CHANNEL2
], axis=0)

THIS_TOTAL_WEIGHT_MATRIX = numpy.stack(
    [THIS_TOTAL_WEIGHT_MATRIX_CHANNEL1, THIS_TOTAL_WEIGHT_MATRIX_CHANNEL2],
    axis=-1
)

THIS_ERROR_MATRIX = (
    THIS_TOTAL_WEIGHT_MATRIX *
    (THIS_TARGET_MATRIX[..., :-1] - THIS_PREDICTION_MATRIX[..., 0]) ** 2
)
DWMSE_2CHANNELS = numpy.mean(THIS_ERROR_MATRIX)


class CustomLossesTests(unittest.TestCase):
    """Each method is a unit test for custom_losses.py."""

    def test_dual_weighted_mse_with_1channel(self):
        """Ensures correct output from dual_weighted_mse.

        In this case, there is only one channel (target variable).
        """

        this_loss_function = custom_losses.dual_weighted_mse(
            channel_weights=numpy.array([1.]),
            function_name='dual_weighted_mse',
            test_mode=True
        )
        this_dwmse = K.eval(this_loss_function(
            TARGET_TENSOR_1CHANNEL, PREDICTION_TENSOR_1CHANNEL
        ))

        self.assertTrue(numpy.isclose(
            this_dwmse, DWMSE_1CHANNEL, atol=TOLERANCE
        ))

    def test_dual_weighted_mse_with_2channels(self):
        """Ensures correct output from dual_weighted_mse.

        In this case, there are two channels.
        """

        this_loss_function = custom_losses.dual_weighted_mse(
            channel_weights=CHANNEL_WEIGHTS_2CHANNELS,
            function_name='dual_weighted_mse',
            test_mode=True
        )
        this_dwmse = K.eval(this_loss_function(
            TARGET_TENSOR_2CHANNELS, PREDICTION_TENSOR_2CHANNELS
        ))

        self.assertTrue(numpy.isclose(
            this_dwmse, DWMSE_2CHANNELS, atol=TOLERANCE
        ))

    def test_dual_weighted_mse_1channel(self):
        """Ensures correct output from dual_weighted_mse_1channel."""

        loss_function_channel1 = custom_losses.dual_weighted_mse_1channel(
            channel_weight=CHANNEL_WEIGHTS_2CHANNELS[0],
            channel_index=0,
            function_name='dwmse_channel1',
            test_mode=True
        )
        dwmse_channel1 = K.eval(loss_function_channel1(
            TARGET_TENSOR_2CHANNELS, PREDICTION_TENSOR_2CHANNELS
        ))

        loss_function_channel2 = custom_losses.dual_weighted_mse_1channel(
            channel_weight=CHANNEL_WEIGHTS_2CHANNELS[1],
            channel_index=1,
            function_name='dwmse_channel2',
            test_mode=True
        )
        dwmse_channel2 = K.eval(loss_function_channel2(
            TARGET_TENSOR_2CHANNELS, PREDICTION_TENSOR_2CHANNELS
        ))

        this_dwmse = (dwmse_channel1 + dwmse_channel2) / 2
        self.assertTrue(numpy.isclose(
            this_dwmse, DWMSE_2CHANNELS, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
