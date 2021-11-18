"""Experiment configurations for a dummy model."""
import tensorflow as tf

from utils.config import Config, configs

# dataset
configs.train.num_epochs = 2
configs.dataset.num_points = 4096
configs.dataset.holdout_area = 5

# dummy model
N = configs.dataset.num_points
CL = configs.data.num_classes
# B = configs.dataset.batch_size
C = 9 if configs.dataset.use_normalized_coords else 6

# inputs has shape [B, C=9, N].
inputs = tf.keras.Input(shape=(C, N))
outputs = tf.keras.layers.ZeroPadding1D(padding=2)(inputs)
# outputs has shape [B, num_classes=13, N].

configs.model = Config(tf.keras.Model)
configs.model.inputs = inputs
configs.model.outputs = outputs

# metrics
configs.metrics.eval.iou.expected_shape = [None, 13, configs.dataset.num_points]
configs.metrics.train.iou.expected_shape = [None, 13, configs.dataset.num_points]  # fmt: skip
