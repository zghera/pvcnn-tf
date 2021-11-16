"""Experiment configurations for a dummy model."""
import tensorflow as tf

from utils.config import Config, configs

# dataset
configs.dataset.num_points = 4096
configs.dataset.holdout_area = 5

# dummy model
N = configs.dataset.num_points
CL =  configs.data.num_classes
# B = configs.dataset.batch_size
C = 9 if configs.dataset.use_normalized_coords else 6

# inputs has shape [B, C, N].
inputs = tf.keras.Input(shape=(C, N))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(CL * N, activation="relu")(x)
outputs = tf.keras.layers.Reshape((CL,N))(x)

configs.model = Config(tf.keras.Model)
configs.model.inputs = inputs
configs.model.outputs = outputs
