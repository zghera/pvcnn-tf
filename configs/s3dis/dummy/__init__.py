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
# C = 9 if configs.dataset.use_normalized_coords else 6

# inputs has shape [B, C, N].
dummy_model = tf.keras.Sequential()
dummy_model.add(tf.keras.layers.Flatten())
dummy_model.add(tf.keras.layers.Dense(CL * N))
# output has shape [B, num_classes=13, N].
dummy_model.add(tf.keras.layers.Reshape((CL, N)))

configs.model = Config(dummy_model)
