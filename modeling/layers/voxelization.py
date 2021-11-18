"""Average voxlelization layer."""
from typing import Tuple
import tensorflow as tf

from ops import avg_voxelize


class Voxelization(tf.keras.layers.Layer):
  """Voxelization layer."""

  def __init__(
    self, resolution: int, normalize: bool = True, eps: float = 0, **kwargs
  ):
    super().__init__(**kwargs)
    self._resolution = resolution
    self._normalize = normalize
    self._eps = eps

  def build(self, input_shape) -> None:
    features_shape, coords_shape = input_shape
    self._num_batches, self._num_channels, _ = features_shape

    self._norm_coords = self.add_weight(
      name="norm_coords", shape=coords_shape, trainable=True
    )
    self._vox_coords = self.add_weight(
      name="vox_coords", shape=coords_shape, dtype=tf.int32, trainable=True
    )
    super().build(input_shape)

  def call(self, inputs, training=None) -> Tuple[tf.Tensor, tf.Tensor]:
    # See modeling/layers/pvconv `PVConv.call` for more info on features, coords
    features, coords = inputs
    B, C, R = self._num_batches, self._num_channels, self._resolution

    self._norm_coords = coords - tf.math.reduce_mean(
      coords, axis=2, keepdims=True
    )

    if self._normalize:
      coords_reduced = tf.math.reduce_max(
        tf.norm(self._norm_coords, axis=1, keepdims=True),
        axis=2,
        keepdims=True,
      )
      self._norm_coords = (
        self._norm_coords / (coords_reduced * 2.0 + self._eps) + 0.5
      )
    else:
      self._norm_coords = (self._norm_coords + 1) / 2.0

    self._norm_coords = tf.clip_by_value(self._norm_coords * R, 0, R - 1)
    self._vox_coords = tf.cast(tf.round(self._norm_coords), dtype=tf.int32)

    vox_features_sqzd, _, _ = avg_voxelize(features, self._vox_coords, R)
    voxelized_features = tf.reshape(vox_features_sqzd, shape=(B, C, R, R, R))

    return voxelized_features, self._norm_coords
