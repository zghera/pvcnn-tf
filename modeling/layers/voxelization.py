"""Average voxlelization layer."""
from typing import Tuple
import tensorflow as tf

from ops.voxelization_ops import avg_voxelize


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
    _, coords_shape = input_shape
    self.norm_coords = self.add_weight(shape=coords_shape, trainable=True)
    self.vox_coords = self.add_weight(
      shape=coords_shape, dtype=tf.int32, trainable=True
    )
    super().build(input_shape)

  def call(self, inputs, training=None) -> Tuple[tf.Tensor, tf.Tensor]:
    # See modeling/layers/pvconv.py for details on features,coords
    features, coords = inputs
    self.norm_coords = coords - tf.math.reduce_mean(
      coords, axis=2, keepdims=True
    )

    if self._normalize:
      coords_reduced = tf.math.reduce_max(
        tf.norm(self.norm_coords, axis=1, keepdims=True),
        axis=2,
        keepdims=True,
      )
      self.norm_coords = (
        self.norm_coords / (coords_reduced * 2.0 + self._eps) + 0.5
      )
    else:
      self.norm_coords = (self.norm_coords + 1) / 2.0

    self.norm_coords = tf.clip_by_value(
      self.norm_coords * self._resoltion, 0, self._resolution - 1
    )
    self.vox_coords = tf.cast(tf.round(self.norm_coords), dtype=tf.int32)

    vox_features_sqzd, _, _ = avg_voxelize(
      features, self.vox_coords, self._resolution
    )
    B, C, _ = features.shape
    R = self._resolution
    voxelized_features = tf.reshape(vox_features_sqzd, shape=(B, C, R, R, R))

    return voxelized_features, self.norm_cords
