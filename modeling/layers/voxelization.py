"""Average voxlelization layer."""
from typing import Tuple
import tensorflow as tf

from ops import avg_voxelize


class Voxelization(tf.keras.layers.Layer):
  """Voxelization layer."""

  def __init__(
    self, resolution: int, normalize: bool, eps: float, **kwargs
  ):
    super().__init__(**kwargs)
    self._resolution = resolution
    self._normalize = normalize
    self._eps = eps

  def build(self, input_shape) -> None:
    features_shape, _ = input_shape
    _, self._num_channels, _ = features_shape
    super().build(input_shape)

  def call(self, inputs, training=None) -> Tuple[tf.Tensor, tf.Tensor]:
    # See modeling/layers/pvconv `PVConv.call` for more info on features, coords
    features, coords = inputs
    C, R = self._num_channels, self._resolution

    norm_coords = coords - tf.math.reduce_mean(coords, axis=2, keepdims=True)

    if self._normalize:
      coords_reduced = tf.math.reduce_max(
        tf.norm(norm_coords, axis=1, keepdims=True),
        axis=2,
        keepdims=True,
      )
      norm_coords = norm_coords / (coords_reduced * 2.0 + self._eps) + 0.5
    else:
      norm_coords = (norm_coords + 1) / 2.0

    norm_coords = tf.clip_by_value(norm_coords * R, 0, R - 1)
    vox_coords = tf.cast(tf.round(norm_coords), dtype=tf.int32)

    vox_features_sqzd, _, _ = avg_voxelize(features, vox_coords, R)
    voxelized_features = tf.reshape(vox_features_sqzd, shape=(-1, C, R, R, R))

    return voxelized_features, norm_coords
