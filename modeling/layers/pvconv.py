"""PVConv block definiton."""

from typing import Tuple
import tensorflow as tf

from modeling.layers.voxelization import Voxelization
from modeling.layers.mlp import ConvBn

from ops.voxelization_ops import trilinear_devoxelize


class PVConv(tf.keras.layers.Layer):
  """The infamous PVConv block."""

  def __init__(
    self,
    out_channels: int,
    kernel_size: int,
    resolution: int,
    eps: float,
    normalize: bool,
    with_se: bool,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self._out_channels = out_channels
    self._kernel_size = kernel_size
    self._resolution = resolution
    self._eps = eps
    self._normalize = normalize
    if with_se:
      raise NotImplementedError("SE3d layer not implemented for PVConv block.")

  def build(self, input_shape) -> None:
    self._voxelization = Voxelization(
      self._resolution, self._normalize, self._eps
    )
    # TODO: Verify 'same' padding is consistent with original implementation
    #       in case they use a kernel size other than 3.
    self._conv = tf.keras.layers.Conv3D(
      self._out_channels, self._kernel_size, padding="same"
    )
    self._bn = tf.keras.layers.BatchNormalization(axis=1, epsilon=1e-4)
    self._lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)
    self._point_features = ConvBn(out_channels=self._out_channels)

    features_shape, _ = input_shape # features_shape = [B, C, R, R, R]
    self._squeeze = tf.keras.layers.Reshape((features_shape[1], -1))
    super().build(input_shape)

  def call(self, inputs, training=None) -> Tuple[tf.Tensor, tf.Tensor]:
    features, coords = inputs
    voxel_features, voxel_coords = self._voxelization(features, coords)
    for _ in range(2):
      voxel_features = self._conv(voxel_features)
      voxel_features = self._bn(voxel_features)
      voxel_features = self._lrelu(voxel_features)
    voxel_features = self._squeeze(voxel_features)
    voxel_features = trilinear_devoxelize(
      voxel_features, voxel_coords, self._resolution, training
    )
    fused_features = voxel_features + self._point_features(features)
    return fused_features, coords
