"""PVConv block definiton."""

from typing import Tuple
import tensorflow as tf

from modeling.layers.voxelization import Voxelization
from modeling.layers.mlp import ConvBn

from ops import trilinear_devoxelize


class PVConv(tf.keras.layers.Layer):
  """The infamous PVConv block."""

  def __init__(
    self,
    out_channels: int,
    kernel_size: int,
    resolution: int,
    eps: float,
    normalize: bool,
    with_se: bool = False,
    kernel_regularizer: tf.keras.regularizers.Regularizer = None,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self._out_channels = out_channels
    self._kernel_size = kernel_size
    self._resolution = resolution
    self._eps = eps
    self._normalize = normalize
    self._kernel_regularizer = kernel_regularizer
    if with_se:
      raise NotImplementedError("SE3d layer not implemented for PVConv block.")

  def build(self, input_shape) -> None:
    self._voxelization = Voxelization(
      self._resolution, self._normalize, self._eps
    )
    # TODO: Verify 'same' padding is consistent with original implementation
    #       in case they use a kernel size other than 3.
    self._voxel_layers = []
    for _ in range(2):
      self._voxel_layers.append(
        tf.keras.layers.Conv3D(
          self._out_channels,
          self._kernel_size,
          padding="same",
          kernel_regularizer=self._kernel_regularizer,
        )
      )
      self._voxel_layers.append(
        tf.keras.layers.BatchNormalization(axis=1, epsilon=1e-4)
      )
      self._voxel_layers.append(tf.keras.layers.LeakyReLU(alpha=0.1))

    self._point_features = ConvBn(
      out_channels=self._out_channels,
      kernel_regularizer=self._kernel_regularizer,
    )

    features_shape, _ = input_shape
    print(f"features shape for sqeeze layer = {features_shape}")
    self._squeeze = tf.keras.layers.Reshape((features_shape[1], -1))
    print(f"sqeeze layer = {self._squeeze}")
    super().build(input_shape)

  def call(self, inputs, training=None) -> Tuple[tf.Tensor, tf.Tensor]:
    # features = [B, C, N]  |  coords = [B, 3, N]
    features, coords = inputs
    voxel_features, voxel_coords = self._voxelization((features, coords))
    # |--> voxel_features = [B, C, R, R, R]  |  voxel_coords = [B, 3, N]
    for layer in self._voxel_layers:
      voxel_features = layer(voxel_features)
    print(f"sqeeze layer input tensor shape = {voxel_features.shape}")
    voxel_features = self._squeeze(voxel_features)
    # |--> voxel_features = [B, C, R**3]
    voxel_features = trilinear_devoxelize(
      voxel_features, voxel_coords, self._resolution, training
    )
    # |--> voxel_features = [B, C, N]
    fused_features = voxel_features + self._point_features(features)
    # |--> fused_features = [B, C, N]
    return fused_features, coords
