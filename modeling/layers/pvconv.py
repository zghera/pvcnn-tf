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
    self._conv_layers = []
    self._bn_layers = []
    self._relu_layers = []
    for _ in range(2):
      self._conv_layers.append(
        tf.keras.layers.Conv3D(
          self._out_channels,
          self._kernel_size,
          padding="same",
          kernel_regularizer=self._kernel_regularizer,
        )
      )
      self._bn_layers.append(
        tf.keras.layers.BatchNormalization(axis=1, epsilon=1e-4)
      )
      self._relu_layers.append(tf.keras.layers.LeakyReLU(alpha=0.1))
    # Add extra batch norm layer to add before 1st conv3d layer
    # self._bn_layers.append(
    #   tf.keras.layers.BatchNormalization(axis=1, epsilon=1e-4)
    # )

    self._point_features = ConvBn(
      out_channels=self._out_channels,
      kernel_regularizer=self._kernel_regularizer,
    )

    self._squeeze = tf.keras.layers.Reshape((self._out_channels, -1))
    super().build(input_shape)

  # @staticmethod
  # def replace_nans_with_zeros(has_nans: tf.Tensor) -> tf.Tensor:
  #   return tf.where(tf.math.is_nan(has_nans), tf.zeros_like(has_nans), has_nans)

  # TODO: Delete debugging prints later
  def call(self, inputs, training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    # IC = input channels | OC = output channels (self._out_channels)
    # features = [B, IC, N]  |  coords = [B, 3, N]
    features, coords = inputs
    tf.print("\nfeatures nans =", tf.size(tf.where(tf.math.is_nan(features))))
    # tf.print("coords nans =", tf.size(tf.where(tf.math.is_nan(coords))))
    # features = tf.debugging.assert_all_finite(features, "features is nan")
    # coords = tf.debugging.assert_all_finite(coords, "coords is nan")

    voxel_features, voxel_coords = self._voxelization((features, coords))
    # |--> voxel_features = [B, IC, R, R, R]  |  voxel_coords = [B, 3, N]
    tf.print("voxelization features nans =", tf.size(tf.where(tf.math.is_nan(voxel_features))))
    voxel_features = tf.where(tf.math.is_nan(voxel_features), tf.zeros_like(voxel_features), voxel_features)
    # tf.print("voxelization features CLIPPED nans =", tf.size(tf.where(tf.math.is_nan(voxel_features))))
    # tf.print("voxelization coords nans =", tf.size(tf.where(tf.math.is_nan(voxel_coords))))
    # voxel_features = tf.debugging.assert_all_finite(voxel_features, "voxelization feat is nan")
    # voxel_coords = tf.debugging.assert_all_finite(voxel_coords, "voxelization coords is nan")

    # voxel_features = self._bn_layers[-1](voxel_features, training=training)
    for conv, bn, relu in zip(self._conv_layers, self._bn_layers, self._relu_layers):
      voxel_features = conv(voxel_features)
      # tf.print("conv3d out features nans =", tf.size(tf.where(tf.math.is_nan(voxel_features))))
      # voxel_features = tf.debugging.assert_all_finite(voxel_features, "conv3d out feat is nan")
      voxel_features = bn(voxel_features, training=training)
      voxel_features = relu(voxel_features)

    # |--> voxel_features = [B, OC, R, R, R]
    voxel_features = self._squeeze(voxel_features)
    # |--> voxel_features = [B, OC, R**3]
    voxel_features, _, _ = trilinear_devoxelize(
      voxel_features, voxel_coords, self._resolution, training
    )
    tf.print("devox out features nans =", tf.size(tf.where(tf.math.is_nan(voxel_features))))
    # voxel_features = tf.debugging.assert_all_finite(voxel_features, "devox out feat is nan")
    # |--> voxel_features = [B, OC, N]
    point_features = self._point_features(features, training=training)
    tf.print("point feautes nans =", tf.size(tf.where(tf.math.is_nan(point_features))))
    # |--> point_features = [B, OC, N]
    fused_features = voxel_features + point_features
    # tf.print("fused out features nans =", tf.size(tf.where(tf.math.is_nan(fused_features))))
    # fused_features = tf.debugging.assert_all_finite(fused_features, "fused feat is nan")
    # |--> fused_features = [B, OC, N]
    return fused_features, coords
