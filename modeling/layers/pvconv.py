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
    voxel_features, voxel_coords = self._voxelization((features, coords))
    # |--> voxel_features = [B, IC, R, R, R]  |  voxel_coords = [B, 3, N]
    # voxel_features = self._bn_layers[-1](voxel_features, training=training)
    # tf.print("\npvconv \n-------")
    # tf.print(" voxelization feats nans =", tf.size(tf.where(tf.math.is_nan(voxel_features))))
    # tf.print(" voxel layers \n ----------")
    for conv, bn, relu in zip(self._conv_layers, self._bn_layers, self._relu_layers):
      voxel_features = conv(voxel_features)
      voxel_features = bn(voxel_features, training=training)
      voxel_features = relu(voxel_features)
    # tf.print(" voxel layers nans =", tf.size(tf.where(tf.math.is_nan(voxel_features))))

    # if tf.greater(tf.size(tf.where(tf.math.is_nan(voxel_features))), 0):
    #   tf.print("PVConv voxel Conv3D layer output has nans")
    #   voxel_features = self.replace_nans_with_zeros(voxel_features)
    # tf.print(" voxel layers nans after =", tf.size(tf.where(tf.math.is_nan(voxel_features))))

    # |--> voxel_features = [B, OC, R, R, R]
    voxel_features = self._squeeze(voxel_features)
    # |--> voxel_features = [B, OC, R**3]
    voxel_features, _, _ = trilinear_devoxelize(
      voxel_features, voxel_coords, self._resolution, training
    )
    # |--> voxel_features = [B, OC, N]
    # tf.print(" trilinear_devox feats nans =", tf.size(tf.where(tf.math.is_nan(voxel_features))))
    fused_features = voxel_features + self._point_features(features)
    # |--> fused_features = [B, OC, N]
    # tf.print(" fused feats nans =", tf.size(tf.where(tf.math.is_nan(fused_features))))
    return fused_features, coords
