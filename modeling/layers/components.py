"""Common sub-models / components for the networks implemented for PVCNN."""

from typing import List, Tuple, Optional
import functools
import tensorflow as tf

from modeling.layers.mlp import ConvBn, DenseBn
from modeling.layers.pvconv import PVConv
from modeling.layers.nan_replace import replace_nans_with_norm


def create_pointnet_components(
  blocks: Tuple,
  width_multiplier: float,
  voxel_resolution_multiplier: float,
  eps: float = 1e-7,
  normalize: bool = True,
  with_se: bool = False,
  kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
) -> List[tf.keras.layers.Layer]:
  r, vr = width_multiplier, voxel_resolution_multiplier

  layers = []
  for out_channels, num_blocks, voxel_resolution in blocks:
    out_channels = int(r * out_channels)
    if voxel_resolution is None:
      block = ConvBn
    else:
      block = functools.partial(
        PVConv,
        kernel_size=3,
        resolution=int(vr * voxel_resolution),
        eps=eps,
        normalize=normalize,
        with_se=with_se,
      )

    for _ in range(num_blocks):
      layers.append(block(out_channels, kernel_regularizer=kernel_regularizer))

  return layers


def create_mlp_components(
  out_channels: List[float],
  is_classifier: bool,
  dim: int,
  width_multiplier: float,
  kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
) -> List[tf.keras.layers.Layer]:
  assert dim in (1, 2), "Only use 1 or 2 dim layers for MLP blocks"

  if dim == 1:
    block = DenseBn
  else:
    block = ConvBn

  r = width_multiplier
  layers = []
  for oc in out_channels[:-1]:
    if oc < 1:
      layers.append(tf.keras.layers.Dropout(oc))
    else:
      layers.append(block(int(r * oc), kernel_regularizer=kernel_regularizer))

  if dim == 1:
    if is_classifier:
      layers.append(
        tf.keras.layers.Dense(
          out_channels[-1], kernel_regularizer=kernel_regularizer
        )
      )
    else:
      layers.append(
        DenseBn(
          int(r * out_channels[-1]), kernel_regularizer=kernel_regularizer
        )
      )
  else:
    if is_classifier:
      layers.append(
        tf.keras.layers.Conv1D(
          filters=out_channels[-1],
          kernel_size=1,
          kernel_regularizer=kernel_regularizer,
          data_format=None,  # Ensure Conv1D uses keras.backend.image_data_format
        )
      )
    else:
      layers.append(
        ConvBn(int(r * out_channels[-1]), kernel_regularizer=kernel_regularizer)
      )

  return layers


class PointFeaturesBranch(tf.keras.layers.Layer):
  """Point-based feature branch."""

  def __init__(
    self,
    blocks: Tuple,
    width_multiplier: float,
    voxel_resolution_multiplier: int,
    kernel_regularizer: tf.keras.regularizers.Regularizer,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self._layers = create_pointnet_components(
      blocks=blocks,
      width_multiplier=width_multiplier,
      voxel_resolution_multiplier=voxel_resolution_multiplier,
      kernel_regularizer=kernel_regularizer,
    )

  def call(self, inputs, training: bool) -> List[tf.Tensor]:
    # FloatTensor shape [B, 3, N] where axis 1 is x,y,z coordinate for a point.
    coords = inputs[:, :3, :]

    features = inputs
    out_features_list = []
    for layer in self._layers:
      features, _ = layer((features, coords), training=training)
      out_features_list.append(features)
    return out_features_list


class CloudFeaturesBranch(tf.keras.layers.Layer):
  """Cloud-based feature branch."""

  def __init__(
    self,
    width_multiplier: float,
    kernel_regularizer: tf.keras.regularizers.Regularizer,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self._layers = create_mlp_components(
      out_channels=[256, 128],
      is_classifier=False,
      dim=1,
      width_multiplier=width_multiplier,
      kernel_regularizer=kernel_regularizer,
    )

  def build(self, input_shape) -> None:
    self._num_points = int(input_shape[-1])
    super().build(input_shape)

  def call(self, inputs, training: bool) -> tf.Tensor:
    # Get maximum channel value for each channel over all of the points
    x = tf.math.reduce_max(inputs, axis=-1)
    for layer in self._layers:
      x = layer(x, training=training)
    x = replace_nans_with_norm(x)
    # Duplicate output tensor for N size num_points dimension
    return tf.repeat(tf.expand_dims(x, axis=-1), self._num_points, axis=-1)


class ClassificationHead(tf.keras.layers.Layer):
  """Segmentation classification head."""

  def __init__(
    self,
    num_classes: int,
    width_multiplier: float,
    kernel_regularizer: tf.keras.regularizers.Regularizer,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self._layers = create_mlp_components(
      out_channels=[512, 0.3, 256, 0.3, num_classes],
      is_classifier=True,
      dim=2,
      width_multiplier=width_multiplier,
      kernel_regularizer=kernel_regularizer,
    )

  def build(self, input_shape) -> None:
    self._softmax = tf.keras.layers.Softmax(axis=1)
    super().build(input_shape)

  def call(self, inputs, training: bool) -> tf.Tensor:
    x = inputs
    for layer in self._layers:
      x = layer(x, training=training)
    x = replace_nans_with_norm(x)
    return self._softmax(x)
