"""Common sub-models / components for the networks implemented for PVCNN."""

from typing import List, Tuple
import functools
import tensorflow as tf

from modeling.layers.mlp import ConvBn, DenseBn
from modeling.layers.pvconv import PVConv


def create_pointnet_components(
  blocks: Tuple,
  width_multiplier: float,
  voxel_resolution_multiplier: float = 1,
  eps: float = 0,
  normalize: bool = True,
  with_se: bool = False,
) -> List[tf.keras.layers.Layer]:
  r, vr = width_multiplier, voxel_resolution_multiplier

  layers = []
  for out_channels, num_blocks, voxel_resolution in blocks:
    out_channels = int(r * out_channels)
    if voxel_resolution is None:
      block = ConvBn
    else:
      block = functools.partial(PVConv, kernel_size=3,
                        resolution=int(vr * voxel_resolution),
                        eps=eps, normalize=normalize, with_se=with_se)

    for _ in range(num_blocks):
      layers.append(block(out_channels))

  return layers


def create_mlp_components(
  out_channels: List[float],
  is_classifier: bool,
  dim: int,
  width_multiplier: float,
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
      layers.append(block(int(r * oc)))

  if dim == 1:
    if is_classifier:
      layers.append(tf.keras.layers.Dense(out_channels[-1]))
    else:
      layers.append(DenseBn(int(r * out_channels[-1])))
  else:
    if is_classifier:
      layers.append(
        tf.keras.layers.Conv1d(filters=out_channels[-1], kernel=1))
    else:
      layers.append(ConvBn(int(r * out_channels[-1])))

  return layers


class PointFeaturesBranch(tf.keras.layers.Layer):
  """Point-based feature branch."""

  def __init__(
    self,
    blocks: Tuple,
    width_multiplier: float,
    voxel_resolution_multiplier: int = 1,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self._layers = create_pointnet_components(
      blocks=blocks,
      width_multiplier=width_multiplier,
    )

  def call(self, inputs, training=None) -> List[tf.Tensor]:
    # FloatTensor shape [B, 3, N] where axis 1 is x,y,z coordinate for a point.
    coords = inputs[:, :3, :]

    x = inputs
    out_features_list = []
    for layer in self._layers:
      x, _ = layer((x, coords), training=training)
      out_features_list.append(x)
    return out_features_list


class CloudFeaturesBranch(tf.keras.layers.Layer):
  """Cloud-based feature branch."""

  def __init__(self, width_multiplier: float, **kwargs):
    super().__init__(**kwargs)
    self._layers = create_mlp_components(
      out_channels=[256, 128],
      is_classifier=False,
      dim=1,
      width_multiplier=width_multiplier,
    )

  def call(self, inputs, training=None) -> tf.Tensor:
    inputs = tf.math.reduce_max(inputs, axis=-1)
    x = inputs
    for layer in self._layers:
      x = layer(x, training=training)
    return tf.repeat(tf.expand_dims(x, axis=-1), inputs.size(-1), axis=-1)


class ClassificationHead(tf.keras.layers.Layer):
  """Segmentation classification head."""

  def __init__(self, num_classes: int, width_multiplier: float, **kwargs):
    super().__init__(**kwargs)
    self._layers = create_mlp_components(
      out_channels=[512, 0.3, 256, 0.3, num_classes],
      is_classifier=True,
      dim=2,
      width_multiplier=width_multiplier,
    )

  def call(self, inputs, training=None) -> tf.Tensor:
    x = inputs
    for layer in self._layers:
      x = layer(x, training=training)
    return x
