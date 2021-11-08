"""MLP blocks."""
from typing import Tuple, Union
import tensorflow as tf


class ConvBn(tf.keras.layers.Layer):
  """ConvBN Block.

  Building block that combines convolutional, batch normalization, and ReLU
  layers. This is refered to as the SharedMLP block in the original
  implementation.
  """

  def __init__(self, out_channels: int, dim: int = 1, **kwargs):
    assert dim in (1, 2), 'Only use 1 or 2 dim conv layers for ConvBn block.'
    super().__init__(**kwargs)

    self._dim = dim
    self._out_channels = out_channels

  def build(self, input_shape) -> None:
    conv_args = {
      'filters': self._out_channels,
      'kernel_size': 1,
    }
    if self._dim == 1:
      self._conv = tf.keras.layers.Conv1D(**conv_args)
    else:
      self._conv = tf.keras.layers.Conv2D(**conv_args)
    self._bn = tf.keras.layers.BatchNormalization()
    self._relu = tf.keras.layers.ReLU()
    super().build(input_shape)

  def _call(self, inputs, training) -> tf.Tensor:
    x = self._conv(inputs)
    x = self._bn(x, training=training)
    return self._relu(x)

  def call(self, inputs, training=None
  ) -> Union[tf.Tensor, Tuple[tf.Tensor, ...]]:
    if isinstance(inputs, (list, tuple)):
      return (self._call(inputs[0], training), *inputs[1:])

    return self._call(inputs, training)

class DenseBn(tf.keras.layers.Layer):
  """LinearBN Block.

  Building block that combines dense, batch normalization, and ReLU layers.
  """

  def __init__(self, out_channels: int, **kwargs):
    super().__init__(**kwargs)
    self._out_channels = out_channels

  def build(self, input_shape) -> None:
    self._fc = tf.keras.layer.Dense(self._out_channels)
    self._bn = tf.keras.layers.BatchNormalization()
    self._relu = tf.keras.layers.ReLU()
    super().build(input_shape)

  def call(self, inputs, training=None) -> tf.Tensor:
    x = self._fc(inputs)
    x = self._bn(x, training=training)
    return self._relu(x)
