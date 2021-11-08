"""ConvBN block definition."""
from typing import Tuple
import tensorflow as tf


class ConvBn(tf.keras.layers.Layer):
  """ConvBN Block

  Building block that combines convolutional, batch normalization, and ReLU
  layers. This is refered to as the SharedMLP block in the original
  implementation."""

  def __init__(self, out_channels: int, dim: int = 1, **kwargs):
    assert dim in (1, 2), "Cnly use 1 or 2 dim Conv layers for ConvBn Block"
    self._dim = dim
    self._out_channels = out_channels
    super().__init__(**kwargs)

  def build(self, input_shape) -> None:
    conv_args = {
      'filters': self._out_channels,
      'kernel_size': 1
    }
    if self._dim == 1:
      self._conv = tf.keras.layers.Conv1D(**conv_args)
    else:
      self._conv = tf.keras.layers.Conv2D(**conv_args)
    self._bn = tf.keras.layers.BatchNormalization()
    self._relu = tf.keras.layers.ReLU()
    super().build(input_shape)

  def call(self, inputs, training=None) -> Tuple[tf.Tensor]:
    x = self._conv(inputs)
    x = self._bn(x, training=training)
    return self._relu(x)
