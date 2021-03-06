"""MLP blocks."""
from typing import Optional, Tuple, Union
import tensorflow as tf
from modeling.layers.nan_replace import replace_nans_with_norm


class ConvBn(tf.keras.layers.Layer):
  """ConvBN Block.

  Building block that combines convolutional, batch normalization, and ReLU
  layers. This is refered to as the SharedMLP block in the original
  implementation.
  """

  def __init__(
    self,
    out_channels: int,
    dim: int = 1,
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    **kwargs,
  ):
    assert dim in (1, 2), "Only use 1 or 2 dim conv layers for ConvBn block."
    super().__init__(**kwargs)

    self._dim = dim
    self._out_channels = out_channels
    self._kernel_regularizer = kernel_regularizer

  def build(self, input_shape) -> None:
    conv_args = {
      "filters": self._out_channels,
      "kernel_size": 1,
      "kernel_regularizer": self._kernel_regularizer,
      "data_format": None,  # Ensure Conv1D uses keras.backend.image_data_format
    }
    if self._dim == 1:
      self._conv = tf.keras.layers.Conv1D(**conv_args)
    else:
      self._conv = tf.keras.layers.Conv2D(**conv_args)
    self._bn = tf.keras.layers.BatchNormalization(axis=1)
    self._relu = tf.keras.layers.ReLU()
    super().build(input_shape)

  def _call(self, inputs, training) -> tf.Tensor:
    x = self._conv(inputs)
    x = replace_nans_with_norm(x)
    x = self._bn(x, training=training)
    x = replace_nans_with_norm(x)
    return self._relu(x)

  def call(
    self, inputs, training=None
  ) -> Union[tf.Tensor, Tuple[tf.Tensor, ...]]:
    if isinstance(inputs, (list, tuple)):
      return (self._call(inputs[0], training), *inputs[1:])

    return self._call(inputs, training)


class DenseBn(tf.keras.layers.Layer):
  """LinearBN Block.

  Building block that combines dense, batch normalization, and ReLU layers.
  """

  def __init__(
    self,
    out_channels: int,
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self._out_channels = out_channels
    self._kernel_regularizer = kernel_regularizer

  def build(self, input_shape) -> None:
    self._fc = tf.keras.layers.Dense(
      self._out_channels, kernel_regularizer=self._kernel_regularizer
    )
    self._bn = tf.keras.layers.BatchNormalization()
    self._relu = tf.keras.layers.ReLU()
    super().build(input_shape)

  def call(self, inputs, training: bool) -> tf.Tensor:
    x = self._fc(inputs)
    x = replace_nans_with_norm(x)
    x = self._bn(x, training=training)
    x = replace_nans_with_norm(x)
    return self._relu(x)
