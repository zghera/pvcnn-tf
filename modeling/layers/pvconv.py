"""PVConv block definiton."""

import tensorflow as tf

class PVConv(tf.keras.layers.Layer):
  """The infamous PVConv block."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # TODO
