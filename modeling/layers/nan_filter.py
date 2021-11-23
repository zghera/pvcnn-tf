import tensorflow as tf

class NanFilter(tf.keras.layers.Layer):
  def __init__(self, window_len: int = 10):
    super().__init__()
    self.window_len = window_len

  def build(self, input_shape):
    self.running_average = self.add_weight(shape=input_shape, trainable=False, initializer='zeros')
    super().build(input_shape)

  def call(self, inputs):
    nans_repl_w_zero = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
    self.running_average.assign(tf.cond(tf.reduce_all(self.running_average == tf.zeros(inputs.shape)),
        true_fn=lambda: inputs,
        false_fn=lambda: ((self.running_average * self.window_len) + nans_repl_w_zero) / (self.window_len + 1)))
    return tf.where(tf.math.is_nan(inputs), self.running_average, inputs)
