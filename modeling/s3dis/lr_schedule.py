"""Define custom learning rate schedulers."""

import tensorflow as tf


class AttentionSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate scheduler based off of the scheduler used in 
     Transformers (Vaswani, Ashish, et al.). Adapted from 
     https://www.tensorflow.org/text/tutorials/transformer#optimizer.
  """

  def __init__(self, d_model=128, warmup_steps=4000, eps=1e-8):
    super().__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps
    self.eps = eps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) + self.eps
