"""A temporary module to fix the NaN issues during training."""
import tensorflow as tf


def replace_nans_with_norm(has_nans: tf.Tensor) -> tf.Tensor:
  """Replace NaNs in the input tensor with the norm of the tensor once NaNs
  have been replaced with zeros."""
  repl_nans_with_zeros = tf.where(
    tf.math.is_nan(has_nans), tf.zeros_like(has_nans), has_nans
  )
  norm = tf.norm(repl_nans_with_zeros)
  return tf.where(
    tf.math.is_nan(has_nans), tf.fill(has_nans.shape, norm), has_nans
  )
