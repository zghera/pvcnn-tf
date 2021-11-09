"""Custom operations for voxelization and devoxlelization."""
from typing import Tuple
import tensorflow as tf


def avg_voxelize(
  features: tf.Tensor, coords: tf.Tensor, resolution: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Average pool voxelization
  Args:
    features: features, FloatTensor[b, c, n]
    coords  : coords of each point, IntTensor[b, 3, n]
    resolution : voxel resolution
  Return:
    out : outputs, FloatTensor[b, c, s], s = r ** 3
    ind : voxel index of each point, IntTensor[b, n]
    cnt : #points in each voxel index, IntTensor[b, s]
  """
  return tf.constant(), tf.constant(), tf.constant()


def trilinear_devoxelize(
  features: tf.Tensor, coords: tf.Tensor, resolution: int, is_training: bool
) -> tf.Tensor:
  """TODO"""
  return tf.constant()
