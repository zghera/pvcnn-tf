""""PVCNN Model Definition."""

from typing import Tuple
import tensorflow as tf
from modeling.layers import (
  PointFeaturesBranch,
  CloudFeaturesBranch,
  ClassificationHead,
)


class PVCNN(tf.keras.Model):
  """PVCNN Model."""

  def __init__(
    self,
    point_voxel_blocks: Tuple,
    voxel_resolution_multiplier: int,
    kernel_regularizer: tf.keras.regularizers.Regularizer,
    num_classes: int,
    width_multiplier: int,
    **kwargs,
  ) -> None:
    super().__init__(**kwargs)

    self._pvconv_branch = PointFeaturesBranch(
      point_voxel_blocks,
      width_multiplier,
      voxel_resolution_multiplier,
      kernel_regularizer,
    )
    self._cloud_features_branch = CloudFeaturesBranch(
      width_multiplier, kernel_regularizer
    )
    self._classification_head = ClassificationHead(
      num_classes, width_multiplier, kernel_regularizer
    )

  # TODO: Delete debugging prints later
  def call(self, inputs, training=None):
    # inputs: Point cloud features with shape [B, C, N]. B is batch size,
    #   N is number of points in the point cloud, and C is 9 or 6 based on
    #   `configs.dataset.use_normalized_coords`. See `dataloaders/s3dis.py` for
    #   more details.

    # print("\npvcnn inputs num nans =", tf.size(tf.where(tf.math.is_nan(inputs))))

    # PVConvs -> 1 ConvBn -> point_features_list[-1] has shape [B, 1024, N].
    point_features_list = self._pvconv_branch(inputs, training=training)
    ######################### Debugging #########################
    # for i, point_features in enumerate(point_features_list):
    # print(f"point features {i} = {point_features}")
    # print(f"point feature {i} shape = {point_features.shape}")
    #############################################################
    # cloud_features has shape [B, 128, N].
    cloud_features = self._cloud_features_branch(
      point_features_list[-1], training=training
    )
    ######################### Debugging #########################
    # print(f"cloud feature shape = {cloud_features.shape}")
    #############################################################
    # comb_features has shape [B, (1024 * len(point_features_list)) + 128, N].
    comb_features = tf.concat([*point_features_list, cloud_features], axis=1)
    # output has shape [B, num_classes=13, N].
    return self._classification_head(comb_features, training=training)
