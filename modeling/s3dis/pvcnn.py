""""PVCNN Model Definition."""

import tensorflow as tf
from modeling.layers.sub_models import (
  PointFeaturesBranch,
  CloudFeaturesBranch,
  ClassificationHead,
)

class PVCNN(tf.keras.Model):
  """PVCNN Model."""
  def __init__(
    self,
    point_voxel_branch: PointFeaturesBranch,
    cloud_features_branch: CloudFeaturesBranch,
    classification_head: ClassificationHead,
    **kwargs,
  ) -> None:
    super().__init__(**kwargs)

    self._pvconv_branch = point_voxel_branch
    self._cloud_features_branch = cloud_features_branch
    self._classification_head = classification_head

  def call(self, inputs, training=None):
    # inputs: Point cloud features with shape [B, C, N]. B is batch size,
    #   N is number of points in the point cloud, and C is 9 or 6 based on
    #   `configs.dataset.use_normalized_coords`. See `dataloaders/s3dis.py` for
    #   more details.
    point_features_list = self._pvconv_branch(inputs, training=training)
    # PVConvs -> 1 ConvBn -> point_features_list[-1] has shape [B, 1024, N].
    cloud_features = self._cloud_features_branch(
      point_features_list[-1], training=training
    )
    # cloud_features has shape [B, 128, N].
    comb_features = tf.concat([*point_features_list, cloud_features], axis=1)
    # comb_features has shape [B, (1024 * len(point_features_list)) + 128, N].
    return self._classification_head(comb_features, training=training)
    # output has shape [B, num_classes=13, N].
