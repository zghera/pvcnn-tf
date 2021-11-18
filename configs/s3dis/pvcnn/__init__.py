"""Experiment configurations for S3DIS PVCNN model."""
# TODO: If TF's CosineDecay much different from Pytorch's equivalent?
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.regularizers import L2

from modeling import (
  PVCNN,
  PointFeaturesBranch,
  CloudFeaturesBranch,
  ClassificationHead,
)
from utils.config import Config, configs

# sub-models
configs.submodel = Config()
configs.submodel.kernel_reg = L2(1e-5)

configs.point_branch = Config(PointFeaturesBranch)
configs.point_branch.blocks = (
  (64, 1, 32),
  (64, 2, 16),
  (128, 1, 16),
  (1024, 1, None),
)
configs.point_branch.voxel_resolution_multiplier = 1
configs.point_branch.kernel_regularizer = configs.submodel.kernel_reg

configs.cloud_branch = Config(CloudFeaturesBranch)
configs.cloud_branch.kernel_regularizer = configs.submodel.kernel_reg

configs.classification_head = Config(ClassificationHead)
configs.classification_head.num_classes = configs.data.num_classes
configs.classification_head.kernel_regularizer = configs.submodel.kernel_reg

# model
configs.model = Config(PVCNN)
configs.model.point_voxel_branch = configs.point_branch()
configs.model.cloud_features_branch = configs.cloud_branch()
configs.model.classification_head = configs.classification_head()

# dataset
configs.dataset.num_points = 4096

# metrics
configs.metrics.eval.iou.expected_shape = [None, 13, configs.dataset.num_points]
configs.metrics.train.iou.expected_shape = [None, 13, configs.dataset.num_points]  # fmt: skip

# train: scheduler
configs.train.optimizer.learning_rate = Config(CosineDecay)
configs.train.optimizer.learning_rate.initial_learning_rate = 1e-3
configs.train.optimizer.learning_rate.decay_steps = configs.train.num_epochs
