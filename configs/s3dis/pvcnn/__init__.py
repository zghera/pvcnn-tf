"""Experiment configurations for S3DIS PVCNN model."""
# TODO: Is TF's CosineDecay much different from Pytorch's equivalent?
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.regularizers import L2

from modeling import PVCNN, AttentionSchedule
from utils.config import Config, configs

# model
configs.model = Config(PVCNN)
configs.model.point_voxel_blocks = (
  (64, 1, 32),
  (64, 2, 16),
  (128, 1, 16),
  (1024, 1, None),
)
configs.model.voxel_resolution_multiplier = 1
configs.model.num_classes = configs.data.num_classes
configs.model.kernel_regularizer = L2(1e-5)

# dataset
configs.dataset.num_points = 4096

# train: scheduler
configs.train.optimizer.learning_rate = Config(AttentionSchedule)
configs.train.optimizer.learning_rate.d_model = 100
configs.train.optimizer.learning_rate.warmup_steps = 1000
configs.train.optimizer.learning_rate.eps = 1e-8
