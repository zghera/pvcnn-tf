"""Experiment configurations for S3DIS PVCNN model."""
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import CosineDecay

from models import PVCNN
from utils.config import Config, configs

# model
configs.model = Config(PVCNN)
configs.model.num_classes = configs.data.num_classes
configs.model.extra_feature_channels = 6
configs.dataset.num_points = 4096

# train: optimizer
configs.train.optimizer.weight_decay = 1e-5

# train: scheduler
configs.train.optimizer.learning_rate = Config(CosineDecay) # TODO: Is tf different?
configs.train.scheduler.learning_rate.initial_learning_rate = 1e-3
configs.train.scheduler.learning_rate.decay_steps = configs.train.num_epochs
