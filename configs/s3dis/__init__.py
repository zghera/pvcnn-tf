"""Experiment configurations for S3DIS dataset."""
import tensorflow as tf
from dataloaders.s3dis import DatasetS3DIS

from metrics.s3dis import IouAccuracy
from utils.config import Config, configs

# data
configs.data = Config()
configs.data.num_classes = 13

# evaluation
configs.eval = Config()
configs.eval.is_evaluating = False
configs.eval.batch_size = 10

# training
configs.train = Config()
configs.train.restart_training = False
configs.train.num_epochs = 50
configs.train.batch_size = 32
configs.train.loss_fn = Config(tf.keras.losses.CategoricalCrossentropy)
configs.train.loss_fn.axis = 1
configs.train.loss_fn.reduction = tf.keras.losses.Reduction.SUM
configs.train.optimizer = Config(tf.keras.optimizers.Adam)

# dataset
configs.dataset = Config(DatasetS3DIS)
configs.dataset.data_dir = "data/s3dis/pointcnn"
configs.dataset.shuffle_size = 1000
configs.dataset.batch_size = None  # Set in train.py
configs.dataset.use_normalized_coords = True
configs.dataset.is_deterministic = configs.deterministic
configs.dataset.num_classes = configs.data.num_classes
configs.dataset.seed = configs.seed

# metrics
configs.metrics = Config()
configs.metrics.eval = Config()
configs.metrics.eval.overall = Config(tf.keras.metrics.CategoricalAccuracy)
configs.metrics.eval.overall.name = "acc/overall_test"
configs.metrics.eval.iou = Config(IouAccuracy)
configs.metrics.eval.iou.split = "test"
configs.metrics.eval.iou.num_classes = configs.data.num_classes
configs.metrics.train = Config()
configs.metrics.train.overall = Config(tf.keras.metrics.CategoricalAccuracy)
configs.metrics.train.overall.name = "acc/overall_train"
configs.metrics.train.iou = Config(IouAccuracy)
configs.metrics.train.iou.split = "train"
configs.metrics.train.iou.num_classes = configs.data.num_classes
