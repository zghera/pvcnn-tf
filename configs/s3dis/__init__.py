"""Experiment configurations for S3DIS dataset."""
import tensorflow as tf
from dataloaders.s3dis import DatasetS3DIS

from metrics.s3dis import MetricS3DIS
from utils.config import Config, configs

configs.data.num_classes = 13

# dataset configs
configs.dataset = Config(DatasetS3DIS)
configs.dataset.data_dir = "data/s3dis/pointcnn"
configs.dataset.shuffle_size = 10000
configs.dataset.batch_size = 32
configs.dataset.with_normalized_coords = True
configs.dataset.is_deterministic = configs.deterministic

# test configs
configs.test = Config()
configs.test.is_testing = False

# metrics configs
configs.metrics = Config()
configs.metrics.test = Config()
configs.metrics.test.overall = Config(
  MetricS3DIS,
  metric="overall",
  split="test",
  num_classes=configs.data.num_classes,
)
configs.metrics.test.iou = Config(
  MetricS3DIS, metric="iou", split="test", num_classes=configs.data.num_classes
)
configs.metrics.train = Config()
configs.metrics.train.overall = Config(
  MetricS3DIS,
  metric="overall",
  split="train",
  num_classes=configs.data.num_classes,
)
configs.metrics.train.iou = Config(
  MetricS3DIS, metric="iou", split="train", num_classes=configs.data.num_classes
)

# train configs
configs.train = Config()
configs.train.from_scratch = False
configs.train.num_epochs = 50
# configs.train.batch_size = 32 <-- I think we set this for dataset

# train: metric for save best checkpoint
configs.train.metric = "acc/iou_test"

# train: loss
configs.train.loss_fn = Config(tf.keras.losses.CategoricalCrossentropy)

# train: optimizer
configs.train.optimizer = Config(tf.keras.optimizers.Adam)
