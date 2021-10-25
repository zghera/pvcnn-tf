"""PVCNN S3DIS Training.

TODO:
* Update methods of Train class (e.g. save checkpoints)
* test method of Train class
* Initialize objects in main()
* Verify model checkpoint saving works the way you think it does
"""
import argparse
import os

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from typing import Callable, Tuple

from utils.common import get_save_path


def get_configs():
  """Return Config object after updating from cmd line arguments."""
  from utils.config import configs  # pylint: disable=import-outside-toplevel

  parser = argparse.ArgumentParser()
  parser.add_argument("configs", nargs="+")
  parser.add_argument("--fromscratch", default=False, action="store_true")
  parser.add_argument("--test", default=False, action="store_true")
  args, opts = parser.parse_known_args()

  print(f"==> loading configs from {args.configs}")
  configs.update_from_modules(*args.configs)

  # define save path
  configs.train.save_path = get_save_path(*args.configs, prefix="runs")

  # override configs with args
  configs.update_from_arguments(*opts)
  configs.test.is_testing = args.test

  configs.train.from_scratch = args.fromscratch
  assert configs.train.from_scratch is False or configs.test.is_testing is False

  if configs.test.is_testing:
    if (
      "best_checkpoint_path" not in configs.test
      or configs.test.best_checkpoint_path is None
    ):
      if (
        "best_checkpoint_path" in configs.train
        and configs.train.best_checkpoint_path is not None
      ):
        configs.test.best_checkpoint_path = configs.train.best_checkpoint_path
      else:
        configs.test.best_checkpoint_path = os.path.join(
          configs.train.save_path, "best_ckpt"
        )
  else:
    train_metrics = []
    if "metric" in configs.train and configs.train.metric is not None:
      train_metrics.append(configs.train.metric)
    if "metrics" in configs.train and configs.train.metrics is not None:
      for m in configs.train.metrics:
        if m not in train_metrics:
          train_metrics.append(m)
    configs.train.metrics = train_metrics
    configs.train.metric = None if len(train_metrics) == 0 else train_metrics[0]

    save_path = configs.train.save_path
    configs.train.train_ckpts_path = os.path.join(save_path, "training_ckpts")
    configs.train.best_ckpt_path = os.path.join(save_path, "best_ckpt")
    if configs.train.from_scratch:
      os.makedirs(configs.train.train_ckpts_path, exist_ok=True)
      os.makedirs(configs.train.best_ckpt_path, exist_ok=True)

  return configs


class Train:
  """Train class."""

  def __init__(
    self,
    epochs: int,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss,
    lr_scheduler: Callable[[int], float],
    train_overall_metric: tf.keras.metrics.Metric,
    train_iou_metric: tf.keras.metrics.Metric,
    test_overall_metric: tf.keras.metrics.Metric,
    test_iou_metric: tf.keras.metrics.Metric,
  ) -> None:
    self._epochs = epochs
    self._model = model
    self._optimizer = optimizer
    self._loss_fn = loss_fn
    self._lr_scheduler = lr_scheduler
    self._train_overall_acc_metric = train_overall_metric
    self._train_iou_acc_metric = train_iou_metric
    self._train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    self._test_overall_acc_metric = test_overall_metric
    self._test_iou_acc_metric = test_iou_metric
    self._test_loss_metric = tf.keras.metrics.Mean(name="test_loss")
    self.autotune = tf.data.experimental.AUTOTUNE

  @tf.function
  def train_step(self, sample: tf.Tensor, label: tf.Tensor) -> None:
    """One train step."""
    with tf.GradientTape() as tape:
      predictions = self.model(sample, training=True)
      loss = self.loss_object(label, predictions)
      loss += sum(self.model.losses)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(
      zip(gradients, self.model.trainable_variables)
    )

    self.train_loss_metric(loss)
    self.train_acc_metric(label, predictions)

  @tf.function
  def test_step(self, sample: tf.Tensor, label: tf.Tensor) -> None:
    """One test step."""
    predictions = self.model(sample, training=False)
    loss = self.loss_object(label, predictions)

    self.test_loss_metric(loss)
    self.test_acc_metric(label, predictions)

  def train_loop(
    self, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
    """Custom training and testing loop.
    Args:
      train_dataset: Training dataset
      test_dataset: Testing dataset
    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """
    for epoch in tqdm(range(self.epochs), desc=f"training"):
      self.optimizer.learning_rate = self._lr_scheduler(epoch)

      for x, y in tqdm(train_dataset, desc=f"epoch {epoch}: train"):
        self.train_step(x, y)
      for x, y in tqdm(test_dataset, desc=f"epoch {epoch}: validation"):
        self.test_step(x, y)

      print(
        f"Epoch {epoch}:\n"
        f"--------------\n"
        f"Train:\n"
        f" - Loss: {self._train_loss_metric.result()}\n"
        f" - Overall Accuracy: {self._train_overall_acc_metric.result()}\n"
        f" - IOU Accuracy: {self._train_iou_acc_metric.result()}\n"
        f"Validation:\n"
        f" - Loss: {self._test_loss_metric.result()}\n"
        f" - Overall Accuracy: {self._test_overall_acc_metric.result()}\n"
        f" - IOU Accuracy: {self._test_iou_acc_metric.result()}\n\n"
      )
      if epoch != self.epochs - 1:
        self.train_loss_metric.reset_states()
        self.train_acc_metric.reset_states()
        self.test_loss_metric.reset_states()
        self.test_acc_metric.reset_states()

    return (
      self.train_loss_metric.result().numpy(),
      self.train_acc_metric.result().numpy(),
      self.test_loss_metric.result().numpy(),
      self.test_acc_metric.result().numpy(),
    )

  def test(self, some_args_go_here):
    return


def main():
  #################
  # Configuration #
  #################
  configs = get_configs()
  tf.random.set_seed(configs.seed)
  np.random.seed(configs.seed)
  print("------------ Configuration ------------")
  print(configs)
  print("---------------------------------------")

  ####################################################
  # Initialize Dataset(s), Model, Optimizer, TODO... #
  ####################################################
  print(f'\n==> loading dataset "{configs.dataset}"')
  dataset = configs.dataset()

  print(f'\n==> creating model "{configs.model}"')
  loss_fn = configs.train.loss_fn()
  model = None
  optimizer = None

  if configs.train.from_scratch:
    model = configs.model(...)
    optimizer = configs.train.optimizer(..., configs.train.optimizer.lr)
  else:
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager: tf.train.CheckpointManager
    if configs.test.is_testing:
      manager = tf.train.CheckpointManager(
        checkpoint, directory=configs.test.best_checkpoint_path
      )
    else:
      # Will resume model state if we have checkpoints in the directory
      # specified by `configs.train.train_ckpts_path`
      manager = tf.train.CheckpointManager(
        checkpoint, directory=configs.train.train_ckpts_path, max_to_keep=5
      )
    checkpoint.restore(manager.latest_checkpoint).assert_consumed()

  ############
  # Training #
  ############
  print("Training...")
  train_obj = Train(...)
  if configs.test.is_testing:
    return train_obj.train_loop(...)
  return train_obj.test(...)


if __name__ == "__main__":
  main()
