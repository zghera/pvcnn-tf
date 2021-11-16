"""PVCNN S3DIS Training."""
from typing import Tuple, Iterator

import os
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from utils.common import get_save_path


def get_configs():
  """Return Config object after updating from cmd line arguments."""
  from utils.config import configs  # pylint: disable=import-outside-toplevel

  parser = argparse.ArgumentParser()
  parser.add_argument("configs", nargs="+")
  parser.add_argument("--restart", default=False, action="store_true")
  parser.add_argument("--eval", default=False, action="store_true")
  args, opts = parser.parse_known_args()

  print(f"==> loading configs from {args.configs}")
  configs.update_from_modules(*args.configs)

  # define save path
  configs.train.save_path = get_save_path(*args.configs, prefix="runs")

  # override configs with args
  configs.update_from_arguments(*opts)
  configs.eval.is_evaluating = args.eval
  configs.train.restart_training = args.restart
  assert configs.train.restart_training != configs.eval.is_evaluating

  save_path = configs.train.save_path
  configs.train.train_ckpts_path = os.path.join(save_path, "training_ckpts")
  configs.train.best_ckpt_path = os.path.join(save_path, "best_ckpt")

  if configs.eval.is_evaluating:
    batch_size = configs.eval.batch_size
  else:
    batch_size = configs.train.batch_size
    if configs.train.restart_training:
      os.makedirs(configs.train.train_ckpts_path, exist_ok=False)
      os.makedirs(configs.train.best_ckpt_path, exist_ok=False)

  configs.dataset.batch_size = batch_size

  return configs


class Train:
  """Train class."""

  def __init__(
    self,
    epochs: int,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss,
    train_step: tf.Variable,
    progress_ckpt_manager: tf.train.CheckpointManager,
    best_ckpt_manager: tf.train.CheckpointManager,
    train_overall_metric: tf.keras.metrics.Metric,
    train_iou_metric: tf.keras.metrics.Metric,
    eval_overall_metric: tf.keras.metrics.Metric,
    eval_iou_metric: tf.keras.metrics.Metric,
    best_ckpt_metric: tf.keras.metrics.Metric,
  ) -> None:
    self.epochs = epochs
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.train_step_idx = train_step
    self.progress_manager = progress_ckpt_manager
    self.best_manager = best_ckpt_manager
    self.train_overall_acc_metric = train_overall_metric
    self.train_iou_acc_metric = train_iou_metric
    self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    self.eval_overall_acc_metric = eval_overall_metric
    self.eval_iou_acc_metric = eval_iou_metric
    self.eval_loss_metric = tf.keras.metrics.Mean(name="eval_loss")
    self.best_ckpt_metric = best_ckpt_metric
    self._best_metric_val = None
    self.autotune = tf.data.experimental.AUTOTUNE

  def _save_train_checkpoint(self) -> None:
    """Save training checkpoint."""
    save_path = self.progress_manager.save(check_interval=True)
    print(f"Saved checkpoint for step {int(self.train_step_idx)}: {save_path}")

    self.train_step_idx.assign_add(1)

  def _save_if_best_checkpoint(self) -> None:
    """Save training checkpoint if best model so far."""
    cur_metric = self.best_ckpt_metric.result()
    if self._best_metric_val is None:
      self._best_metric_val = cur_metric
    elif tf.math.greater(cur_metric, self._best_metric_val):
      self._best_metric_val = cur_metric
      save_path = self.best_manager.save()
      print(f"NEW BEST checkpoint. Saved to {save_path}")

  @tf.function
  def train_step(self, sample: tf.Tensor, label: tf.Tensor) -> None:
    """One train step."""
    with tf.GradientTape() as tape:
      predictions = self.model(sample, training=True)
      loss = self.loss_fn(label, predictions)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(
      zip(gradients, self.model.trainable_variables)
    )

    self.train_loss_metric.update_state(loss)
    self.train_overall_acc_metric.update_state(label, predictions)
    self.train_iou_acc_metric.update_state(label, predictions)

    self._save_train_checkpoint()

  @tf.function
  def test_step(self, sample: tf.Tensor, label: tf.Tensor) -> None:
    """One test step."""
    predictions = self.model(sample, training=False)
    loss = self.loss_object(label, predictions)

    self.eval_loss_metric.update_state(loss)
    self.eval_overall_acc_metric.update_state(label, predictions)
    self.eval_iou_acc_metric.update_state(label, predictions)

  def train(
    self,
    train_dataset_it: Iterator[tf.Tensor],
    test_dataset_it: Iterator[tf.Tensor],
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Custom training loop."""
    for epoch in tqdm(range(self.epochs), desc="training"):
      for x, y in tqdm(train_dataset_it, desc=f"epoch {epoch}: train"):
        self.train_step(x, y)
      for x, y in tqdm(test_dataset_it, desc=f"epoch {epoch}: validation"):
        self.test_step(x, y)

      print(
        f"Epoch {epoch}:\n"
        f"--------------\n"
        f"Train:\n"
        f" - Loss: {self.train_loss_metric.result()}\n"
        f" - Overall Accuracy: {self.train_overall_acc_metric.result()}\n"
        f" - IOU Accuracy: {self.train_iou_acc_metric.result()}\n"
        f"Validation:\n"
        f" - Loss: {self.eval_loss_metric.result()}\n"
        f" - Overall Accuracy: {self.eval_overall_acc_metric.result()}\n"
        f" - IOU Accuracy: {self.eval_iou_acc_metric.result()}\n\n"
      )
      self._save_if_best_checkpoint()
      if epoch != self.epochs - 1:
        self.train_loss_metric.reset_states()
        self.train_overall_acc_metric.reset_states()
        self.train_iou_acc_metric.reset_states()
        self.eval_loss_metric.reset_states()
        self.eval_overall_acc_metric.reset_states()
        self.eval_iou_acc_metric.reset_states()

    return (
      self.train_loss_metric.result().numpy(),
      self.train_overall_acc_metric.result().numpy(),
      self.train_iou_acc_metric.result().numpy(),
      self.eval_loss_metric.result().numpy(),
      self.eval_overall_acc_metric.result().numpy(),
      self.eval_iou_acc_metric.result().numpy(),
    )

  # TODO: Need to do this more accurately like orig implementation?
  def eval(
    self, test_dataset_it: Iterator[tf.Tensor]
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Custom model evaluation function."""
    for x, y in tqdm(test_dataset_it, desc="evaluation"):
      self.test_step(x, y)

    print(
      f"Evaluation Results:\n"
      f" - Loss: {self.eval_loss_metric.result()}\n"
      f" - Overall Accuracy: {self.eval_overall_acc_metric.result()}\n"
      f" - IOU Accuracy: {self.eval_iou_acc_metric.result()}\n\n"
    )
    return (
      self.eval_loss_metric.result().numpy(),
      self.eval_overall_acc_metric.result().numpy(),
      self.eval_iou_acc_metric.result().numpy(),
    )


def main():
  #################
  # Configuration #
  #################
  configs = get_configs()
  tf.random.set_seed(configs.seed)
  np.random.seed(configs.seed)
  # Use channels first format for ease of comparing shapres with original impl.
  tf.keras.backend.set_image_data_format("channels_first")
  print("------------ Configuration ------------")
  print(configs)
  print("---------------------------------------")

  ############################################################
  # Initialize Dataset(s), Model, Optimizer, & Loss Function #
  ############################################################
  print(f'\n==> loading dataset "{configs.dataset}"')
  dataset = configs.dataset()
  print(dataset)
  train_dataset_it = iter(dataset["train"])
  print(next(train_dataset_it))
  test_dataset_it = iter(dataset["test"])
  return

  print(f'\n==> creating model "{configs.model}"')
  loss_fn = configs.train.loss_fn()
  model = None
  optimizer = None

  # Init training checkpoint objs to determine how we initialize training objs
  cur_step = tf.Variable(1)
  checkpoint = tf.train.Checkpoint(
    step=cur_step,
    model=model,
    optimizer=optimizer,
    train_dataset_it=train_dataset_it,
    test_dataset_it=test_dataset_it,
  )
  progress_manager = tf.train.CheckpointManager(
    checkpoint,
    directory=configs.train.train_ckpts_path,
    max_to_keep=3,
    step_counter=cur_step,
    checkpoint_interval=5,  # TODO: Tune this based on how long train step takes
  )
  best_manager = tf.train.CheckpointManager(
    checkpoint,
    directory=configs.eval.best_ckpt_path
    if configs.eval.is_evaluating
    else configs.train.best_ckpt_path,
    max_to_keep=1,
  )
  if configs.train.restart_training:
    model = configs.model()
    optimizer = configs.train.optimizer()
  elif configs.eval.is_evaluating:
    checkpoint.restore(best_manager.latest_checkpoint).assert_consumed()
  else:
    # Training and resuming progress from last created checkpoint
    checkpoint.restore(progress_manager.latest_checkpoint).assert_consumed()

  ############
  # Training #
  ############
  print("Training...")
  train_obj = Train(
    epochs=configs.train.num_epochs,
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_step=cur_step,
    progress_ckpt_manager=progress_manager,
    best_ckpt_manager=best_manager,
    train_overall_metric=configs.train.overall,
    train_iou_metric=configs.train.iou,
    eval_overall_metric=configs.eval.overall,
    eval_iou_metric=configs.eval.iou,
    best_ckpt_metric=configs.train.best_ckpt_metric,
  )
  if configs.eval.is_evaluating:
    return train_obj.eval(test_dataset_it)
  return train_obj.train(train_dataset_it, test_dataset_it)


if __name__ == "__main__":
  main()
