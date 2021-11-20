"""PVCNN S3DIS Training."""
from typing import Tuple, Iterator, Dict, Optional

import os
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.common import get_save_path, config_gpu

MetricsDict = Dict[str, tf.keras.metrics.Metric]


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
  assert (
    not configs.train.restart_training or not configs.eval.is_evaluating
  ), "Cannot set '--restart' and '--eval' flag at the same time."

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
    else:
      assert os.path.exists(
        configs.train.train_ckpts_path
      ), f"Training without '--restart' flag set but {configs.train.train_ckpts_path} path does not exist."
      assert os.path.exists(
        configs.train.best_ckpts_path
      ), f"Training without '--restart' flag set but {configs.train.best_ckpts_path} path does not exist."

  configs.dataset.batch_size = batch_size

  return configs

class NanLoss(Exception): pass
class NanGradients(Exception): pass

class Train:
  """Train class."""

  def __init__(
    self,
    epochs: int,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss,
    train_epoch: tf.Variable,
    train_iter_in_epoch: tf.Variable,
    progress_ckpt_manager: tf.train.CheckpointManager,
    best_ckpt_manager: tf.train.CheckpointManager,
    train_overall_metric: tf.keras.metrics.Metric,
    train_iou_metric: tf.keras.metrics.Metric,
    eval_overall_metric: tf.keras.metrics.Metric,
    eval_iou_metric: tf.keras.metrics.Metric,
    saved_metrics: MetricsDict,
  ) -> None:
    self.epochs = epochs
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.train_epoch = train_epoch
    self.train_iter_in_epoch = train_iter_in_epoch
    self.progress_manager = progress_ckpt_manager
    self.best_manager = best_ckpt_manager
    self.train_overall_acc_metric = train_overall_metric
    self.train_iou_acc_metric = train_iou_metric
    self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    self.eval_overall_acc_metric = eval_overall_metric
    self.eval_iou_acc_metric = eval_iou_metric
    self.eval_loss_metric = tf.keras.metrics.Mean(name="eval_loss")
    self._smallest_val_loss = None
    self.saved_metrics = saved_metrics
    self.autotune = tf.data.experimental.AUTOTUNE

  def _save_if_best_checkpoint(self, epoch: int) -> None:
    """Save training checkpoint if best model so far."""
    cur_val_loss = self.eval_loss_metric.result().numpy()
    if self._smallest_val_loss is None:
      self._smallest_val_loss = cur_val_loss
    if cur_val_loss < self._smallest_val_loss:
      self._smallest_val_loss = cur_val_loss
      save_path = self.best_manager.save(checkpoint_number=epoch)
      print(f"NEW BEST checkpoint at epoch {epoch}! Saved to {save_path}\n\n")

  def _print_training_results(
    self, epoch: int, iter_in_epoch: Optional[int] = None
  ):
    it_str = "" if iter_in_epoch is None else f" | Iteration {iter_in_epoch}"
    # fmt: off
    print(
      f"\nTraining Results | Epoch {epoch}{it_str}:\n"
      f"--------------\n"
      f"Train:\n"
      f" - Loss: {self.train_loss_metric.result().numpy()}\n"
      f" - Overall Accuracy: {self.train_overall_acc_metric.result().numpy() * 100}\n" # pylint: disable=line-too-long
      f" - IOU Accuracy: {self.train_iou_acc_metric.result().numpy() * 100}\n"
      f"Validation:\n"
      f" - Loss: {self.eval_loss_metric.result().numpy()}\n"
      f" - Overall Accuracy: {self.eval_overall_acc_metric.result().numpy() * 100}\n"  # pylint: disable=line-too-long
      f" - IOU Accuracy: {self.eval_iou_acc_metric.result().numpy() * 100}\n\n"
    )
    # fmt: on

  def _save_train_metrics(self):
    self.saved_metrics["train_loss"].append(
      self.train_loss_metric.result().numpy()
    )
    self.saved_metrics["train_overall_acc"].append(
      self.train_overall_acc_metric.result().numpy() * 100
    )
    self.saved_metrics["train_iou_acc"].append(
      self.train_iou_acc_metric.result().numpy() * 100
    )

  def _save_val_metrics(self):
    self.saved_metrics["val_loss"].append(self.eval_loss_metric.result())
    self.saved_metrics["val_overall_acc"].append(
      self.eval_overall_acc_metric.result().numpy() * 100
    )
    self.saved_metrics["val_iou_acc"].append(
      self.eval_iou_acc_metric.result().numpy() * 100
    )

  def _save_metrics(self):
    self._save_train_metrics()
    self._save_val_metrics()

  def _reset_metrics(self):
    self.train_loss_metric.reset_state()
    self.train_overall_acc_metric.reset_state()
    self.train_iou_acc_metric.reset_state()
    self.eval_loss_metric.reset_state()
    self.eval_overall_acc_metric.reset_state()
    self.eval_iou_acc_metric.reset_state()

  # TODO: Delete debugging prints later
  @tf.function
  def train_step(
    self, sample: tf.Tensor, label: tf.Tensor, batches_per_epoch: int
  ) -> None:
    """One train step."""
    with tf.GradientTape() as tape:
      predictions = self.model(sample, training=True)
      # loss = self.loss_fn(label, predictions)
      ######################### Debugging #########################
      # tf.print(f"\nlabel shape = {label.shape} | prediction shape = {predictions.shape}")
      tf.print("\npredictions")
      # tf.print(predictions)
      tf.print("  prediction nans =", tf.where(tf.math.is_nan(predictions)))
      tf.print("  prediction first value = ", predictions[0,0,0])
      tf.print("  prediction all same value = ", tf.reduce_all(predictions == predictions[0,0,0]))
      # tf.print("-----------------")
      # tf.print(f"\n sum prediction along axis 1 = {tf.reduce_sum(predictions, axis=1)}")
      # tf.print("---------------------------------------------------------")
      # tf.print("labels \n---------------------------------------------------------")
      # tf.print(label)
      # tf.print(f"\n labels valid one hot = {tf.reduce_all(1 == tf.reduce_sum(label, axis=1))}")
      # tf.print("---------------------------------------------------------")

      # loss = tf.keras.losses.CategoricalCrossentropy(axis=1, reduction = tf.keras.losses.Reduction.NONE)(label, predictions)
      # tf.print(f"\nNo Reduction Loss = {loss}")
      # loss = tf.keras.losses.CategoricalCrossentropy(axis=1, reduction = tf.keras.losses.Reduction.SUM)(label, predictions)
      loss = self.loss_fn(label, predictions)
      tf.print("Loss = ", loss)
      # loss = tf.debugging.assert_all_finite(loss, "loss is nan")
      # if tf.greater(tf.size(tf.where(tf.math.is_nan(loss))), 0):
      #   raise NanLoss

      #############################################################
    gradients = tape.gradient(loss, self.model.trainable_variables)
    tf.print("global gradient norm pre-clip =", tf.linalg.global_norm(gradients))
    gradients, _ = tf.clip_by_global_norm(gradients, 250000.0)
    tf.print("global gradient norm post-clip =", tf.linalg.global_norm(gradients))
    self.optimizer.apply_gradients(
      zip(gradients, self.model.trainable_variables)
    )
    # for gradient in gradients:
    #   gradient = tf.debugging.assert_all_finite(gradient, "gradient is nan")
      # if tf.greater(tf.size(tf.where(tf.math.is_nan(gradient))), 0):
      #   raise NanGradients
    ######################### Debugging #########################
    # tf.print(f"\nGradients = {gradients}")
    tf.print("\n-------------------------------------------------------------------------\n")
    #############################################################

    self.train_loss_metric.update_state(loss)
    self.train_overall_acc_metric.update_state(label, predictions)
    self.train_iou_acc_metric.update_state(label, predictions)

    def _save():
      ckpt_num = batches_per_epoch * self.train_epoch + self.train_iter_in_epoch
      self.progress_manager.save(checkpoint_number=ckpt_num)

    tf.py_function(_save, [], [])
    self.train_iter_in_epoch.assign_add(1)

  @tf.function
  def test_step(self, sample: tf.Tensor, label: tf.Tensor) -> None:
    """One test step."""
    predictions = self.model(sample, training=False)
    tf.print("\npredictions \n----------")
    # tf.print("  prediction nans =", tf.where(tf.math.is_nan(predictions)))
    tf.print("  prediction first value = ", predictions[0,0,0])
    tf.print("  prediction all same value = ", tf.reduce_all(predictions == predictions[0,0,0]))
    tf.print("-----------------\n")
    loss = self.loss_fn(label, predictions)
    tf.print("\nLoss = ", loss)
    # loss = tf.debugging.assert_all_finite(loss, "loss is nan")
    if tf.math.is_nan(loss):
      raise NanLoss

    tf.print("\n-------------------------------------------------------------------------\n")

    self.eval_loss_metric.update_state(loss)
    self.eval_overall_acc_metric.update_state(label, predictions)
    self.eval_iou_acc_metric.update_state(label, predictions)

  # TODO: Get rid of all the NaN garbage if I fix this
  def train(
    self,
    train_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    train_dataset_len: int,
    test_dataset_len: int,
  ) -> MetricsDict:
    """Custom training loop."""
    starting_iter = int(self.train_iter_in_epoch)
    for epoch in range(int(self.train_epoch), self.epochs):
      print(f"\nEpoch {epoch}:")
      for i, (x, y) in enumerate(
        tqdm(train_dataset, total=train_dataset_len, desc="Training set: ")
      ):
        if i < starting_iter: continue

        self.train_step(x, y, train_dataset_len)
        if np.isnan(self.train_loss_metric.result()):
          print(f"Failed on epoch {epoch} train step {i}: Loss was NaN.")
          return self.saved_metrics
        self._save_train_metrics()

      starting_iter = 0  # Only start part-way through epoch on 1st epoch
      self.train_iter_in_epoch.assign(0)

      for i, (x, y) in enumerate(
        tqdm(test_dataset, total=test_dataset_len, desc="Validation set: ")
      ):
        self.test_step(x, y)
        if np.isnan(self.eval_loss_metric.result()):
          print(f"Failed on epoch {epoch} val step {i}: : Loss was NaN.")
          self.saved_metrics["val_loss"] += [0] * (test_dataset_len - i)
          self.saved_metrics["val_overall_acc"] += [0] * (test_dataset_len - i)
          self.saved_metrics["val_iou_acc"] += [0] * (test_dataset_len - i)
          return self.saved_metrics
        self._save_val_metrics()

      self._print_training_results(epoch)
      self._save_if_best_checkpoint(epoch)
      # TODO: Uncomment if we can train for multiple epochs
      # self._save_metrics()
      self._reset_metrics()

      self.train_epoch.assign_add(1)

    return self.saved_metrics

  # TODO: Need to do this more accurately like orig implementation?
  def eval(
    self, test_dataset_it: Iterator[tf.Tensor], test_dataset_len: int
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Custom model evaluation function."""
    for x, y in tqdm(test_dataset_it, total=test_dataset_len):
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


def plot_train_results(train_metrics: MetricsDict, save_path: str) -> None:
  _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 10))

  # TODO: Replace 'iteration' with 'epoch' if we can train for multiple epochs
  # num_train = len(train_metrics["train_loss"])
  # sample_idx = np.linspace(0, num_train - 1, num=len(train_metrics["val_loss"]), dtype=int)

  # def get_sampled_train_metric(metric_list) -> np.array:
  #   return np.asarray(metric_list)[sample_idx]

  ax1.plot((train_metrics["train_loss"]))
  # ax1.plot(train_metrics["val_loss"])
  # ax1.legend(["Train Set", "Validation Set"], loc="upper right")
  ax1.set_title("Train Loss vs Iteration")
  ax1.set_ylabel("Loss")
  ax1.set_xlabel("Iteration")

  ax2.plot((train_metrics["train_overall_acc"]))
  ax2.plot((train_metrics["train_iou_acc"]))
  # ax2.plot(train_metrics["val_overall_acc"])
  # ax2.plot(train_metrics["val_iou_acc"])
  # ax2.legend(
  #   ["Train Overall", "Train IoU", "Validation Overall", "Validation IoU"],
  #   loc="upper right",
  # )
  # ax2.set_title("Accuracy vs Iteration")
  ax2.legend(
    ["Train Overall", "Train IoU"],
    loc="upper right",
  )
  ax2.set_title("Train Accuracy vs Iteration")
  ax2.set_ylabel("Accuracy")
  ax2.set_xlabel("Iteration")

  plot_path = os.path.join(save_path, "train-metrics-vs-epoch.png")
  plt.savefig(plot_path)


def main():
  #################
  # Configuration #
  #################
  # Use channels first format for ease of comparing shapes with original impl.
  tf.keras.backend.set_image_data_format("channels_first")
  config_gpu()
  configs = get_configs()
  tf.random.set_seed(configs.seed)
  np.random.seed(configs.seed)
  print("------------ Configuration ------------")
  print(configs)
  print("---------------------------------------")

  ############################################################
  # Initialize Dataset(s), Model, Optimizer, & Loss Function #
  ############################################################
  print(f'\n==> Loading dataset "{configs.dataset}"')
  dataset = configs.dataset()
  train_dataset, train_dataset_len = dataset["train"], dataset["train_len"]
  test_dataset, test_dataset_len = dataset["test"], dataset["test_len"]

  print(f'\n==> Creating model "{configs.model}"')
  loss_fn = configs.train.loss_fn()
  model = configs.model()
  optimizer = configs.train.optimizer()
  saved_metrics: MetricsDict = {
    "train_loss": [],
    "train_overall_acc": [],
    "train_iou_acc": [],
    "val_loss": [],
    "val_overall_acc": [],
    "val_iou_acc": [],
  }

  # Init training checkpoint objs to determine how we initialize training objs
  cur_epoch = tf.Variable(0)
  cur_iter_in_epoch = tf.Variable(0)
  checkpoint = tf.train.Checkpoint(
    cur_epoch=cur_epoch,
    cur_iter_in_epoch=cur_iter_in_epoch,
    model=model,
    optimizer=optimizer,
    saved_metrics=saved_metrics,
  )
  progress_manager = tf.train.CheckpointManager(
    checkpoint,
    directory=configs.train.train_ckpts_path,
    max_to_keep=3,
    step_counter=cur_iter_in_epoch,
    checkpoint_interval=100,
  )
  best_manager = tf.train.CheckpointManager(
    checkpoint,
    directory=configs.train.best_ckpt_path,
    max_to_keep=1,
  )
  if configs.eval.is_evaluating:
    checkpoint.restore(
      best_manager.latest_checkpoint
    ).assert_existing_objects_matched()
  elif not configs.train.restart_training:
    # Training and resuming progress from last created checkpoint
    checkpoint.restore(
      progress_manager.latest_checkpoint
    ).assert_existing_objects_matched()

  #########################
  # Training / Evaluation #
  #########################
  print("\n==> Training...")
  train_obj = Train(
    epochs=configs.train.num_epochs,
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_epoch=cur_epoch,
    train_iter_in_epoch=cur_iter_in_epoch,
    progress_ckpt_manager=progress_manager,
    best_ckpt_manager=best_manager,
    train_overall_metric=configs.metrics.train.overall(),
    train_iou_metric=configs.metrics.train.iou(),
    eval_overall_metric=configs.metrics.eval.overall(),
    eval_iou_metric=configs.metrics.eval.iou(),
    saved_metrics=saved_metrics,
  )
  if configs.eval.is_evaluating:
    train_obj.eval(test_dataset, test_dataset_len)
  train_metrics = train_obj.train(
    train_dataset, test_dataset, train_dataset_len, test_dataset_len
  )

  plot_train_results(train_metrics, configs.train.save_path)


if __name__ == "__main__":
  ################# Debugging #################
  # tf.data.experimental.enable_debug_mode()
  # tf.config.run_functions_eagerly(True)
  #############################################
  main()
