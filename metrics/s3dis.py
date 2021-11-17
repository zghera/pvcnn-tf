"""Evaluation metrics for PVCNN S3DIS dataset."""
import tensorflow as tf


class OverallAccuracy(tf.keras.metrics.Metric):
  """Mean overall accuracy metric."""

  def __init__(self, split: str, **kwargs) -> None:
    assert split in ["train", "test"]
    super().__init__(name=f"acc/overall_{split}", **kwargs)
    self._total_seen_num = self.add_weight(
      name="seen", initializer="zeros", dtype=tf.float32
    )
    self._total_correct_num = self.add_weight(
      name="correct", initializer="zeros", dtype=tf.float32
    )
    self.reset_state()

  def reset_state(self) -> None:
    self._total_seen_num = tf.Variable(0.0)
    self._total_correct_num = tf.Variable(0.0)

  def update_state(self, y_pred: tf.Tensor, y_true: tf.Tensor):
    # y_pred shape is [B, 13, num_points] | y_true shape is [B, 13, num_points]
    y_true_categ = tf.math.argmax(y_pred, axis=1)
    y_pred_categ = tf.math.argmax(y_pred, axis=1)
    tf.debugging.assert_equal(tf.size(y_true_categ), tf.size(y_pred_categ))

    self._total_seen_num.assign_add(tf.size(y_true))
    self._total_correct_num.assign_add(
      tf.reduce_sum(tf.cast(y_true_categ == y_pred_categ, tf.int32))
    )

  def result(self) -> None:
    return self._total_correct_num / self._total_seen_num


class IouAccuracy(tf.keras.metrics.MeanIoU):
  """Mean IoU accuracy metric.

  The only difference between this and the original MeanIoU Metric class
  is handling the extra dimension in y_pred that contain the probabilities
  of each class. Like the original implementation, just grab the most
  likely class to match the labels for comparision.
  """

  def __init__(self, split: str, num_classes: int, **kwargs):
    assert split in ["train", "test"]
    super().__init__(num_classes, name=f"acc/iou_{split}", **kwargs)

  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
    # y_pred shape is [B, 13, num_points] | y_true shape is [B, 13, num_points]
    y_true_categ = tf.math.argmax(y_true, axis=1)
    y_pred_categ = tf.math.argmax(y_pred, axis=1)
    super().update_state(y_true_categ, y_pred_categ)
