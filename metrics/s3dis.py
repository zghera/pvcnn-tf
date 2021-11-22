"""Evaluation metrics for PVCNN S3DIS dataset."""
import tensorflow as tf
import numpy as np
from keras import backend

class IouAccuracy(tf.keras.metrics.Metric):
  """Mean IoU accuracy metric."""

  def __init__(self, split: str, num_classes: int, **kwargs):
    assert split in ["train", "test"]
    super().__init__(name=f"acc/iou_{split}", **kwargs)
    self._num_classes = num_classes
    self._total_seen = self.add_weight(
      name="seen", shape=(num_classes,), initializer="zeros", dtype=tf.float32
    )
    self._total_positive = self.add_weight(
      name="positive",
      shape=(num_classes,),
      initializer="zeros",
      dtype=tf.float32,
    )
    self._total_correct = self.add_weight(
      name="correct",
      shape=(num_classes,),
      initializer="zeros",
      dtype=tf.float32,
    )

  def reset_state(self) -> None:
    backend.batch_set_value(
      [
        (v, np.zeros(v.shape.as_list()))
        for v in [self._total_seen, self._total_positive, self._total_correct]
      ]
    )

  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
    # y_pred shape is [B, 13, num_points] | y_true shape is [B, 13, num_points]
    y_true_categ = tf.reshape(
      tf.math.argmax(y_true, axis=1, output_type=tf.int32), [-1, 1]
    )
    y_pred_categ = tf.reshape(
      tf.math.argmax(y_pred, axis=1, output_type=tf.int32), [-1, 1]
    )

    equal = tf.squeeze(tf.cast(y_true_categ == y_pred_categ, dtype=tf.float32))
    updates = tf.ones(shape=(y_true_categ.shape[0],), dtype=tf.float32)
    shape = tf.constant([self._num_classes])

    self._total_seen.assign_add(tf.scatter_nd(y_true_categ, updates, shape))
    self._total_positive.assign_add(tf.scatter_nd(y_pred_categ, updates, shape))
    self._total_correct.assign_add(tf.scatter_nd(y_pred_categ, equal, shape))

  def result(self) -> None:
    # Set IOU to 1 for class where num_seen = 0
    iou = tf.cast(self._total_seen == 0, tf.float32)
    iou = iou + tf.math.divide_no_nan(
      self._total_correct,
      (self._total_seen + self._total_positive - self._total_correct),
    )
    # tf.print("IOU accuracy for each class:", iou, summarize=-1)
    return tf.reduce_sum(iou) / self._num_classes
