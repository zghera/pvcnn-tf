"""PVCNN S3DIS Configuration and Data Pipeline"""
import math
import h5py
from typing import Tuple, List
from pathlib import Path

import tensorflow as tf
import tensorflow_io as tfio

AUTOTUNE = tf.data.AUTOTUNE


def create_s3dis_dataset(
  data_dir: str,
  shuffle_size: int,
  batch_size: int,
  num_points: int,
  use_normalized_coords: bool,
  holdout_area: int,
  is_deterministic: bool,
  num_classes: int,
  seed: int,
  is_train_split: bool,
) -> Tuple[tf.data.Dataset, int]:
  """Creates train or test `tf.data.Dataset`.
  Args:
    is_train_split: True if create train dataset. False if create test dataset.
    See S3DIS.__init__ for other arguments.
  """
  filenames, num_dataset_samples = _get_file_info(
    is_train_split, data_dir, holdout_area
  )
  filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)

  dataset_len = math.ceil(num_dataset_samples / batch_size)

  h5_dataset_specs = {
    "/label_seg": tf.int32,
    "/data_num": tf.int32,
    "/data": tf.float64,
  }
  h5_datasets = tuple(
    filenames_ds.interleave(
      lambda filename, dataset=h5_dataset, spec=spec: tfio.IODataset.from_hdf5(
        filename, dataset, spec
      ),
      num_parallel_calls=AUTOTUNE,
      deterministic=is_deterministic,
    )
    for h5_dataset, spec in h5_dataset_specs.items()
  )
  dataset = tf.data.Dataset.zip(h5_datasets)
  # fmt: off
  dataset = dataset.map(
    lambda label, data_num_points, data,
           use_normalized_coords=use_normalized_coords,
           desired_num_points=num_points,
           num_classes=num_classes: _pre_process(
      data,
      label,
      data_num_points,
      desired_num_points,
      use_normalized_coords,
      num_classes,
    ),
    num_parallel_calls=AUTOTUNE,
    deterministic=is_deterministic,
  )
  # fmt: on

  if is_train_split:
    dataset = dataset.shuffle(
      shuffle_size, seed=seed, reshuffle_each_iteration=True
    )
  dataset = dataset.batch(
    batch_size,
    num_parallel_calls=AUTOTUNE,
    deterministic=is_deterministic,
  ).prefetch(buffer_size=AUTOTUNE)

  return dataset, dataset_len


def _get_file_info(
  is_train_split: bool, data_dir: str, holdout_area: int
) -> List[str]:
  """Gets the dataset filenames and total number of sampels (scene pointcoulds)
  for the split indicated by `is_training_split`."""
  root_path = Path(data_dir)
  assert root_path.is_dir()
  areas = []
  if is_train_split:
    for a in [x for x in range(1, 7) if x != holdout_area]:
      areas.append(root_path / f"Area_{a}")
  else:
    areas.append(root_path / f"Area_{holdout_area}")

  num_dataset_samples = 0
  determine_num_dataset_samples = True
  if holdout_area == 5:
    if is_train_split:
      print(
        "\nNote: It takes a really long time to look through all of the "
        "files to determine the number of samples. So for holdout area 5, "
        "use a hardcoded value based off the number of samples found when "
        "counting the data length by looping through each file (method used "
        "below when `determine_num_dataset_samples` is True).\n"
      )
    num_dataset_samples = 57216 if is_train_split else 23104
    determine_num_dataset_samples = False

  filenames: List[str] = []
  for area in areas:
    assert area.is_dir()
    for scene in area.iterdir():
      assert scene.is_dir()
      splits = list(scene.glob("*.h5"))
      assert len(splits) == 2
      for split in splits:
        filename = str(split.resolve())
        filenames.append(filename)

        if determine_num_dataset_samples:
          h5f = h5py.File(filename, "r")
          num_dataset_samples += h5f["data"].shape[0]

  return filenames, num_dataset_samples


@tf.function
def _pre_process(
  data: tf.Tensor,
  label: tf.Tensor,
  data_num_points: tf.Tensor,
  desired_num_points: int,
  use_normalized_coords: bool,
  num_classes: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Map function for dataset to get randomly sampled (data,label) examples."""
  for tensor, expected_rank in [[data, 2], [label, 1], [data_num_points, 0]]:
    tf.debugging.assert_rank(tensor, expected_rank)

  data = tf.cast(data, tf.float32)
  label = tf.cast(label, tf.int64)

  def sample_with_replacement():
    return tf.random.uniform(
      shape=[desired_num_points],
      minval=0,
      maxval=data_num_points,
      dtype=tf.int32,
      seed=0,
    )

  def sample_without_replacement():
    # Courtesy of https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125
    logits = tf.zeros([data_num_points])  # Uniform distribution
    # pylint: disable=invalid-unary-operand-type
    z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))
    _, indices = tf.nn.top_k(logits + z, k=desired_num_points)
    return indices

  # Randomly sample from pointcloud
  indices = tf.cond(
    data_num_points < desired_num_points,
    true_fn=sample_with_replacement,
    false_fn=sample_without_replacement,
  )
  data = tf.gather(data, indices=indices)
  label = tf.gather(label, indices=indices)

  # Filter out any points with class labels > num_classes or < 0
  # fmt: off
  indices = tf.squeeze(
    tf.where(tf.math.logical_and(
        tf.less_equal(label, num_classes - 1),
        tf.greater_equal(label, 0),
    )), 1)
  # fmt: on
  data = tf.gather(data, indices=indices)
  label = tf.cast(tf.gather(label, indices=indices), dtype=tf.int32)

  data = tf.transpose(data)
  # label has shape [13, num_points] where axis 0 is the one-hot encoding
  #   of the categorical labels 0 - 12.
  label = tf.one_hot(label, depth=num_classes, axis=0)
  if use_normalized_coords:
    # data has shape [9, num_points] where axis 0 is
    #   [x_in_block, y_in_block, z_in_block, r, g, b, x / x_room, y / y_room, z / z_room]
    return data, label
  # data has shape [6, num_points] where axis 0 is
  #   [x_in_block, y_in_block, z_in_block, r, g, b]
  return data[:-3, :], label


class DatasetS3DIS(dict):
  """
  Holds the train and/or test split `tf.data.Dataset` object(s) for the S3DIS
  dataset.
  """

  def __init__(
    self,
    data_dir: str,
    shuffle_size: int,
    batch_size: int,
    num_points: int,
    use_normalized_coords: bool,
    holdout_area: int,
    is_deterministic: bool,
    num_classes: int,
    seed: int,
    split=None,
  ):
    """
    Initialize the `tf.data.Dataset` object(s) for the split(s) specified.
    Args:
      data_dir: Directory where prepared dataset is stored.
      shuffle_size: Shuffle buffer size.
      batch_size: Batch size.
      num_points: Number of points to process for each scene.
      use_normalized_coords: Whether include the normalized coords in features.
      holdout_area: Area to hold out for testing.
      is_deterministic: When False, the dataset can yield elements out of order.
      seed: Random seed.
      num_classes: Number of classes in the labels.
      split: 'train', 'test', or None. None will create both the train and
             test splits.
    """
    super().__init__()
    if split is None:
      split = ["train", "test"]
    elif not isinstance(split, (list, tuple)):
      split = [split]
    for s in split:
      self[s], self[s + "_len"] = create_s3dis_dataset(
        data_dir,
        shuffle_size,
        batch_size,
        num_points,
        use_normalized_coords,
        holdout_area,
        is_deterministic,
        num_classes,
        seed,
        is_train_split=(s == "train"),
      )
