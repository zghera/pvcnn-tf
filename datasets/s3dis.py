"""PVCNN S3DIS Configuration and Data Pipeline"""
from typing import Dict, Any, Tuple, List
from pathlib import Path
from absl import flags
import tensorflow as tf
import tensorflow_io as tfio

FLAGS = flags.FLAGS
AUTOTUNE = tf.data.AUTOTUNE


def define_flags() -> None:
  """Defining all the necessary flags."""
  # pylint: disable=line-too-long
  # fmt: off
  flags.DEFINE_string("data_dir", "./data/s3dis/pointcnn/", "Directory to store the dataset")
  flags.DEFINE_integer("shuffle_size", 50000, "Shuffle buffer size")
  flags.DEFINE_integer("train_batch_size", 32, "Training batch size")
  flags.DEFINE_integer("eval_batch_size", 10, "Evaluation batch size")
  flags.DEFINE_integer("num_points", 8192, "Number of points to process for each scene")
  flags.DEFINE_boolean("use_normalized_coords", True, "Whether include the normalized coords in feature")
  flags.DEFINE_integer("holdout_area", 5, "Area to hold out for testing")
  # fmt: on


def get_flags_dict() -> Dict[str, Any]:
  """Returns the command line arguments as a dict."""
  kwargs = {
    "data_dir": FLAGS.data_dir,
    "shuffle_size": FLAGS.shuffle_size,
    "train_batch_size": FLAGS.train_batch_size,
    "eval_batch_size": FLAGS.eval_batch_size,
    "num_points": FLAGS.num_points,
    "use_normalized_coords": FLAGS.use_normalized_coords,
    "holdout_area": FLAGS.holdout_area,
  }
  return kwargs


def create_dataset(
  data_dir: str,
  shuffle_size: int,
  batch_size: int,
  num_points: int,
  use_normalized_coords: bool,
  holdout_area: int,
  is_train_split: bool,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Creates train or test `tf.data.Dataset`.
  Args:
    is_train_split: True if create train dataset. False if create test dataset.
    data_dir: Directory where prepared dataset is stored.
    shuffle_size: Shuffle buffer size.
    batch_size: Batch size.
    num_points: Number of points to process for each scene.
    use_normalized_coords: Whether include the normalized coords in features.
    holdout_area: Area to hold out for testing.
  """
  filenames = _get_filenames(is_train_split, data_dir, holdout_area)
  filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)

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
      deterministic=False,
    )
    for h5_dataset, spec in h5_dataset_specs.items()
  )
  dataset = tf.data.Dataset.zip(h5_datasets)
  dataset = dataset.map(
    lambda label, data_num_points, data, use_normalized_coords=use_normalized_coords, desired_num_points=num_points: _random_sample_data(
      data,
      label,
      data_num_points,
      desired_num_points,
      use_normalized_coords,
    )
  )

  dataset = dataset.cache()
  if is_train_split:
    dataset = dataset.shuffle(shuffle_size)
  dataset = dataset.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

  return dataset


def _get_filenames(
  is_train_split: bool, data_dir: str, holdout_area: int
) -> List[str]:
  """Gets the dataset filenames for the split indicated by `is_training_split`."""
  root_path = Path(data_dir)
  assert root_path.is_dir()
  areas = []
  if is_train_split:
    for a in [x for x in range(1, 7) if x != holdout_area]:
      areas.append(root_path / f"Area_{a}")
  else:
    areas.append(root_path / f"Area_{holdout_area}")

  filenames: List[str] = []
  for area in areas:
    assert area.is_dir()
    for scene in area.iterdir():
      assert scene.is_dir()
      splits = list(scene.glob("*.h5"))
      assert len(splits) == 2
      for split in splits:
        filenames.append(str(split.resolve()))

  return filenames


@tf.function
def _random_sample_data(
  data: tf.Tensor,
  label: tf.Tensor,
  data_num_points: tf.Tensor,
  desired_num_points: int,
  use_normalized_coords: bool,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Map function for dataset to get randomly sampled (data,label) examples."""
  for tensor, expected_rank in [[data, 2], [label, 1], [data_num_points, 0]]:
    tf.debugging.assert_rank(tensor, expected_rank)

  data = tf.cast(data, tf.float32)
  label = tf.cast(label, tf.int64)

  if data_num_points < desired_num_points:
    # Sample with replacement
    tf.print("sample with replacement", data_num_points, desired_num_points)
    indices = tf.random.uniform(
      shape=[desired_num_points],
      minval=0,
      maxval=data_num_points,
      dtype=tf.int32,
    )
  else:
    # Sample without replacement courtesy of https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125
    tf.print("sample without replacement", data_num_points, desired_num_points)
    logits = tf.zeros([data_num_points])  # Uniform distribution
    # pylint: disable=invalid-unary-operand-type
    z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))
    _, indices = tf.nn.top_k(logits + z, k=desired_num_points)
  tf.print("indices=", indices)

  tf.print("PRE data shape=", tf.shape(data))
  data = tf.transpose(tf.gather(data, indices=indices))
  tf.print("POST data shape=", tf.shape(data))
  tf.print("PRE label shape=", tf.shape(label))
  label = tf.gather(label, indices=indices)
  tf.print("POST label shape=", tf.shape(label))
  if use_normalized_coords:
    # data[9, num_points] =  [x_in_block, y_in_block, z_in_block, r, g, b, x / x_room, y / y_room, z / z_room]
    return data, label
  # data[6, num_points] = [x_in_block, y_in_block, z_in_block, r, g, b]
  return data[:-3, :], label
