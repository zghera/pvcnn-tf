"""Demo to visualize data pipeline output."""
import matplotlib.pyplot as plt
import tensorflow as tf
from dataloaders.s3dis import create_s3dis_dataset


def main(create_pointcloud_dump: bool):
  objects = {
    0: "clutter",
    1: "ceiling",
    2: "floor",
    3: "wall",
    4: "beam",
    5: "column",
    6: "door",
    7: "window",
    8: "table",
    9: "chair",
    10: "sofa",
    11: "bookcase",
    12: "board",
  }
  dataset, _ = create_s3dis_dataset(
    "./data/s3dis/pointcnn/",
    shuffle_size=1,
    batch_size=1,
    num_points=10000,
    use_normalized_coords=False,
    holdout_area=5,
    is_train_split=False,
    is_deterministic=False,
    num_classes=13,
    seed=1,
  )

  x, y = tuple(tf.squeeze(tensor) for tensor in next(iter(dataset)))
  x = x[:3, :]
  y = tf.argmax(y, axis=0)
  print(f"sample shape = {x.shape} | label shape = {y.shape}")

  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")

  if create_pointcloud_dump:
    open("scene-data.txt", "w", encoding="UTF-8").close()
  for i, item_name in objects.items():
    if i <= 3:
      continue
    mask = tf.equal(y, tf.cast(tf.fill([y.shape[0]], i), dtype=tf.int64))
    cur_x = tf.boolean_mask(x, mask, axis=1)
    print(item_name, cur_x.shape)
    if cur_x.shape[1] > 0:
      ax.scatter(cur_x[0, :], cur_x[1, :], cur_x[2, :], label=item_name)
      if create_pointcloud_dump:
        with open("scene-data.txt", "a", encoding="UTF-8") as fp:
          for j in range(cur_x.shape[1]):
            print(f"{cur_x[0,j]} {cur_x[1,j]} {cur_x[2,j]}", file=fp)

  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")
  plt.legend()
  plt.show()
  # plt.savefig("s3dis-data-pipeline-output.png") # Use for WSL dev


if __name__ == "__main__":
  main(create_pointcloud_dump=False)
