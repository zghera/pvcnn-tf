import argparse
import tensorflow as tf

from modeling import PVCNN

import os
from utils.common import get_configs

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint_dir", nargs="+")
  args, _ = parser.parse_known_args()

  path = args.checkpoint_dir[0]
  configs = get_configs(path, is_evaluating=True, restart_training=False)

  # model = PVCNN([], 0, None, 0, 0)
  # checkpoint = tf.train.Checkpoint(
  #   model=model,
  # )
  # manager = tf.train.CheckpointManager(
  #   checkpoint, directory=path, max_to_keep=1
  # )
  # checkpoint.restore(
  #   manager.latest_checkpoint
  # ).assert_existing_objects_matched()
  # print(model)


if __name__ == "__main__":
  main()
