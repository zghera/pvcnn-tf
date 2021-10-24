"""PVCNN S3DIS Training"""
import argparse
import os
import random
import shutil

import tensorflow as tf
import numpy as np
import tensorboard

# import torch
# import torch.backends.cudnn as cudnn
# from torch.utils.data import DataLoader
# from tqdm import tqdm

from utils.common import get_save_path, set_cuda_visible_devices


def get_configs():
  """Return Config object after updating from cmd line arguments."""
  from utils.config import configs  # pylint: disable=import-outside-toplevel

  parser = argparse.ArgumentParser()
  parser.add_argument("configs", nargs="+")
  parser.add_argument("--devices", default=None)
  parser.add_argument("--evaluate", default=False, action="store_true")
  args, opts = parser.parse_known_args()
  if args.devices is not None and args.devices != "cpu":
    gpus = set_cuda_visible_devices(args.devices)
  else:
    gpus = []

  print(f"==> loading configs from {args.configs}")
  configs.update_from_modules(*args.configs)
  # define save path
  configs.train.save_path = get_save_path(*args.configs, prefix="runs")

  # override configs with args
  configs.update_from_arguments(*opts)
  if len(gpus) == 0:
    configs.device = "cpu"
    configs.device_ids = []
  else:
    configs.device = "cuda"
    configs.device_ids = gpus
  if args.evaluate and configs.evaluate.fn is not None:
    if "dataset" in configs.evaluate:
      for k, v in configs.evaluate.dataset.items():
        configs.dataset[k] = v
  else:
    configs.evaluate = None

  if configs.evaluate is None:
    metrics = []
    if "metric" in configs.train and configs.train.metric is not None:
      metrics.append(configs.train.metric)
    if "metrics" in configs.train and configs.train.metrics is not None:
      for m in configs.train.metrics:
        if m not in metrics:
          metrics.append(m)
    configs.train.metrics = metrics
    configs.train.metric = None if len(metrics) == 0 else metrics[0]

    save_path = configs.train.save_path
    configs.train.checkpoint_path = os.path.join(save_path, "latest.pth.tar")
    configs.train.checkpoints_path = os.path.join(
      save_path, "latest", "e{}.pth.tar"
    )
    configs.train.best_checkpoint_path = os.path.join(
      configs.train.save_path, "best.pth.tar"
    )
    best_checkpoints_dir = os.path.join(save_path, "best")
    configs.train.best_checkpoint_paths = {
      m: os.path.join(
        best_checkpoints_dir,
        "best.{}.pth.tar".format(m.replace("/", ".")),
      )
      for m in configs.train.metrics
    }
    os.makedirs(os.path.dirname(configs.train.checkpoints_path), exist_ok=True)
    os.makedirs(best_checkpoints_dir, exist_ok=True)
  else:
    if (
      "best_checkpoint_path" not in configs.evaluate
      or configs.evaluate.best_checkpoint_path is None
    ):
      if (
        "best_checkpoint_path" in configs.train
        and configs.train.best_checkpoint_path is not None
      ):
        configs.evaluate.best_checkpoint_path = (
          configs.train.best_checkpoint_path
        )
      else:
        configs.evaluate.best_checkpoint_path = os.path.join(
          configs.train.save_path, "best.pth.tar"
        )
    assert configs.evaluate.best_checkpoint_path.endswith(".pth.tar")
    configs.evaluate.predictions_path = (
      configs.evaluate.best_checkpoint_path.replace(".pth.tar", ".predictions")
    )
    configs.evaluate.stats_path = configs.evaluate.best_checkpoint_path.replace(
      ".pth.tar", ".eval.npy"
    )

  return configs


def main():
  configs = get_configs()
  if configs.evaluate is not None:
    configs.evaluate.fn(configs)
    return


if __name__ == "__main__":
  main()
