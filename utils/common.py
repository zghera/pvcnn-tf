"""Other common utilities for filepaths and device info."""
import os
import tensorflow as tf

__all__ = ["get_save_path", "config_gpu"]

def get_configs(configs_path: str, is_evaluating: bool, restart_training: bool):
  """Return Config object."""
  from utils.config import configs  # pylint: disable=import-outside-toplevel

  print(f"==> loading configs from {configs_path}")
  configs.update_from_modules(configs_path)

  # define save path
  configs.train.save_path = get_save_path(configs_path, prefix="runs")

  # override configs with args
  configs.eval.is_evaluating = is_evaluating
  configs.train.restart_training = restart_training
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
        configs.train.best_ckpt_path
      ), f"Training without '--restart' flag set but {configs.train.best_ckpt_path} path does not exist."

  configs.dataset.batch_size = batch_size

  return configs


def get_save_path(*configs, prefix: str = "runs") -> str:
  """Get string path to save model checkpoints."""
  memo = {}
  for c in configs:
    cmemo = memo
    c = c.replace("configs/", "").replace(".py", "").split("/")
    for m in c:
      if m not in cmemo:
        cmemo[m] = dict()
      cmemo = cmemo[m]

  def get_str(m, p):
    n = len(m)
    if n > 1:
      p += "["
    for i, (k, v) in enumerate(m.items()):
      p += k
      if len(v) > 0:
        p += "."
      p = get_str(v, p)
      if n > 1 and i < n - 1:
        p += "+"
    if n > 1:
      p += "]"
    return p

  return os.path.join(prefix, get_str(memo, ""))


def config_gpu():
  # TODO: The following may be useful but caused colab to crash initially
  # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
  # os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"
  gpus = tf.config.list_physical_devices("GPU")
  print(f"Num GPUs Available: {len(gpus)}")
  if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      tf.config.set_visible_devices(gpus[0], "GPU")
      logical_gpus = tf.config.list_logical_devices("GPU")
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
      # Visible devices must be set before GPUs have been initialized
      print(e)
    os.environ["TF_CUDNN_DETERMINISTIC"]="1"
  else:
    raise Exception("No GPUs available. Unable to run model.")
  print()
