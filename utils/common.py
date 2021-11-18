"""Other common utilities for filepaths and device info."""
import os
import tensorflow as tf

__all__ = ["get_save_path", "config_gpu"]


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
  gpu_devices = tf.config.list_physical_devices("GPU")
  print(f"Num GPUs Available: {len(gpu_devices)}\n")
  for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
