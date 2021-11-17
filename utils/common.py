"""Other common utilities for filepaths and device info."""
from typing import List
import os

__all__ = ["get_save_path", "set_cuda_visible_devices"]


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
