"""Custom operations for voxelization and devoxlelization."""
from ops.voxelization_ops.avg_vox import avg_voxelize_forward as avg_voxelize
from ops.voxelization_ops.trilinear_devox import (
  trilinear_devoxelize_forward as trilinear_devoxelize,
)
