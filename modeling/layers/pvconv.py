"""TODO"""

class PVConv:
  """TODO


    features: Point cloud features. Float[B, C, N] where B is batch size, N is
      number of points in the point cloud, and C is 9 or 6 based on
      `configs.dataset.use_normalized_coords`. See `dataloaders/s3dis.py` for
      more details.
    coords: Coords of points. Float[B, 3, N] created by slicing `features`
      such that coords = features[:, :3, :].
  """
  pass
