"""Custom container classes."""


class AttrDict(dict):
  """Container that access dictionary items using attribute methods.
  From: https://github.com/vacancy/Jacinle/blob/master/jacinle/utils/container.py
  """

  def __getattr__(self, k):
    if k not in self:
      raise AttributeError(k)
    return self[k]

  def __setattr__(self, k, v):
    self[k] = v

  def __delattr__(self, k):
    del self[k]
