import numpy as np

def MakeHomogeneous(pts, ax=0):
  """
  Adds a dimension with ones i.e. makes them homogenous
  
  """
  assert(pts.ndim <= 2)
  assert(ax < pts.ndim)

  if pts.ndim == 2:
    num_pts = pts.shape[1-ax]
    if ax == 0:
      return np.append(pts, np.ones((1, num_pts)), ax)
    else:
      return np.append(pts, np.ones((num_pts, 1)), ax)

  else:
    return np.append(pts, 1)


def HNormalize(pts, ax=0):
  """
  Takes homogenous points as inputs and normalizes/scales them such that their last element is 1

  Return: normalized elements without 1 at the end  
  """
  assert(pts.ndim <= 2)
  assert(ax < pts.ndim)

  if pts.ndim == 2:
    if ax == 0:
      return pts[:-1,:] / pts[[-1], :]
    else:
      return pts[:,:-1] / pts[:, [-1]]

  else:
    return pts[:-1] / pts[-1]
