import numpy as np
# from code.impl.calib.geometry import NormalizePoints2D
from impl.calib.geometry import NormalizePoints2D
import scipy.optimize as spo

from impl.util import MakeHomogeneous, HNormalize

# Compute the reprojection error for a single correspondence
def ReprojectionError(P, point3D, point2D):
    # TODO
    # Project the 3D point into the image and compare it to the keypoint.
    # Make sure to properly normalize homogeneous coordinates.
    #! point3D and point2D should already be normalized
    # print("point3D",point3D)
    # print("point2D",point2D)
    point3DH = MakeHomogeneous(point3D)
    # print("point3DH",point3DH)
    # print("point2DH",point2DH)
    proj = np.dot(P, point3DH)
    # print("projected", proj)
    projN = HNormalize(proj)
    # print("projectedN", projN)
    error = point2D - projN
    # print("error",error)
  
    return error

# Compute the residuals for all correspondences of the image
def ImageResiduals(P, points2D, points3D):

  num_residuals = points2D.shape[0]
  res = np.zeros(num_residuals*2)

  for res_idx in range(num_residuals):
    p3D = points3D[res_idx]
    p2D = points2D[res_idx]

    err = ReprojectionError(P, p3D, p2D)

    res[res_idx*2:res_idx*2+2] = err

  return res

# Optimize the projection matrix given the 2D-3D point correspondences.
# 2D and 3D points with the same index are assumed to correspond.
def OptimizeProjectionMatrix(P, points2D, points3D):

    # The optimization requires a scalar cost value.
    # We use the sum of squared differences of all correspondences
    f = lambda x : np.linalg.norm(ImageResiduals(np.reshape(x, (3, 4)), points2D, points3D)) ** 2
    
    # Since the projection matrix is scale invariant we have an open degree of freedom from just the constraints.
    # Make sure this is fixed by keeping the last component close to 1.
    scale_constraint = {'type': 'eq', 'fun': lambda x : x[11] - 1}

    # Make sure the scale constraint is fulfilled at the beginning
    result = spo.minimize(f, np.reshape(P / P[2,3], 12), options={'disp': True}, constraints=[scale_constraint], tol=1e-12)
    
    return np.reshape(result.x, (3, 4))
