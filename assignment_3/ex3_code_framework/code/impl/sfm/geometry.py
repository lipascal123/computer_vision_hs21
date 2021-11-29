import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  """

  Input:
  - K: Camera intrinsics
  - im1: Image 1 as image class object
  - im2: Image 2 as image class object
  - matches: nx2 array, entries correspond to the indices of matching keypoints of im1 and im2
  """
  # TODO
  # Normalize coordinates (to points on the normalized image plane)
  # These are the keypoints on the normalized image plane (not to be confused with the normalization in the calibration exercise)
  
  # get keypoints of image 1 and 2
  kp1 = im1.kps
  # print("kp1: ",kp1[0:10,:])
  kp2 = im2.kps
  # make keypoints homogenous
  one1 = np.ones((kp1.shape[0],1))
  one2 = np.ones((kp2.shape[0],1))
  kp1H = np.concatenate((kp1, one1), axis = 1)
  # print("kp1H: ",kp1H[0:10,:])
  kp2H = np.concatenate((kp2, one2), axis = 1)
  # project keypoints to normalized image plane
  
  K_inv = np.linalg.inv(K)
  normalized_kps1 = (np.dot( K_inv, kp1H.T)).T
  # print("normalized_kps1:", normalized_kps1[0:10,:])  
  normalized_kps2 = (np.dot( K_inv, kp2H.T)).T

  # TODO
  # Assemble constraint matrix
  constraint_matrix = np.zeros((matches.shape[0], 9))
  
  for i in range(matches.shape[0]):
    # TODO
    # Add the constraints
    u_i = normalized_kps1[matches[i,0],0]
    v_i = normalized_kps1[matches[i,0],1]
    u_pi = normalized_kps2[matches[i,1],0]
    v_pi = normalized_kps2[matches[i,1],1]
    constraint_matrix[i,:] = [u_i*u_pi, u_i*v_pi, u_i, v_i*u_pi, v_i*v_pi, v_i, u_pi, v_pi, 1]

  
  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1,:]
  # print("vectorized_E_hat",vectorized_E_hat)

  # TODO
  # Reshape the vectorized matrix to it's proper shape again
  E_hat = np.reshape(vectorized_E_hat,(3,3))
  # print("E_hat", E_hat)

  # TODO
  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily
  U, s, Vh = np.linalg.svd(E_hat)
  # set last singular value to zero
  S = np.diag([1,1,0])
  # reassemble E
  E = np.dot(np.dot(U,S),Vh)

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[matches[i,0],:]
    kp2 = normalized_kps2[matches[i,1],:]
    
    assert(abs(kp1.transpose() @ E @ kp2) < 0.01)

  return E


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)


  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]


  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  # TODO
  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`
  idx_behind = np.argwhere(points3D[:,2]<0)
  points3D = points3D[points3D[:,2]>=0,:]
  im1_corrs = np.delete(im1_corrs, idx_behind)
  im2_corrs = np.delete(im2_corrs, idx_behind)

  return points3D, im1_corrs, im2_corrs

def EstimateImagePose(points2D, points3D, K):  

  # TODO
  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.
  K_inv = np.linalg.inv(K)
  points2DH = np.concatenate( ( points2D, np.ones((points2D.shape[0],1)) ), axis=1)
  normalized_points2D = (np.dot(K_inv, points2DH.T)).T[:,0:2]

  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t

def TriangulateImage(K, image_name, images, registered_images, matches):
  
  points3D = np.zeros((0,3))
  corrs = {}
  im_corrs_new = []
  # TODO 
  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered images

  # Query every regeistered image and find matching kp with image_name
  for query_image in registered_images:
    pair_matches = GetPairMatches(image_name, query_image, matches)
    
    # Triangulate matches. Only the not exisiting 3D points are triangulated and returned
    new_points3D, im_corrs, imquery_corrs = TriangulatePoints(K, images[image_name], images[query_image], pair_matches)

    # It could be that newly found new_points3D have already been found in previous iterations of this for loop
    # Thus, keep track of im_corrs over all loops and only add 3d point when not already seen.
    # Always update imquery_coors even 3d point has been seen before
    imquery_corrs_new = []
    idx_imquery = []
    for i in range(len(im_corrs)):

      # only add corr to current image and a new 3d point WHEN not seen before
      if im_corrs[i] not in im_corrs_new: 
        points3D = np.append(points3D, np.expand_dims(new_points3D[i],0), axis=0)
        im_corrs_new.append(im_corrs[i])
        # For 2d-3d corr via idx
        idx_imquery.append(points3D.shape[0]-1)
      else:
        # If 3d point has been seen before, find idx
        tuple_temp = np.where( (points3D == new_points3D[i]).all(axis=1) )
        if tuple_temp:
          idx_imquery.append( np.where( (points3D == new_points3D[i]).all(axis=1) )[0] )
          

      # Add every new correspondance query image
      imquery_corrs_new.append(imquery_corrs[i])
    
    corrs[query_image] = np.array([idx_imquery, imquery_corrs_new]).T
        
  # store corr of image in dictionary with indes of points3D
  idx_im = np.arange(0,len(im_corrs_new),1)
  corrs[image_name] = np.array([idx_im, im_corrs_new]).T
   


  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.

  return points3D, corrs