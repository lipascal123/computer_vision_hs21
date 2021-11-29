import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))
  for i in range(0,num_corrs,2):
    # TODO Add your code here
    # concat arrays
    temp1 = np.concatenate( ([0,0,0,0], -points3D[i, :].T, [-1],  points2D[i, 1]*points3D[i, :].T, [points2D[i, 1]]) )
    temp2 = np.concatenate( (points3D[i, :].T, [1], [0,0,0,0], -points2D[i, 0]*points3D[i, :].T, [-points2D[i, 0]]) )

    # build constraint matrix
    constraint_matrix[i,:] = temp1      
    constraint_matrix[i+1,:] = temp2

  return constraint_matrix
