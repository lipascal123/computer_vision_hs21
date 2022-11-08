import numpy as np
import matplotlib.pyplot as plt

def propagate(particles, frame_height, frame_width, params):
    """
    :param particles with shape [num_particles, num_states]
    :param frame_height the height of the box 
    :param frame_width the widt of the box
    :param params dictionary with all parameters    
    """
    
    # particles can have shape [num_particles,2] or [num_particles,4].
    # The first is with and the second without velocity states
    # Same for the model that either has [2,2] or [4,4] 
    # print("particles", particles.shape)
    # print("particles", particles[0:10,:])
    # print("frame_height", frame_height)
    # print("frame_width", frame_width)
    # print("model", params["model"])
    # print("A matrix", params)
    
    # no motion
    if params["model"]==0:
        A = np.eye(2,2)
        w = np.random.normal(loc=0.0, scale=params["sigma_position"], size=(2, particles.shape[0])) # [2,n]
        particles_prop = (np.matmul(A, particles.T) + w).T # [n,2]
  
    # constant velocity
    elif params["model"]==1:
        #! what is dt???? Shouldn't this be given somehow
        dt=1
        A = np.array([[1,0,dt,0],
                      [0,1,0,dt],
                      [0,0,1,0],
                      [0,0,0,1]]) # [4,4]
        # position noise
        w_pos = np.random.normal(loc=0.0, scale=params["sigma_position"], size=(2, particles.shape[0])) # [2,n]
        # velocity noise
        w_vel = np.random.normal(loc=0.0, scale=params["sigma_velocity"], size=(2, particles.shape[0])) # [2,n]
        # stack pos and vel noise
        w = np.vstack((w_pos,w_vel)) # [4,1]

        particles_prop = (np.matmul(A,particles.T) + w).T # [num_particles, num_states]
        # print("111",particles_prop.shape)
        
    else:
        print("WARNING: No model available")
        assert(False)

    # delete particles outside of frame
    # ensure x position >= 0
    particles_prop[particles_prop[:, 0] < 0, 0] = 0
    # ensure y position >= 0
    particles_prop[particles_prop[:, 1] < 0, 1] = 0
    # inside width
    particles_prop[particles_prop[:, 0] > frame_width-1, 0] = frame_width-1
    # inside height
    particles_prop[particles_prop[:, 1] > frame_height-1, 1] = frame_height-1

    # retrun propagated and cleaned particles
    return particles_prop 


