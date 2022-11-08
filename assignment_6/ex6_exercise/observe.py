import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost


def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    """
    :param particles [n,2] matrix with all particles
    :param frame current image of video stream [width, height]
    :param bbox_height heigt of bounding box centered around pixel
    :param bbox_width width of bounding box centered around pixel
    :param hist_bin number of bins for the color histogram
    :param sigma_observe std deviation of observation
    """
    pi = np.zeros((particles.shape[0],1))
    # loop over every particle
    for i in range(particles.shape[0]):
        # compute color histogram
        hist_obs = color_histogram(min(max(0, round(particles[i, 0]-0.5*bbox_width)), frame.shape[1]-1),
                                   min(max(0, round(particles[i, 1]-0.5*bbox_height)), frame.shape[0]-1),
                                   min(max(0, round(particles[i, 0]+0.5*bbox_width)), frame.shape[1]-1),
                                   min(max(0, round(particles[i, 1]+0.5*bbox_height)), frame.shape[0]-1),
                                   frame, hist_bin)

        # compute chi distance between observation and target histogram
        dist = chi2_cost(hist_obs, hist)
        # print("dist", dist)
        # idx = np.argwhSiere(particles[:,1]>(frame.shape[0]-3))
        # if idx.any() == i:
        #     print("dist y<3",dist)
     
        # update weight & normalize s.t. it sums to 1
        # print("(dist**2)", (dist**2))
        # print("(2*(sigma_observe**2))", (2*(sigma_observe**2)))
        # print("(dist**2)/(2*(sigma_observe**2))",(dist**2)/(2*(sigma_observe**2)))
        # print("e^(-(dist**2)/(2*(sigma_observe**2)))", np.exp(-(dist**2)/(2*(sigma_observe**2))))
        pi[i,0] = (1 / (np.sqrt(2*np.pi)*sigma_observe) ) * np.exp( 
                  -(dist**2)/(2*(sigma_observe**2)) ) # [n,1]
        # # print("pi",pi)
        # # give border pixels zero weight
        # # print("particles",particles[0:20,:])
        # # print("pi1",pi[0:20,:])
        # idx0 = np.argwhere(particles[:,0]==0)
        # # print("idx0",idx0)
        # idx1 = np.argwhere(particles[:, 1] == 0)
        # # print("idx1", idx1)
        # idx2 = np.argwhere(particles[:, 0] > frame.shape[1]-2)
        # # print("idx2", idx2)
        # idx3 = np.argwhere(particles[:, 1] > frame.shape[0]-20)
        # # print("idx3", idx3)
        # idx_all = np.vstack((idx0,idx1,idx2,idx3))
        # # print("idx",idx)
        # pi[idx_all, 0] = 0
        # # if idx.any() == i:
        # #     print("pi[idx,0]", pi[i, 0])
        # # print("pi[0:20,:]",pi[0:20,:])
        

    if np.sum(pi, axis=0) != 0.0:
        pi = pi / np.sum(pi,axis=0)

    return pi


