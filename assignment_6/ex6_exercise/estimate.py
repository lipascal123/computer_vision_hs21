import numpy as np
from numpy.core.numeric import NaN


def estimate(particles, particles_w):
    """
    Computes the mean of the weighted particles
    :param particles [n_particles,n_states]
    :param particles_w weight for each particle [n_particles,1]
    """
    nom = particles*particles_w
    nom = np.sum(nom,axis=0)
    denom = np.sum(particles_w)

    return nom/denom 
    
