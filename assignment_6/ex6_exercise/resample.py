import numpy as np

def resample(particles, particles_w):
    """
    resample the particles with their corresponding weight
    :param particles matrix [n_particles, n_states]
    :param particles_w particle weight [n_particles,1]
    """
    # get idx array of particles
    idx = np.arange(0,particles.shape[0],1)
    # resample idx with particles_w
    resam_idx = np.random.choice(idx, size=particles.shape[0],p=particles_w.flatten())
    # back from idx vector to particles matrix
    resam_particles = particles[resam_idx]
    # print("resam_particles", resam_particles.shape)
    # now all weights have uniform weight
    particles_w = np.ones((resam_particles.shape[0], 1))*(1/resam_particles.shape[0])
    # print("w shape",particles_w.shape)
    return resam_particles, particles_w
