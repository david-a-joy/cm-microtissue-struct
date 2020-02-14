""" Fast-er implementation of sphere packing by random sampling """

import numpy as np
cimport numpy as np

ctypedef np.float64_t FLOAT_TYPE_t

from libc.stdlib cimport rand
cdef extern from "limits.h":
    int RAND_MAX

cdef extern from "math.h":
    long double acos(long double a)
    long double sin(long double a)
    long double cos(long double a)
    long double sqrt(long double a)


cdef random_float(float low, float high):
    """ Random float in a range """
    cdef float rng = high - low
    return rng * (rand() / float(RAND_MAX)) + low


def simulate_spheres_in_sphere(int num_particles,
                               float particle_radius,
                               float sphere_radius,
                               float umin = 0.0,
                               float umax = 1.0,
                               str udist = 'uniform'):
    """ Simulate a set of spheres packed in a sphere

    :param int num_points:
        The number of points to draw inside the sphere
    :param float particle_radius:
        The radius of the spherical particles to pack
    :param float sphere_radius:
        Radius of the sphere to pack into
    :param RandomState rnd:
        The random number generator
    :param float umean:
        0 for center biased, 1 for edge biased, 0.5 for no bias
    :param float urange:
        The range of the generator (1 for no bias)
    :param str udist:
        The distribution for the parameter
    :returns:
        x, y, z coordinates for points in the sphere
    """
    cdef float particle_rr, inner_radius
    cdef int good_sample, sphere_idx, i, bad_rounds
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] spheres

    cdef float phi, costheta, u, theta, r, x, y, z, sx, sy, sz

    # Allocate space for enough spheres
    spheres = np.zeros((num_particles, 3))

    # Helpful constants
    particle_rr = particle_radius**2
    inner_radius = sphere_radius - particle_radius

    if inner_radius < 0.0:
        raise ValueError('Cannot have particle radius > sphere radius')

    bad_rounds = 0
    sphere_idx = 0

    while sphere_idx < num_particles:
        # Draw the angles uniformly at random
        phi = random_float(0.0, 2*np.pi)
        costheta = random_float(-1.0, 1.0)

        # Calculate the u-value to allow for radial bias
        u = random_float(umin, umax)
        if udist == 'left_triangle':
            u = umax - (umax - umin)*sqrt(1.0 - u)
        elif udist == 'right_triangle':
            u = umin + (umax - umin)*sqrt(u)

        theta = acos(costheta)
        r = inner_radius * u**(1.0/3.0)

        # Convert from spherical coordinates to euclidean
        x = r * sin(theta) * cos(phi)
        y = r * sin(theta) * sin(phi)
        z = r * cos(theta)

        if sphere_idx < 1:
            spheres[sphere_idx, 0] = x
            spheres[sphere_idx, 1] = y
            spheres[sphere_idx, 2] = z

            sphere_idx += 1
            bad_rounds = 0
            continue

        # Check if this one is a good sample
        good_sample = True
        for i in range(sphere_idx):
            sx = spheres[i, 0]
            sy = spheres[i, 1]
            sz = spheres[i, 2]
            if (sx-x)**2 + (sy-y)**2 + (sz-z)**2 < particle_rr:
                good_sample = False
                break

        if good_sample:
            spheres[sphere_idx, 0] = x
            spheres[sphere_idx, 1] = y
            spheres[sphere_idx, 2] = z

            sphere_idx += 1
            bad_rounds = 0
            continue

        bad_rounds += 1
        if bad_rounds > 10:
            break

    if sphere_idx < num_particles:
        raise ValueError('Failed to pack spheres. Got {} needed {}'.format(sphere_idx, num_particles))
    return spheres[:, 0], spheres[:, 1], spheres[:, 2]
