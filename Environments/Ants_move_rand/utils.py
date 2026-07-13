import numpy as np

# normalization of a vector
def norm_vec(vectors, axis=-1):
    norm = np.linalg.norm(vectors, axis=axis, keepdims=True)
    norm[norm == 0] = 1
    normalized_vectors = vectors / norm
    return normalized_vectors
    

# rotate a vector by an angle theta in 2D
def rotate_vec(unitary_vector, theta):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    rotated_vector = np.dot(rotation_matrix, unitary_vector.T)
    return rotated_vector
    

# signed angle between two 2d-vectors
def angle_btw(v1, v2):
    return np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
    

# signed value of an angle between a 2d vector (v1) and a list of 2d-vectors (v2)
def angle_btw_vect(v1, v2):
    return np.arctan2(v1[0]*v2[:,1] - v1[1]*v2[:,0], v1[0]*v2[:,0] + v1[1]*v2[:,1])

# signed value of an angle between two 3d-vectors
def angle_btw3D(v1, v2):
    return np.arctan2(np.cross(v1, v2), np.dot(v1, v2))

# return a list of angles between a 3d-vector and a list of 3d-vectors
def angle_btw_vect3D(v1, v2):
    return np.arccos(np.dot(v1, v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2, axis=1)))

# rotate a 3d-vector by an angle theta, phi in spherical coordinates
def rotate_vec3D(v, theta, phi):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    rotated_vector = np.dot(rotation_matrix, v.T)
    rotation_matrix = np.array([[np.cos(phi), 0, np.sin(phi)],
                                [0, 1, 0],
                                [-np.sin(phi), 0, np.cos(phi)]])
    rotated_vector = np.dot(rotation_matrix, rotated_vector)
    return rotated_vector


# compute polarization of particles
def polarization(particles, speed, num_particles):
    return np.linalg.norm(np.sum([p.velocity for p in particles], axis=0))/(num_particles*speed)


# compute average fluctuations around the mean velocity
def average_fluctuations(particles, num_particles):
    N = len(particles)
    fluctuations = np.zeros((N,2))
    for i in range(N):
        fluctuations[i] = particles[i].velocity - np.sum([p.velocity for p in particles], axis=0)/num_particles
        
    return np.linalg.norm(np.sum(fluctuations, axis=0))/num_particles


# Rotate a vector `vi` towards another vector `vf` by angle `theta`.
def rotate_towards(vi, vf, theta):

    x = np.cross(vi, vf)
    _x_ = np.linalg.norm(x)
    if _x_ < 1e-15:
        return vi
    else:
        x /= _x_
    A = np.array([[ 0, -x[2], x[1] ],
                  [ x[2], 0, -x[0] ],
                  [ -x[1], x[0], 0 ]])
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(3) + s * A + (1-c) * A.dot(A)

    return R.dot(vi)


# Return the spherical angles `theta` and `phi` associated with a unit vector `v`.
def cart2sphere(v):

    x,y,z = v
    theta = np.arccos(np.clip(z,-1,1))
    phi = np.arctan2(x,y)

    return theta, phi


# Return the unit vector `v` associated with angles `theta` and `phi`.
def sphere2cart(theta, phi):

    if theta < 0:
        theta = np.pi + theta
        phi += np.pi
    elif theta > np.pi:
        theta = theta - np.pi
        phi += np.pi
    st, sp = np.sin(theta), np.sin(phi)
    ct, cp = np.cos(theta), np.cos(phi)
    x = st * sp
    y = st * cp
    z = ct

    return np.array([x,y,z])


# Return the list of unit vectors perpendicular to the respectively vectors in 2D space
def perpendicular(a) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b




