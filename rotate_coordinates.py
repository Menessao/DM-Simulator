import numpy as np

def cw_rotate(vec, angles):
    """
    Rotates vector vec clockwise about the origin
    by all angles in input

    Parameters
    ----------
    vec : ndarray [2,Npoints]
        Array of 2D point coordinates.
    angles : ndarray [Nangles,]
        List of angles to rotate the ponints by.

    Returns
    -------
    rot_vec: ndarray [Npoints,2]
        Array of rotated point coordinates.

    """

    L = len(angles)

    n_pts = int(np.size(vec)/2)
    rot_vec = np.zeros([2,n_pts*L])

    for k, angle in enumerate(angles):
        c = np.cos(angle)
        s = np.sin(angle)
        rot_mat = [[c,s],[-s,c]]

        if np.shape(vec)[0] != 2:
            aux_vec = rot_mat @ vec.transpose()
        else:
            aux_vec = rot_mat @ vec

        rot_vec[:,k*n_pts:(k+1)*n_pts] = np.reshape(aux_vec,[2,n_pts])

    return rot_vec.T
