import numpy as np
from scipy.spatial.distance import euclidean


def _c(ca, i, j, P, Q):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = euclidean(P[0], Q[0])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i - 1, 0, P, Q),
                       euclidean(P[i], Q[0]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j - 1, P, Q),
                       euclidean(P[0], Q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(_c(ca, i - 1, j, P, Q),
                _c(ca, i - 1, j - 1, P, Q),
                _c(ca, i, j - 1, P, Q)),
            euclidean(P[i], Q[j]))
    else:
        ca[i, j] = float("inf")
    return ca[i, j]


def frechet_dist(P, Q):
    """
    Discrete Frechet Distance

    Usage
    -----
    Compute the discret frechet distance between trajectories P and Q
    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q
    Returns
    -------
    frech : float, the discret frechet distance between trajectories P and Q
    """
    ca = np.ones((len(P), len(Q)))
    ca = np.multiply(ca, -1)
    return _c(ca, len(P) - 1, len(Q) - 1, P, Q)
