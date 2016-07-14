
"""A few miscellaneous Cython routines to speed up critical operations.
"""

from cython.parallel import prange, parallel
cimport cython

import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _offset_dist(double [:, ::1] grid, double x, double y, double max_dist):

    cdef double [::1] t
    cdef double max2 = max_dist**2

    cdef int nz = 0
    cdef int i, j

    N = grid.shape[0]

    t = np.zeros(N, dtype=np.float64)

    for i in prange(N, nogil=True, num_threads=4, schedule='static'):

        t[i] = (grid[i, 0] - x)**2 + (grid[i, 1] - y)**2

        if t[i] < max2:
            nz += 1

    cdef double [::1] dist = np.zeros(nz, dtype=np.float64)
    cdef int [::1] pos = np.zeros(nz, dtype=np.int32)

    j = 0

    for i in range(N):
        if t[i] < max2:
            dist[j] = t[i]**0.5
            pos[j] = i
            j += 1

    return np.asarray(dist), np.asarray(pos)
