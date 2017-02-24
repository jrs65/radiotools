# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _offset_dist_2d(double [:, :, ::1] grid, double x, double y, double max_dist):

    cdef double [::1] t
    cdef double max2

    cdef double x0, y0, dx, dy, d2

    cdef int i, j, ii, jj, k = 0
    cdef int Nx, Ny, N, Mx, My, cx, cy, sx, ex, sy, ey

    Nx = grid.shape[0]
    Ny = grid.shape[1]
    N = Nx * Ny

    dx = grid[1, 0, 0] - grid[0, 0, 0]
    dy = grid[0, 1, 1] - grid[0, 0, 1]

    x0 = grid[Nx/2, 0, 0]
    y0 = grid[0, Ny/2, 1]

    Mx = int(max_dist / dx) + 2
    My = int(max_dist / dy) + 2

    cx = int((x - x0) / dx)
    cy = int((y - y0) / dy)

    sx = max(0, cx - Mx)
    ex = min(Nx, cx + Mx)
    sy = max(0, cy - My)
    ey = min(Ny, cy + My)

    max2 = max_dist**2 + dx**2 + dy**2

    cdef double [::1] dist = np.zeros(5 * Mx * My, dtype=np.float64)
    cdef int [::1] pos = np.zeros(5 * Mx * My, dtype=np.int32)

    for i in range(sx, ex):
        for j in range(sy, ey):

            ii = (i + Nx / 2) % Nx
            jj = (j + Ny / 2) % Ny

            d2 = (grid[ii, jj, 0] - x)**2 + (grid[ii, jj, 1] - y)**2

            if d2 < max2:
                dist[k] = d2**0.5
                pos[k] = ii * Nx + jj
                k += 1

    if k == 0:
        dist_trim = np.array([])
        pos_trim = np.array([])
    else:
        dist_trim = dist[:k].copy()
        pos_trim = pos[:k].copy()

    return np.asarray(dist_trim), np.asarray(pos_trim)
