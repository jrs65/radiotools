import numpy as np
import scipy.sparse as ss


def fourier_h2f(nx, ny):
    """Generate the 2D Fourier expansion matrix.

    This is the matrix which takes the flattened results of a 2D real-FFT
    (packed as all real elements, and then all imaginary elements) into the full
    2D FFT space.

    Parameters
    ----------
    nx, ny : int
        Shape of the original image.

    Returns
    -------
    mat : :class:`scipy.spare.csr_matrix`
    """

    msize = (2 * nx * ny, 2 * nx * (ny / 2 + 1))

    data = np.zeros(msize[0], dtype=np.float64)
    i = np.zeros(msize[0], dtype=np.int)
    j = np.zeros(msize[0], dtype=np.int)

    def _copy_into(ld, li, lj, s):
        e = s + ny / 2
        data[s:e] = ld
        i[s:e] = li
        j[s:e] = lj

    for ix in range(nx):

        # Set positive frequencies (real part)
        td = np.ones(ny / 2)
        ti = ix * ny + np.arange(ny / 2)
        tj = ix * (ny / 2 + 1) + np.arange(ny / 2)
        s = ix * ny
        _copy_into(td, ti, tj, s)

        # Set positive frequencies (imaginary part)
        td = np.ones(ny / 2)
        ti = (nx + ix) * ny + np.arange(ny / 2)
        tj = (nx + ix) * (ny / 2 + 1) + np.arange(ny / 2)
        s = (nx + ix) * ny
        _copy_into(td, ti, tj, s)

        # Set negative frequencies (row zero)
        if ix == 0:

            # Real part
            td = np.ones(ny / 2)
            ti = (ix + 1) * ny - np.arange(ny / 2) - 1
            tj = ix * (ny / 2 + 1) + np.arange(ny / 2) + 1
            s = ix * ny + ny / 2
            _copy_into(td, ti, tj, s)

            # Imag part
            td = -np.ones(ny / 2)
            ti = (nx + ix + 1) * ny - np.arange(ny / 2) - 1
            tj = (nx + ix) * (ny / 2 + 1) + np.arange(ny / 2) + 1
            s = (nx + ix) * ny + ny / 2
            _copy_into(td, ti, tj, s)

        # Negative freq (non-zero row)
        else:
            # Real part
            td = np.ones(ny / 2)
            ti = (nx - ix + 1) * ny - np.arange(ny / 2) - 1
            tj = ix * (ny / 2 + 1) + np.arange(ny / 2) + 1
            s = ix * ny + ny / 2
            _copy_into(td, ti, tj, s)

            # Imag part
            td = -np.ones(ny / 2)
            ti = (2 * nx - ix + 1) * ny - np.arange(ny / 2) - 1
            tj = (nx + ix) * (ny / 2 + 1) + np.arange(ny / 2) + 1
            s = (nx + ix) * ny + ny / 2
            _copy_into(td, ti, tj, s)

    expansion_matrix = ss.coo_matrix((data, (i, j)), shape=msize)

    return expansion_matrix.tocsr()


def _uv_proj_dist(uv_grid, uv_pos, uv_max):
    """Generate a sparse matrix projecting from the full UV plane, into measurements.

    This does not include a primary beam model. The entries in the matrix are
    the distance in wavelengths.

    Parameters
    ----------
    uv_grid : np.ndarray
        Co-ordinates of the UV-plane grid.
    uv_pos : np.ndarray
        Positions in the UV plane of each observation.
    uv_max : float
        Largest distance in the UV plane to include.

    Returns
    -------
    mat : :class:`scipy.spare.csr_matrix`
    """

    import _proj

    msize = (2 * uv_pos.shape[0], 2 * uv_grid.shape[0])

    npos = uv_pos.shape[0]
    ngrid = uv_grid.shape[0]

    data_list = []
    i_list = []
    j_list = []

    for ui, uv in enumerate(uv_pos):

        uv_dist, loc = _proj._offset_dist(uv_grid, uv[0], uv[1], uv_max)

        nloc = len(loc)

        # Real to real projection
        data_list.append(uv_dist)
        i_list.append(ui * np.ones(nloc))
        j_list.append(loc)

        # # Real to imaginary projection
        # data = np.append(data, uv_dist)
        # i = np.append(i, (ui + npos) * np.ones(nloc))
        # j = np.append(j, loc)
        #
        # # Imaginary to real projection
        # data = np.append(data, uv_dist)
        # i = np.append(i, ui * np.ones(nloc))
        # j = np.append(j, loc + ngrid)

        # Imaginary to imaginary projection
        data_list.append(uv_dist)
        i_list.append((ui + npos) * np.ones(nloc))
        j_list.append(loc + ngrid)

    data = np.concatenate(data_list)
    i = np.concatenate(i_list)
    j = np.concatenate(j_list)

    uv_matrix = ss.coo_matrix((data, (i, j)), shape=msize)

    return uv_matrix.tocsr()


def projection_uniform_beam(uv_grid, uv_pos, dish_size):
    """Generate the sparse projection matrix for a uniformly illuminated dish.

    Parameters
    ----------
    uv_grid : np.ndarray
        Co-ordinates of the UV-plane grid.
    uv_pos : np.ndarray
        Positions in the UV plane of each observation.
    dish_size : float
        Size of the dish in wavelengths. Seriously float only, you can't pass in
        an array even if you have different frequencies in here.

    Returns
    -------
    mat : :class:`scipy.sparse.csr_matrix`
        Projection matrix.
    """
    norm = 1.0 / (4 * np.pi * dish_size**2)

    uvm = _uv_proj_dist(uv_grid, uv_pos, dish_size)

    uvm.data *= norm

    return uvm


def grid(max_baseline, dish_size, max_freq, samp=10.0):
    """Generate a co-ordinate grid in the UV-plane.

    Parameters
    ----------
    max_baseline : float
        Largest baseline length in metres.
    dish_size : float
        Size of antenna/dish in metres.
    max_freq : float
        Highest frequency we are considering (in MHz).
    samp : float, optional
        Sampling factor. Roughly the number of samples per antenna element.

    Returns
    -------
    shape_full : tuple
        Shape for the full grid.
    shape_half : tuple
        Shape for the half FFT grid.
    grid_full : np.ndarray[:, 2]
        Points in the full grid.
    grid_half : np.ndarray[:, 2]
        Points in the half grid.
    """

    wavelength = 3.0e2 / max_freq

    max_uv = 1.2 * max_baseline / wavelength

    num_uv = int(samp * max_baseline / dish_size)

    u1 = np.linspace(-max_uv, max_uv, 2 * num_uv, endpoint=False)
    u1 = np.fft.fftshift(u1)
    v1 = u1[:(num_uv + 1)]

    uv_full = np.dstack(np.broadcast_arrays(u1[:, np.newaxis],
                                            u1[np.newaxis, :]))

    uv_half = np.dstack(np.broadcast_arrays(u1[:, np.newaxis],
                                            v1[np.newaxis, :]))

    shape_full = uv_full.shape[:2]
    shape_half = uv_half.shape[:2]

    return shape_full, shape_half, uv_full.reshape(-1, 2), uv_half.reshape(-1, 2)


class SkyCovariance(object):
    r"""Statistics of the sky.

    Attributes
    ----------
    clustered_amplitude : float
        Amplitude of the clustered part of the power spectrum.
    clustered_index : float
        Clustering spectral index.
    pivot_l : int
        Pivot multipole.
    poisson_amplitude : float
        Poisson amplitude.

    Notes
    -----

    The covariance is modelled as:

    .. math::
        \left\langle s(\mathbf{u}) s^*(\mathb{u}) \right\angle = A_c \left(\frac{u}{u_0}\right)^beta + A_p
    """

    clustered_amplitude = 70.0
    clustered_index = -1.5
    pivot_l = 1000

    poisson_amplitude = 50.0

    def powerspectrum(self, l):

        ps_amp = (self.clustered_amplitude * (l / self.pivot_l)**self.clustered_index
            + self.poisson_amplitude)

        return ps_amp

    def covariance(self, uv_grid, diag=False):
        """Generate the covariance between UV-grid points.

        Parameters
        ----------
        uv_grid : np.ndarray[:, 2]
            Location in the UV-plane.
        mat : bool, optional
            Return just the diagonal.

        Returns
        -------
        c_sky : :class:`scipy.sparse.csr_matrix` or :class:`np.ndarray`
            Sparse covariance matrix, or the diagonal of it (as `np.ndarray`)
        """

        l = np.hypot(uv_grid[:, 0], uv_grid[:, 1])
        l = np.where(l == 0.0, 1.0, l)

        ps_amp = self.powerspectrum(l)

        c_diag = np.concatenate([ps_amp, ps_amp]) / 2.0

        if diag:
            return c_diag
        else:
            c_sky = ss.diags(c_diag, 0)
            return c_sky


def covariance_band(uv_grid, u_start, u_end, mat=True):
    r"""Generate the covariance matrix for a power spectrum band.

    Parameters
    ----------
    uv_grid : np.ndarray[:, 2]
        Location in the UV-plane.
    u_start, u_end : float
        Start and end of the power spectrum band.
    mat : bool, optional
        Return sparse covariance matrix. Otherwise, just return the diagonal.

    Returns
    -------
    c_sky : :class:`scipy.sparse.csr_matrix`
        Sparse covariance matrix.

    Notes
    -----

    The covariance is modelled as:

    .. math::
        \left\langle s(\mathbf{u}) s^*(\mathb{u}) \right\angle = A_c u^beta + A_p
    """

    uv_dist = np.hypot(uv_grid[:, 0], uv_grid[:, 1])

    mask = np.logical_and(uv_dist >= u_start, uv_dist < u_end)

    c_diag = np.concatenate([mask, mask]).astype(np.float64) / 2.0

    if mat:
        c_sky = ss.diags(c_diag, 0)
        return c_sky
    else:
        return c_diag
