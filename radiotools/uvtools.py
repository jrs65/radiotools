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

    npos = uv_pos.shape[0]
    ngrid = uv_grid.shape[0] * uv_grid.shape[1]

    msize = (2 * npos, 2 * ngrid)

    data_list = []
    i_list = []
    j_list = []

    for ui, uv in enumerate(uv_pos):

        uv_dist, loc = _proj._offset_dist_2d(uv_grid, uv[0], uv[1], uv_max)

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

    # Calculate the normalisation due to the dish size
    norm = 1.0 / (np.pi * dish_size**2)

    # We need to include the finite size of the grid in our projection
    # This works for the moment
    gs = np.abs(uv_grid[0] - uv_grid[1]).sum()
    norm *= gs**2

    uvm = _uv_proj_dist(uv_grid, uv_pos, dish_size)

    uvm.data[:] = norm

    return uvm


def projection_gaussian_beam(uv_grid, uv_pos, fwhm):
    """Generate the sparse projection matrix for a Gaussian beam.

    Parameters
    ----------
    uv_grid : np.ndarray
        Co-ordinates of the UV-plane grid.
    uv_pos : np.ndarray
        Positions in the UV plane of each observation.
    fwhm : float
        Size of the dish in radians. Seriously float only, you can't pass in
        an array even if you have different frequencies in here.

    Returns
    -------
    mat : :class:`scipy.sparse.csr_matrix`
        Projection matrix.
    """

    U_0 = 0.53 / fwhm

    # Calculate the normalisation due to the dish size
    norm = 1.0 / (np.pi * U_0**2)

    # We need to include the finite size of the grid in our projection
    # This works for the moment
    gs = np.abs(uv_grid[0] - uv_grid[1]).sum()
    norm *= gs**2

    uvm = _uv_proj_dist(uv_grid, uv_pos, 3 * U_0)

    uvm.data[:] = norm * np.exp(-1.0 * (uvm.data[:] / U_0)**2)

    # We need to normalise the beam integral to unity to take into account the
    # effects of pixelisation
    beam_int = uvm.sum(axis=1).view(np.ndarray)
    beam_norm = np.where(beam_int == 0.0, 0.0, np.abs(1.0 / beam_int))
    uvm = uvm.multiply(ss.csc_matrix(beam_norm))

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

    num_uv = int(samp * max_uv * wavelength / dish_size)

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


def img_grid(shape, size):
    """Generate a co-ordinate grid in the UV-plane that can represent a given image.

    Parameters
    ----------
    shape : tuple
        Image shape, i.e. pixels in each dimension (RA, DEC)
    size : tuple
        Image size in radians in the (RA, DEC) directions (but not the size in RA, DEC).

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

    spacing = np.array(size) / np.array(shape)

    u1 = np.fft.fftfreq(shape[0], spacing[0])
    v1 = np.fft.fftfreq(shape[1], spacing[1])

    v2 = np.linspace(0, 0.5 / spacing[1], shape[1] // 2 + 1, endpoint=True)

    uv_full = np.dstack(np.broadcast_arrays(u1[:, np.newaxis],
                                            v1[np.newaxis, :]))

    uv_half = np.dstack(np.broadcast_arrays(u1[:, np.newaxis],
                                            v2[np.newaxis, :]))

    shape_full = uv_full.shape[:2]
    shape_half = uv_half.shape[:2]

    return shape_full, shape_half, uv_full.reshape(-1, 2), uv_half.reshape(-1, 2)


def image_to_uv(img, shape_half=None):
    """Take an image as a 2D array and generate the UV data.

    Parameters
    ----------
    img : np.ndarray[nx, ny]
        Image to transform. Axes should be in order of (RA, DEC) directions.
        Centre of the image should be at the centre of the array.
    shape_half : tuple, optional
        Size of the (half-)UV plane.

    Returns
    -------
    uv : np.ndarray[2 * nx, ny / 2 + 1]
        UV format data in the right format.
    """

    img = np.fft.fftshift(img)
    uvc = np.fft.rfftn(img)

    if shape_half is not None:

        nxh = shape_half[0] / 2
        ny = shape_half[1]

        uvc = np.concatenate([uvc[:nxh, :ny], uvc[-nxh:, :ny]])

    uvr = np.concatenate([uvc.real, uvc.imag])

    return uvr


def uv_to_image(uv, shape=None):
    """Take UV data and transform into an image by FFT.

    Parameters
    ----------
    uv : np.ndarray[2 * nx, ny / 2 + 1]
        UV format data. The first axis is the real and imaginary parts for every
        point in U, and the second axis is half of the V axis (only half because
        the image space is real).

    Returns
    -------
    img : np.ndarray[nx, ny]
        Image to generated from the UV data. Should be (RA, DEC) axis order with
        the centre of the image in the centre of the array.
    """
    nx = uv.shape[0] / 2

    uvc = uv[:nx] + 1.0J * uv[nx:]

    img = np.fft.irfftn(uvc, s=shape)

    return np.fft.fftshift(img)


def projection_from_miriad(miriad_data, grid, fwhm):
    """Construct a projection matrix from miriad data.

    Parameters
    ----------
    miriad_data : dict
        MIRIAD data as returned by `miriad.read`
    grid : tuple
        Grid definition as returned by one of `img_grid` or `grid`. This is a
        tuple consisting of (shape of full grid, shape of half grid, UV
        locations of full grid cells, UV locations of half grid cells)
    fwhm : float
        FWHM of the primary beam of the telescope (in radians).
    """

    # Unpack grid argument
    shape_full, shape_half, grid_full, grid_half = grid

    # Get base frequencies and wavelengths
    frequencies = miriad_data['freq']
    wavelengths = 0.299792 / frequencies
    nfreq = len(frequencies)

    if miriad_data['pol'][0] != 'I':
        raise RuntimeError('Polarization is not instrumental Stokes.')

    # Get the mask for Stokes I
    weight = miriad_data['weight'][:, 0]
    weight = weight.reshape(-1, nfreq)

    fe = fourier_h2f(*shape_full)

    # Get the antenna position as a function of time (in metres)
    ant_pos = miriad_data['uvw'][..., :2].reshape(-1, 2)

    uvp_list = []

    # Iterate over all frequencies and generate the projection matrix for each
    for fi, wavelength in enumerate(wavelengths):

        # Calculate the UV positions in wavelengths
        uv_pos = ant_pos / wavelength

        # Get the projection matrix
        uvm = projection_gaussian_beam(grid_full.reshape(shape_full + (2,)), uv_pos[:], fwhm)

        # Construct the mask array and apply it
        t = weight[:, fi]
        w = ss.diags(np.concatenate([t, t]), 0)
        uvm = w.dot(uvm)

        # Apply the Fourier expansion matrix to make the projection be from the half plane
        uvm = uvm.dot(fe)

        # Add the projection to the list
        uvp_list.append(uvm)

    return uvp_list


def projection_multi_frequency(proj, weight):
    """Combine together a projection for individual frequencies into one from a
    single scaled UV plane.

    Parameters
    ----------
    proj : list
        A sparse projection matrix for each frequenecy.
    weight : np.ndarray
        The weight to give each frequency.
    """

    weighted_proj = [p * w for p, w in zip(proj, weight)]

    return ss.vstack(weighted_proj)


def pack_vis(vis):
    """Pack visibility data into a vector.

    Parameters
    ----------
    vis : np.ndarray[baseline, time, freq]
        Unpolarised visibility data

    Returns
    -------
    vis_vec : np.ndarray[:]
    """
    # Get the frequency axis to the front
    vis = vis.transpose((2, 0, 1))

    # Turn the real and imaginary parts into a new axis
    ds = np.concatenate((vis[:, np.newaxis].real, vis[:, np.newaxis].imag), axis=1)

    # Flatten into a vector and return
    return ds.flatten()


def unpack_vis(vis_vec, shape):
    """Unpack visibility vector into order for unpolarised data.

    Parameters
    ----------
    vis_vec : np.ndarray[:]
        Packed visibility data.
    shape : tuple
        Shape of the data (baseline, time, freq).

    Returns
    -------
    vis_vec : np.ndarray[baseline, time, freq]
        Unpolarised visibility data
    """
    vecr = vis_vec.reshape((shape[-1], 2) + shape[:-1])
    return (vecr[:, 0] + 1.0J * vecr[:, 1]).transpose(1, 2, 0)


def dirty_map(vis, proj_matrix, grid):
    """Generate a quick dirty map.

    Parameters
    ----------
    vis : np.ndarray[:]
        Visibility vector.
    proj_matrix : :class:`scipy.sparse.csr_matrix`
        A sparse projection matrix.
    grid : tuple
        Grid definition.

    Returns
    -------
    image : np.ndarray[:, :]
    """

    # Unpack grid argument
    shape_full, shape_half, grid_full, grid_half = grid

    dirty_uv_half = proj_matrix.T.dot(vis)

    # bt = np.ones_like(vis)
    # bt[(bt.size/2):] = 0.0
    # dirty_b = proj_matrix.T.dot(bt)

    # dirty_beam = uv_to_image(dirty_b.reshape(-1, shape_half[-1]))
    dirty_map = uv_to_image(dirty_uv_half.reshape(-1, shape_half[-1]))

    return dirty_map  # dirty_beam


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

    clustered_amplitude = 5.0
    clustered_index = -1.5
    pivot_l = 1000

    poisson_amplitude = 4.0

    def powerspectrum(self, l):

        ps_amp = (self.clustered_amplitude * (l / self.pivot_l)**self.clustered_index +
                  self.poisson_amplitude)

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

        l = 2 * np.pi * np.hypot(uv_grid[:, 0], uv_grid[:, 1])
        l = np.where(l == 0.0, 1.0, l)

        ps_amp = self.powerspectrum(l)

        c_diag = np.concatenate([ps_amp, ps_amp]) / 2.0

        if diag:
            return c_diag
        else:
            c_sky = ss.diags(c_diag, 0)
            return c_sky


def covariance_band(uv_grid, l_start, l_end, mat=True):
    r"""Generate the covariance matrix for a power spectrum band.

    Parameters
    ----------
    uv_grid : np.ndarray[:, 2]
        Location in the UV-plane.
    l_start, l_end : float
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

    l = 2 * np.pi * np.hypot(uv_grid[:, 0], uv_grid[:, 1])

    mask = np.logical_and(l >= l_start, l < l_end)

    c_diag = np.concatenate([mask, mask]).astype(np.float64) / 2.0

    if mat:
        c_sky = ss.diags(c_diag, 0)
        return c_sky
    else:
        return c_diag
