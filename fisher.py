import numpy as np

import scipy.sparse as ss
import scipy.sparse.linalg as sla

import miriad
import uvtools

# Read in MIRIAD data and configure some stuff
miriad_data = miriad.read('data/el1_6_wlsm_expbal2_rr_22m2.64.1588')

freq = miriad_data['freq'][0]
max_freq = miriad_data['freq'].max()
wavelength = 0.3 / freq

max_baseline = np.abs(miriad_data['uvw'][:2]).max()
dish_size = 15.0

# Fetch the list of UV plane samples, and generate the UV-plane grid
uv_pos = miriad_data['uvw'][..., :2].reshape(-1, 2) / wavelength
shape_full, shape_half, grid_full, grid_half = uvtools.grid(max_baseline, dish_size, max_freq * 1e3)

# Configure the statistics of the sky and noise
sky_ps = uvtools.SkyCovariance()
noise_amp = 0.1  # Jy

# Generate the projection matrices
uvm = uvtools.projection_uniform_beam(grid_full, uv_pos[:2000], dish_size / wavelength)
fe = uvtools.fourier_h2f(*shape_full)
proj = uvm.dot(fe)


def vis_simulation(uv_grid, proj, ps_func, noise_amp):
    """Generate a routine for creating random sky simulations.

    Parameters
    ----------
    uv_grid : np.ndarray[:, 2]
        Location in the UV-plane.
    proj : :class:`ss.csr_matrix`
        Sparse matrix projecting the UV-plane into observations.
    ps_func : function
        Power spectrum function.
    noise_amp : float
        The noise amplitude.
    """

    l = np.hypot(uv_grid[:, 0], uv_grid[:, 1])

    w = ps_func(l)**0.5

    def _sim():

        sky = np.random.standard_normal(w.shape) * w
        vis = proj.dot(sky)

        noise = noise_amp * np.random.standard_normal(vis.shape) / 2**0.5

        vis += noise

        return sky, noise, vis

    return _sim


sim = vis_simulation(grid_half, proj, sky_ps.powerspectrum, noise_amp)

lbands = np.linspace(0, 1e4, 26)


c_a = np.array([
    uvtools.covariance_band(grid_half, ls, le, mat=False)
    for ls, le in zip(lbands[:-1], lbands[1:])
])

C_sky = sky_ps.covariance(grid_half, 70.0, -1.5, 0.0)
C_noise = ss.identity(proj.shape[0]) * noise_amp**2

C_full = proj.dot(C_sky.dot(proj.T)) + C_noise


def q_estimator(vis):

    Ci_v, info = sla.cg(C_full, vis, tol=1e-9)

    if info != 0:
        raise RuntimeError('meh')

    Cp = proj.T.dot(Ci_v)

    q = []

    Cp2 = np.abs(Cp)**2

    q = np.dot(c_a, Cp2)

    return q


qav = []

for i in range(200):

    s1, n1, v1 = sim()

    qa = q_estimator(v1)

    qav.append(qa)

F_ab = np.cov(np.array(qav).T)
