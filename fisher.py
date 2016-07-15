import numpy as np

import scipy.sparse as ss
import scipy.sparse.linalg as sla

import miriad
import uvtools

# Read in MIRIAD data and configure some stuff
miriad_data = miriad.read('data/el1_6_wlsm_expbal2_rr_22m2.64.1588')

max_freq = miriad_data['freq'].max()

max_baseline = np.abs(miriad_data['uvw'][..., :2]).max()
dish_size = 22.0

# Fetch the list of UV plane samples, and generate the UV-plane grid
shape_full, shape_half, grid_full, grid_half = uvtools.grid(max_baseline, dish_size, max_freq * 1e3)

# Configure the statistics of the sky and noise
sky_ps = uvtools.SkyCovariance()

# Setup list of frequencies
frequencies = miriad_data['freq']
frequencies = frequencies.reshape(-1, 4).mean(axis=-1)

int_time = np.nanmedian(np.diff(miriad_data['time']) * 24 * 3600.0)
freq_width = np.nanmedian(np.diff(frequencies))

sefd = 55.0
noise_std = sefd / np.abs(int_time * freq_width * 1e9)**0.5

# Generate the projection matrices

uvp_list = []

for fi, freq in enumerate(frequencies):

    print "Stacking freq:", freq

    wavelength = 0.3 / freq

    uv_pos = miriad_data['uvw'][..., :2].reshape(-1, 2) / wavelength
    uvm = uvtools.projection_uniform_beam(grid_full, uv_pos[:], dish_size / wavelength)

    uvm *= (freq / 1.75)**-1.0

    print type(uvm)

    # if uvp is None:
    #     uvp = uvm
    # else:
    #     uvp = ss.vstack(uvp, uvm)
    uvp_list.append(uvm)

uvp = ss.vstack(uvp_list)

print uvp.shape

fe = uvtools.fourier_h2f(*shape_full)
proj = uvp.dot(fe)


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
    l = np.where(l == 0.0, 1.0, l)
    psa = ps_func(l)

    w = (np.concatenate([psa, psa]) / 2.0)**0.5

    def _sim():

        sky = np.random.standard_normal(w.shape) * w
        vis = proj.dot(sky)

        noise = noise_amp * np.random.standard_normal(vis.shape) / 2**0.5

        vis += noise

        return sky, noise, vis

    return _sim


class PSEstimator(object):

    proj = None

    sky = None
    noise = None
    grid = None

    l_bands = np.logspace(np.log10(300.0), np.log10(3e4), 21)

    _bands = None

    def __init__(self, proj, sky, noise, grid):

        self.proj = proj
        self.sky = sky
        self.noise = noise
        self.grid = grid

    def setup(self):

        self.sim = vis_simulation(self.grid, self.proj, self.sky, self.noise)

        self._bands = zip(self.l_bands[:-1], self.l_bands[1:])

        self.l_centre = [ 0.5 * (ls + le) for ls, le in self._bands]

        self.C_a = np.array([
            uvtools.covariance_band(self.grid, ls, le, mat=False)
            for ls, le in self._bands
        ])

        self.C_sky = sky_ps.covariance(grid_half)
        self.C_noise = ss.identity(proj.shape[0]) * self.noise**2

        self.C_full = proj.dot(self.C_sky.dot(proj.T)) + self.C_noise

    def q_estimator(self, vis):

        Ci_v, info = sla.cg(self.C_full, vis, tol=1e-9)

        if info != 0:
            raise RuntimeError('meh')

        Cp = self.proj.T.dot(Ci_v)

        q = []

        Cp2 = np.abs(Cp)**2

        q = np.dot(self.C_a, Cp2)

        return q

    def fisher(self, n):

        qav = []

        for i in range(n):

            s1, n1, v1 = self.sim()

            qa = self.q_estimator(v1)

            qav.append(qa)

        F_ab = np.cov(np.array(qav).T)

        return F_ab


pse = PSEstimator(proj, sky_ps.powerspectrum, noise_std, grid_half)
#pse.setup()

#np.save('fisher.npy', F_ab)
