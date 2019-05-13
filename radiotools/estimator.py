# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import numpy as np

import scipy.linalg as la
import scipy.sparse as ss
import scipy.sparse.linalg as sla

from . import uvtools


class PSEstimatorBase(object):
    """Base class for quadractic power spectrum estimators.

    Parameters
    ----------
    proj : ss.csr_matrix
        Sparse matrix projector.
    sky : SkyCovariance instance
        Function that returns a sky covariance matrix.
    noise : float
        Noise standard deviation.
    grid : np.ndarray
        UV grid to project onto.
    """

    proj = None

    sky = None
    noise = None
    grid = None

    #l_bands = np.logspace(np.log10(300.0), np.log10(3e4), 21)
    l_bands = np.linspace(0, 3e4, 21)
    _bands = None

    def __init__(self, proj, sky, noise, grid):

        self.proj = proj
        self.sky = sky
        self.noise = noise
        self.grid = grid

    def setup(self):
        """Setup the power spectrum bands and covariance matrices.
        """

        # Setup routine for generating sky sims
        self.sim = vis_simulation(self.grid, self.proj, self.sky.powerspectrum, self.noise)

        # Set up powerspectrum bands and sky covariances
        self._bands = list(zip(self.l_bands[:-1], self.l_bands[1:]))
        self.l_centre = [ 0.5 * (ls + le) for ls, le in self._bands]
        self.C_a = np.array([
            uvtools.covariance_band(self.grid, ls, le, mat=False)
            for ls, le in self._bands
        ])

        # Construct the noise and sky covariances in their respective basis
        self.C_sky = self.sky.covariance(self.grid)
        self.C_noise = ss.identity(self.proj.shape[0]) * self.noise**2 / 2.0  # Factor of two to give variance of real/imag separately

        # Construct the full covariance matrix as a LinearOperator
        def mv(v):
            n = self.C_noise.dot(v)
            s = self.proj.dot(self.C_sky.dot(self.proj.T.dot(v)))
            return s + n

        self.C_full = sla.LinearOperator(shape=self.C_noise.shape, dtype=np.float64,
                                         matvec=mv, matmat=mv, rmatvec=mv)

    def q_estimator(self, vis):
        """Calculate the biased q-estimator for a set of visibilities.
        """
        pass

    def p_estimator(self, vis):
        """Calculate the unbiased p-estimator.
        """
        q_a = self.q_estimator(vis)

        p_a = np.dot(self.M_ab, q_a - self.b_a)

        return p_a

    _M_ab = None

    @property
    def M_ab(self):
        """The power spectrum bin unmixing matrix M_ab.
        """

        if self._M_ab is None:
            self._calculate_M_ab()

        return self._M_ab

    _b_a = None

    @property
    def b_a(self):
        """The estimator bias.
        """
        if self._b_a is None:
            self._calculate_b_a()

        return self._b_a


class OptimalEstimator(PSEstimatorBase):

    fisher_samples = 500

    def q_estimator(self, vis):

        Ci_v, info = sla.cg(self.C_full, vis, tol=1e-4)

        if info != 0:
            raise RuntimeError('meh')

        Cp = self.proj.T.dot(Ci_v)

        Cp2 = np.abs(Cp)**2

        q = 0.5 * np.dot(self.C_a, Cp2)

        return q

    def fisher_bias(self, n=500, fisher_only=False):

        qav = []

        bav = []

        for i in range(n):

            s1, n1, v1 = self.sim()

            qa = self.q_estimator(v1)
            qav.append(qa)

            if not fisher_only:
                ba = self.q_estimator(n1)
                bav.append(ba)

        F_ab = np.cov(np.array(qav).T)

        if fisher_only:
            return F_ab
        else:
            b_a = np.mean(np.array(ba), axis=0)
            return F_ab, b_a

    def _calculate_M_ab(self):

        F_ab, b_a = self.fisher_bias(self.fisher_samples)

        self._M_ab = la.pinv(F_ab)
        self._b_a = b_a

    def _calculate_b_a(self):
        self._calculate_M_ab()


class BareEstimator(PSEstimatorBase):

    def q_estimator(self, vis):

        Cp = self.proj.T.dot(vis)

        Cp2 = np.abs(Cp)**2

        q = 0.5 * np.dot(self.C_a, Cp2)

        return q

    def _calculate_M_ab(self):

        nband = len(self._bands)

        # Form sparse matrices for C_a^{1/2}
        hC_a = [ss.diags(cband**0.5, 0) for cband in self.C_a]

        # Get the projection matrix product
        BTB = self.proj.T.dot(self.proj)

        # Create a placeholder matrix for the normalisation
        iM_ab = np.zeros((nband, nband))

        # Iterate over all band pairs and calculate the Trace
        for ii in range(nband):

            Cp = BTB.dot(hC_a[ii])

            for ij in range(ii, nband):

                iC2 = hC_a[ij].dot(Cp)

                tr = np.sum(iC2.data**2)

                iM_ab[ii, ij] = 0.5 * tr
                iM_ab[ij, ii] = 0.5 * tr

        # Invert the matrix to get M_ab
        self._M_ab = la.pinv(iM_ab)

    def _calculate_b_a(self):

        # Get the projection matrix product
        BTNB_diag = self.proj.T.dot(self.C_noise.dot(self.proj)).diagonal()

        self._b_a = 0.5 * (self.C_a * BTNB_diag[np.newaxis, :]).sum(axis=1)


class WeightedEstimator(PSEstimatorBase):

    def setup(self):

        super(WeightedEstimator, self).setup()

        self.C_full_diag = (
            np.array(self.proj.multiply(self.C_sky.dot(self.proj.T).T).sum(axis=1)).reshape(-1) +
            self.C_noise.diagonal()
        )

    def q_estimator(self, vis):

        Cp = self.proj.T.dot(vis / self.C_full_diag)

        Cp2 = np.abs(Cp)**2

        q = 0.5 * np.dot(self.C_a, Cp2)

        return q

    def _calculate_M_ab(self):

        nband = len(self._bands)

        # Form sparse matrices for C_a^{1/2}
        hC_a = [ss.diags(cband**0.5, 0) for cband in self.C_a]

        Cd = ss.diags(1.0 / self.C_full_diag, 0)

        # Get the projection matrix product
        BTCdB = self.proj.T.dot(Cd.dot(self.proj))

        # Create a placeholder matrix for the normalisation
        iM_ab = np.zeros((nband, nband))

        # Iterate over all band pairs and calculate the Trace
        for ii in range(nband):

            Cp = BTCdB.dot(hC_a[ii])

            for ij in range(ii, nband):

                iC2 = hC_a[ij].dot(Cp)

                tr = np.sum(iC2.data**2)

                iM_ab[ii, ij] = 0.5 * tr
                iM_ab[ij, ii] = 0.5 * tr

        # Invert the matrix to get M_ab
        self._iM_ab = iM_ab
        self._M_ab = la.pinv(iM_ab)

    def _calculate_b_a(self):

        # Create a new weighted noise matrix
        wN = ss.diags(self.C_noise.diagonal() / self.C_full_diag**2, 0)

        # Get the projection matrix product
        BTNB_diag = self.proj.T.dot(wN.dot(self.proj)).diagonal()

        self._b_a = 0.5 * (self.C_a * BTNB_diag[np.newaxis, :]).sum(axis=1)


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

    l = 2 * np.pi * np.hypot(uv_grid[:, 0], uv_grid[:, 1])
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
