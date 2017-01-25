import numpy as np

import scipy.sparse as ss

import miriad
import uvtools
import estimator

# Read in MIRIAD data and configure some stuff
miriad_data = miriad.read('data/el1_6_wlsm_expbal2_rr_22m2.64.1588')

max_freq = miriad_data['freq'].max()
max_baseline = np.abs(miriad_data['uvw'][..., :2]).max()
dish_size = 22.0

# Fetch the list of UV plane samples, and generate the UV-plane grid
shape_full, shape_half, grid_full, grid_half = uvtools.grid(max_baseline, dish_size, max_freq * 1e3)

# Configure the statistics of the sky and noise
sky_ps = uvtools.SkyCovariance()

# Get base frequencies and widths
frequencies = miriad_data['freq']
freq_width = np.ones_like(frequencies) * np.abs(np.nanmedian(np.diff(frequencies)))

# Construct frequencies to use in this run
frequencies = frequencies.reshape(-1, 4).mean(axis=-1)
freq_width = freq_width.reshape(-1, 4).sum(axis=-1)

# Find integration time (assuming it's the same for all samples)
int_time = np.nanmedian(np.diff(miriad_data['time']) * 24 * 3600.0)

# Calculate the noise level. Assume the SEFD from the ATCA documentation
sefd = 55.0
noise_std = sefd / np.abs(int_time * np.median(freq_width) * 1e9)**0.5

# Generate the projection matrices

uvp_list = []

for fi, freq in enumerate(frequencies):

    print "Stacking freq:", freq

    wavelength = 0.3 / freq

    uv_pos = miriad_data['uvw'][..., :2].reshape(-1, 2) / wavelength
    uvm = uvtools.projection_uniform_beam(grid_full, uv_pos[:], dish_size / wavelength)

    uvm *= (freq / 1.75)**-1.0

    uvp_list.append(uvm)

uvp = ss.vstack(uvp_list)

fe = uvtools.fourier_h2f(*shape_full)
proj = uvp.dot(fe)


pse = estimator.OptimalEstimator(proj, sky_ps, noise_std, grid_half)
#pse.setup()

#np.save('fisher.npy', F_ab)
