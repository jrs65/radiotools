"""Tools for reading MIRIAD formatted data.
"""

import aipy

import numpy as np


def read_dict(path):
    """Read in a MIRIAD dataset.

    Parameters
    ----------
    path : string
        Path to MIRIAD dataset.

    Returns
    -------
    data : dict
        Dictionary containing the contents of the MIRIAD file.

    Notes
    -----
    Data dictionary entries are:

    - data
    - mask
    - time
    - length
    - uvw [in metres]
    - ant
    - pol
    - freq [in GHz]
    """

    # Open the dataset
    miriad_data = aipy.miriad.UV(path)

    # Construct the set of frequency channels (in GHz)
    nfreq = miriad_data['nchan']
    delta_freq = miriad_data['sdf']  # GHz
    sfreq = miriad_data['sfreq']  # GHz
    freq = np.arange(nfreq) * delta_freq + sfreq

    # TODO: should generalise this to select other polarisation types
    miriad_data.select('polarization', -8, -5, include=True)
    miriad_data.select('polarization', -7, -5, include=True)
    miriad_data.select('polarization', -6, -5, include=True)
    miriad_data.select('polarization', -5, -5, include=True)

    miriad_data.rewind()

    data, mask, times, lengths, uvw, ant, pol = [], [], [], [], [], [], []

    # Iterate over all entries in MIRIAD dataset and pull out their useful
    # quantities
    for pream, data_row, mask_row in miriad_data.all(raw=True):

        # Ensure that data arrays are of the correct type
        data_row = data_row.astype(np.complex64)
        mask_row = mask_row.astype(np.bool)

        # Unpack co-ordinates
        uvw_row, t, ant_row = pream
        pp = aipy.miriad.pol2str[miriad_data['pol']]

        # Append this rows data to the global set
        lengths.append(len(data))
        times.append(t)
        ant.append(ant_row)
        uvw.append(uvw_row)
        data.append(data_row)
        mask.append(mask_row)
        pol.append(pp)

    data_dict = {
        'data': np.array(data),
        'mask': np.array(mask),
        'time': np.array(times),
        'length': np.array(lengths),
        'uvw': np.array(uvw),
        'ant': np.array(ant),
        'pol': np.array(pol),
        'freq': freq
    }

    return data_dict


def read(path, stokes=True):
    """More advanced file reader.


    Parameters
    ----------
    stokes : boolean, optional
        Convert polarization into instrumental Stokes.

    Returns
    -------
    data : dict

    Notes
    -----
    This will massage the data into a set of useful arrays in a dictionary.

    - vis[baseline, polarisation, time, freq]
    - weight[baseline, polarisation, time, freq]
    - uvw[baseline, time, uvw]
    - time[time]
    - freq[freq]
    - pair[baseline, 2]
    """

    data = read_dict(path)

    u_time, i_time = np.unique(data['time'], return_inverse=True)
    ntime = len(u_time)

    u_pol, i_pol = np.unique(data['pol'], return_inverse=True)
    npol = len(u_pol)

    u_pair, i_pair = np.unique(data['ant'][:, 0] + 1.0J * data['ant'][:, 1], return_inverse=True)
    npair = len(u_pair)
    u_pair = np.dstack((u_pair.real, u_pair.imag))[0]

    nfreq = len(data['freq'])

    vis_data = np.zeros((npair, npol, ntime, nfreq), dtype=data['data'].dtype)
    uvw = np.zeros((npair, ntime, 3), dtype=np.float64)
    weight = np.zeros((npair, npol, ntime, nfreq), dtype=np.bool)

    for vis_ind in range(data['data'].shape[0]):

        # Decode indices
        time_ind = i_time[vis_ind]
        pol_ind = i_pol[vis_ind]
        pair_ind = i_pair[vis_ind]

        # Reassign data
        vis_data[pair_ind, pol_ind, time_ind] = data['data'][vis_ind]
        uvw[pair_ind, time_ind] = data['uvw'][vis_ind] / np.pi  # Conversion to m
        weight[pair_ind, pol_ind, time_ind] = (~data['mask'][vis_ind]).astype(np.float64)

    # Generate Stokes visibilities.
    # Missing polarisations cause all Stokes parameters containing them to masked out.
    if stokes:

        vis_data_stokes = vis_data.copy()

        vis_data_stokes[:, 0] = vis_data[:, 0] + vis_data[:, 3]  # I
        vis_data_stokes[:, 1] = vis_data[:, 0] - vis_data[:, 3]  # Q
        vis_data_stokes[:, 2] = vis_data[:, 2] + vis_data[:, 1]  # U
        vis_data_stokes[:, 3] = vis_data[:, 2] - vis_data[:, 1]  # V

        weight_stokes = weight.copy()
        weight_stokes[:, 0] = np.logical_and(weight[:, 0], weight[:, 3])
        weight_stokes[:, 1] = np.logical_and(weight[:, 0], weight[:, 3])
        weight_stokes[:, 2] = np.logical_and(weight[:, 1], weight[:, 2])
        weight_stokes[:, 3] = np.logical_and(weight[:, 1], weight[:, 2])

        vis_data = vis_data_stokes
        weight = weight_stokes

        u_pol = np.array(['I', 'Q', 'U', 'V'])

    new_data = {
        'pair': u_pair,
        'pol': u_pol,
        'time': u_time,
        'freq': data['freq'],

        'uvw': uvw,
        'vis': vis_data,
        'weight': weight
    }

    return new_data
