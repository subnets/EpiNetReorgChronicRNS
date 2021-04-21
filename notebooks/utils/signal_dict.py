"""
Utilities to work with the signal data_dict.

Author: Ankit N. Khambhati
Last Updated: 2019/03/17
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig


def load_data_dict(dd_path):
        df = np.load(dd_path, allow_pickle=True)
        data_dict = {}
        for d in df:
            data_dict[d] = df[d][()]
        return data_dict


def check_dict_layout(data_dict):
    """Check that the data_dict conforms to the basic template."""

    assert 'signal' in data_dict
    assert 'axis_ord' in data_dict

    assert 'sample' in data_dict
    assert 'channel' in data_dict
    assert 'sample' in data_dict['axis_ord']
    assert 'channel' in data_dict['axis_ord']

    assert 'timestamp' in data_dict['sample']
    assert 'label' in data_dict['channel']

    sig_shape = data_dict['signal'].shape
    n_ts = len(data_dict['sample']['timestamp'])
    ax = get_axis(data_dict, 'sample')
    assert n_ts == sig_shape[ax]

    n_ch = len(data_dict['channel']['label'])
    ax = get_axis(data_dict, 'channel')
    assert n_ch == sig_shape[ax]

    for key in data_dict['sample']:
        assert n_ts == len(data_dict['sample'][key])

    for key in data_dict['channel']:
        assert n_ch == len(data_dict['channel'][key])


def get_axis(data_dict, lbl):
    if (type(lbl) != str) or (lbl not in data_dict['axis_ord']):
        raise Exception('Cannot locate axis label `{}`.'.format(lbl))
    return np.flatnonzero(data_dict['axis_ord'] == lbl)[0]


def pivot_axis(data_dict, lbl):
    """Transpose axis with lbl to the first dimension"""

    piv_ax = get_axis(data_dict, lbl)
    all_ax = np.arange(len(data_dict['axis_ord']))
    reord_ax = np.insert(np.delete(all_ax, piv_ax), 0, piv_ax)

    data_dict['axis_ord'] = data_dict['axis_ord'][reord_ax]
    data_dict['signal'] = np.transpose(data_dict['signal'], reord_ax)


def get_fs(data_dict):
    return np.median(1 / np.diff(data_dict['sample']['timestamp']))


def subset(data_dict, **kwargs):
    """
    Retrieve a data_dict copy corresponding to a data subset.

    Parameters
    ----------
        kwargs: keys correspond to axis labels, with contents of index lists or slices

    Return
    ------
        data_dict: a DEEP copy of the data_dict corresponding to the data subset.
    """

    slice_inds = ()
    for k_ii, key in enumerate(data_dict['axis_ord']):
        if key in kwargs:
            slice_inds += (kwargs[key], )
        else:
            slice_inds += (slice(0, data_dict['signal'].shape[k_ii]), )

    ### Manual procedure, no practical way of dimension checking
    # Make a DEEEEP copy of the dictionary
    new_dict = {}
    new_dict['signal'] = data_dict['signal'][slice_inds]
    new_dict['axis_ord'] = data_dict['axis_ord'][...]

    for k_ii, key in enumerate(data_dict['axis_ord']):
        new_dict[key] = {}

        for k_jj, subkey in enumerate(data_dict[key]):
            new_dict[key][subkey] = data_dict[key][subkey][slice_inds[k_ii]]

    check_dict_layout(new_dict)

    return new_dict


def combine(data_dicts, lbl):
    """
    Concatenate a list of data_dicts along an existing label dimension.

    Parameters
    ----------
        kwargs: keys correspond to axis labels, with contents of index lists or slices

    Return
    ------
        data_dict: a DEEP copy of the data_dict corresponding to the data subset.
    """

    # Check all axis ords match up
    for d in data_dicts:
        check_dict_layout(d)

    # Check label dimension is consistent
    lbl_axs = []
    for d in data_dicts:
        lbl_axs.append(get_axis(data_dicts[0], lbl))
    assert len(np.unique(lbl_axs)) == 1
    lbl_ax = lbl_axs[0]

    ### Manual procedure, no practical way of dimension checking
    # Make a DEEEEP copy of the dictionary
    new_dict = {}
    new_dict['signal'] = np.concatenate(
        [d['signal'] for d in data_dicts], axis=lbl_ax)
    new_dict['axis_ord'] = data_dicts[0]['axis_ord'][...]

    for k_ii, key in enumerate(data_dicts[0]['axis_ord']):
        new_dict[key] = {}

        if lbl != key:
            new_dict[key] = data_dicts[0][key]
        else:
            for k_jj, subkey in enumerate(data_dicts[0][key]):
                # Accumulate key data across concatenation list dicts
                subkey_arr = np.concatenate(
                    [d[key][subkey] for d in data_dicts], axis=0)
                new_dict[key][subkey] = subkey_arr

    check_dict_layout(new_dict)

    return new_dict


def common_avg_reref(data_dict, channel_dist=None, channel_group=None):
    """Re-reference the signal array to the (weighted) common average.

    Parameters
    ----------
        channel_dist: numpy.ndarray, shape: [n_chan x n_chan]
            Array specifying inter-channel distances
            (e.g. euclidean, geodesic, etc). If None, no distance-weighting is
            performed.
    """

    pivot_axis(data_dict, 'channel')

    if channel_group is None:
        channel_group = np.ones(data_dict['signal'].shape[0])
    assert data_dict['signal'].shape[0] == len(channel_group)

    for grp_id in np.unique(channel_group):
        grp_ix = np.flatnonzero(channel_group == grp_id)

        if type(channel_dist) == np.ndarray:
            channel_dist_grp = channel_dist[grp_ix, :][:, grp_ix]

            chan_prox = 1 / (channel_dist_grp)
            chan_prox[np.isinf(chan_prox)] = 0
            chan_prox /= chan_prox.sum(axis=1)
            common = np.tensordot(chan_prox, data_dict['signal'][grp_ix], axes=1)
        else:
            common = data_dict['signal'][grp_ix].mean(axis=0)
        data_dict['signal'][grp_ix] -= common

    pivot_axis(data_dict, 'sample')


def decimate(data_dict, fs_new):
    """Decimate the signal array, and anti-alias filter."""

    ts_ax = get_axis(data_dict, 'sample')
    fs = get_fs(data_dict)

    q = int(np.round(fs / fs_new))
    fs = fs / q

    data_dict['signal'] = sig.decimate(
        data_dict['signal'], q=q, ftype='fir', zero_phase=True, axis=ts_ax)

    n_ts = data_dict['signal'].shape[ts_ax]
    for subkey in data_dict['sample']:
        data_dict['sample'][subkey] = data_dict['sample'][subkey][::q]


def notchline(data_dict, freq_list, bw=2, harm=True):
    """Notch filter the line noise and harmonics"""

    ts_ax = get_axis(data_dict, 'sample')

    fs = get_fs(data_dict)
    nyq_fs = fs / 2

    freq_list = np.unique(freq_list)
    freq_list = freq_list[freq_list > 0]
    freq_list = freq_list[freq_list < nyq_fs]

    for ff in freq_list:
        if (ff + bw) >= nyq_fs:
            continue
        b, a = sig.iirnotch(ff / nyq_fs, ff / bw)
        data_dict['signal'] = sig.filtfilt(
            b, a, data_dict['signal'], axis=ts_ax)


def zscore(data_dict, lbl, method='robust', scale=1.4826):
    """Z-Score the signal along the provided label"""

    pivot_axis(data_dict, lbl)

    if method == 'robust':
        dev = data_dict['signal'] - np.nanmedian(data_dict['signal'], axis=0)
        med_abs_dev = scale * np.nanmedian(np.abs(dev), axis=0)
        data_dict['signal'] = dev / med_abs_dev

    if method == 'standard':
        data_dict['signal'] -= np.nanmean(data_dict['signal'], axis=0)
        data_dict['signal'] /= np.nanstd(data_dict['signal'], axis=0)


def plot_time_stacked(data_dict, ax):
    """Plot of the normalized signal in a stacked montage."""

    sig = data_dict['signal'][...]
    sig_Z = (sig - np.nanmean(sig, axis=0)) / np.nanstd(sig, axis=0)

    offset = np.arange(sig_Z.shape[1]) * 3

    for ch, sig_ch in enumerate(sig_Z.T):
        ax.plot(
            data_dict['sample']['timestamp'],
            sig_ch + offset[ch],
            color='b',
            alpha=0.5,
            linewidth=0.5)

        ax.hlines(
            offset[ch],
            data_dict['sample']['timestamp'][0],
            data_dict['sample']['timestamp'][-1],
            color='k',
            alpha=0.5,
            linewidth=0.1)

    ax.set_yticks(offset)
    ax.set_yticklabels(data_dict['channel']['label'])

    ax.set_xlim([
        data_dict['sample']['timestamp'][0],
        data_dict['sample']['timestamp'][0] + 10
    ])
    #    ax.set_ylim([
    return ax
