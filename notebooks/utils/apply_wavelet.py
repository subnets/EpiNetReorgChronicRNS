"""
Automate wavelet pipeline to a data dictionary.

Author: Ankit N. Khambhati
Last Updated: 2018/10/25
"""

import functools

import numpy as np

import pyeisen
import pyfftw

from . import signal_dict

print = functools.partial(print, flush=True)

FREQ_LOW = 3
FREQ_HIGH = 96
N_CYCLE = 6
N_KERN = 25
MEM_FFT = True


def _reserve_fftw_mem(kernel_len, signal_len, n_kernel=N_KERN, threads=6):
    a = pyfftw.empty_aligned(
        (kernel_len + signal_len, n_kernel), dtype=complex)
    fft = pyfftw.builders.fft(a, axis=0, threads=threads)
    ifft = pyfftw.builders.ifft(a, axis=0, threads=threads)

    return fft, ifft


def _create_signal_dict(sample_dict, channel_dict, wavelet_dict):

    data_dict = {}

    # Setup signal
    data_dict['signal'] = np.zeros(
        (len(sample_dict['timestamp']), len(wavelet_dict['freqs']),
         len(channel_dict['label'])),
        dtype=np.complex)

    # Setup axis ord
    data_dict['axis_ord'] = np.array(['sample', 'wavelet', 'channel'])

    # Copy each dict
    for name, name_dict in [['sample', sample_dict], ['channel', channel_dict],
                            ['wavelet', wavelet_dict]]:

        data_dict[name] = {}
        for key in name_dict:
            data_dict[name][key] = name_dict[key].copy()

    return data_dict


def apply(data_dict,
          fs_resample=4 * (FREQ_HIGH / N_CYCLE),
          wv_freqs=np.logspace(
              np.log10(FREQ_HIGH), np.log10(FREQ_LOW), N_KERN),
          wv_cycles=N_CYCLE * np.ones(N_KERN),
          mem_fft=MEM_FFT):

    ### Prepare data_dict
    ### Create kernel family
    print('- Constructing wavelet kernels')
    Fs = signal_dict.get_fs(data_dict)
    family = pyeisen.family.morlet(freqs=wv_freqs, cycles=wv_cycles, Fs=Fs)

    fft, ifft = [None, None]
    if mem_fft:
        print('- Reserving memory for wavelet convolution')
        fft, ifft = _reserve_fftw_mem(
            kernel_len=family['kernel'].shape[1],
            signal_len=len(data_dict['sample']['timestamp']))

    ### Create the wavelet dictionary
    print('- Reserving memory for storing wavelet coefficients')
    # Get a view of the sample_dict after resampling
    sample_dict = signal_dict.subset(data_dict, channel=[0])
    signal_dict.decimate(sample_dict, fs_new=fs_resample)
    ds_fac = int(np.round(Fs / signal_dict.get_fs(sample_dict)))

    # Create a placeholder wavelet dictionary
    wavelet_dict = _create_signal_dict(sample_dict['sample'],
                                       data_dict['channel'], family['wavelet'])

    # Iterate over each channel and convolve
    print('- Iteratively convolving wavelet with each channel')
    for ch_ii, ch in enumerate(data_dict['channel']['label']):
        print('    - {} of {} :: {}'.format(ch_ii + 1,
                                            len(data_dict['channel']['label']),
                                            ch))

        if (fft is not None) & (ifft is not None):
            out = pyeisen.convolve.fconv(
                family['kernel'][:, :].T,
                data_dict['signal'][:, ch_ii].reshape(-1, 1),
                fft=fft,
                ifft=ifft,
                interp_nan=True)
        else:
            out = pyeisen.convolve.fconv(
                family['kernel'][:, :].T,
                data_dict['signal'][:, ch_ii].reshape(-1, 1),
                fft=fft,
                ifft=ifft,
                interp_nan=True)

        wavelet_dict['signal'][:, :, ch_ii] = out[::ds_fac, :, :][:, :, 0]

    return wavelet_dict


def calc_pow(data_dict):
    assert 'wavelet' in data_dict['axis_ord']

    # Get axes sizes
    n_ts = data_dict['sample']['timestamp'].shape[0]

    # Compute power magnitude
    X_a = np.abs(data_dict['signal'])**2
    X_a = X_a.mean(axis=0)

    # Create a new data_dict
    pow_dict = {}
    pow_dict['signal'] = np.expand_dims(X_a, 0)
    pow_dict['axis_ord'] = np.array(['sample', 'wavelet', 'channel'])
    pow_dict['wavelet'] = data_dict['wavelet'].copy()
    pow_dict['channel'] = data_dict['channel'].copy()
    pow_dict['sample'] = data_dict['sample'].copy()

    # Truncate first sample
    for k in pow_dict['sample']:
        pow_dict['sample'][k] = pow_dict['sample'][k][:1]

    return pow_dict


def calc_plv(data_dict, cross_freq=False, imag=False):
    assert 'wavelet' in data_dict['axis_ord']

    # Get axes sizes
    n_ts = data_dict['sample']['timestamp'].shape[0]
    n_wv = data_dict['wavelet']['freqs'].shape[0]
    n_ch = data_dict['channel']['label'].shape[0]

    # Normalize signal to unit magnitude
    X_cn = data_dict['signal'].copy()
    X_a = np.abs(data_dict['signal'])
    X_cn /= X_a

    # Check NaN
    if np.isnan(X_cn).any():
        X_cn_masked = np.ma.masked_invalid(X_cn).reshape(n_ts, n_wv*n_ch)
        dotted = np.ma.dot(X_cn_masked.T, np.conj(X_cn_masked))
        if imag:
            X_plv = np.abs(dotted.imag / n_ts)
        else:
            X_plv = np.abs(dotted / n_ts)
        X_plv = X_plv.reshape(n_wv, n_ch, n_wv, n_ch)
    else:
        dotted = np.tensordot(X_cn, np.conj(X_cn), axes=((0), (0)))
        if imag:
            X_plv = np.abs(dotted.imag / n_ts)
        else:
            X_plv = np.abs(dotted / n_ts)
    X_plv = np.transpose(X_plv, (0, 2, 1, 3))

    if not cross_freq:
        X_plv = X_plv[np.arange(n_wv), np.arange(n_wv), :, :]

    # Create a new data_dict
    plv_dict = {}
    plv_dict['signal'] = np.expand_dims(X_plv, 0)
    plv_dict['sample'] = data_dict['sample'].copy()
    plv_dict['wavelet'] = data_dict['wavelet'].copy()
    plv_dict['channel'] = data_dict['channel'].copy()

    if cross_freq:
        plv_dict['axis_ord'] = np.array(
            ['sample', 'wavelet', 'wavelet', 'channel', 'channel'])
    else:
        plv_dict['axis_ord'] = np.array(
            ['sample', 'wavelet', 'channel', 'channel'])

    # Truncate first sample
    for k in plv_dict['sample']:
        plv_dict['sample'][k] = plv_dict['sample'][k][-1:]

    return plv_dict

