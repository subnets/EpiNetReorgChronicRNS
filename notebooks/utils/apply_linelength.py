"""
Automate line-length pipeline to a data dictionary.

Author: Ankit N. Khambhati
Last Updated: 2018/11/08
"""

import numpy as np
from . import signal_dict
import pyfftw
import pyeisen
import functools
print = functools.partial(print, flush=True)

DUR = 0.04
N_KERN = 1
MEM_FFT = True


def _reserve_fftw_mem(kernel_len, signal_len, n_kernel=N_KERN, threads=6):
    a = pyfftw.empty_aligned((kernel_len + signal_len, n_kernel), dtype=complex)
    fft = pyfftw.builders.fft(a, axis=0, threads=threads)
    ifft = pyfftw.builders.ifft(a, axis=0, threads=threads)

    return fft, ifft


def _create_signal_dict(sample_dict, channel_dict, kernel_dict):

    data_dict = {}

    # Setup signal
    data_dict['signal'] = np.zeros((len(sample_dict['timestamp']),
                                    len(kernel_dict['duration']),
                                    len(channel_dict['label'])),
                                   dtype=np.float)

    # Setup axis ord
    data_dict['axis_ord'] = np.array(['sample', 'kernel', 'channel'])

    # Copy each dict
    for name, name_dict in [['sample', sample_dict],
                            ['channel', channel_dict],
                            ['kernel', kernel_dict]]:

        data_dict[name] = {}
        for key in name_dict:
            data_dict[name][key] = name_dict[key].copy()

    return data_dict


def apply(data_dict, ll_dur=DUR, fs_resample=4/DUR, mem_fft=MEM_FFT):

    ### Create kernel family
    print('- Constructing line-length kernel')
    Fs = signal_dict.get_fs(data_dict)
    family = {'kernel': np.ones((N_KERN, int(ll_dur * Fs))),
              'linelength': {'duration': np.array([ll_dur])},
              'sample': {'time': np.arange(int(ll_dur * Fs)) / Fs},
              'axis_ord': np.array(['linelength', 'sample'])}

    ### Memory setup for convolution
    fft, ifft = [None, None]
    if mem_fft:
        print('- Reserving memory for kernel convolution')
        fft, ifft = _reserve_fftw_mem(
                kernel_len=family['kernel'].shape[1],
                signal_len=len(data_dict['sample']['timestamp']))


    ### Create the linelength dictionary
    print('- Reserving memory for storing kernel coefficients')
    # Get a view of the sample_dict after resampling
    sample_dict = signal_dict.subset(data_dict, channel=[0])
    signal_dict.decimate(sample_dict, fs_new=fs_resample)
    ds_fac = int(np.round(Fs / signal_dict.get_fs(sample_dict)))

    # Create a placeholder linelength dictionary
    ll_dict = _create_signal_dict(
            sample_dict['sample'], data_dict['channel'], family['linelength'])

    # Iterate over each channel and convolve
    print('- Iteratively convolving line-length kernel with each channel')
    for ch_ii, ch in enumerate(data_dict['channel']['label']):
        print('    - {} of {} :: {}'.format(ch_ii+1, 
            len(data_dict['channel']['label']), ch))

        # Subset the channel
        subset_dict = signal_dict.subset(data_dict, channel=[ch_ii])
        subset_sig = subset_dict['signal']

        # Absolute first-order difference
        subset_sig[:, 0] = np.concatenate(([0], np.abs(np.diff(subset_sig[:, 0]))))

        if (fft is not None) & (ifft is not None):
            out = pyeisen.convolve.fconv(
                    family['kernel'][:,:].T,
                    subset_dict['signal'][:, 0].reshape(-1, 1),
                    fft=fft, ifft=ifft,
                    interp_nan=True)
        else:
            out = pyeisen.convolve.fconv(
                    family['kernel'][:,:].T,
                    subset_dict['signal'][:, ch_ii].reshape(-1, 1),
                    fft=fft, ifft=ifft,
                    interp_nan=True)

        ll_dict['signal'][:, :, ch_ii] = np.abs(out[::ds_fac, :, :][:, :, 0])

    return ll_dict
