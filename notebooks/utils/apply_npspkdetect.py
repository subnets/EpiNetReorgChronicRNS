"""
Automate standard deviation detector to a data dictionary.

Author: Ankit N. Khambhati
Last Updated: 2018/11/08
"""

import numpy as np
from . import signal_dict
import scipy.signal as sp_sig
import scipy.stats as sp_stats

kernel_dict = {'threshold': np.arange(1, 20.5, 0.5)}


def _create_signal_dict(sample_dict, channel_dict, kernel_dict):

    data_dict = {}

    # Setup signal
    data_dict['signal'] = np.zeros((len(sample_dict['timestamp']),
                                    len(kernel_dict['threshold']),
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


def apply_bpass(signal, fs):

    # Get a copy of the signal
    signal = signal.copy()

    # Get signal attributes
    n_s, n_ch = signal.shape

    # Get butterworth filter parameters
    buttord_params = {'wp': [10, 100],      # Passband 1 Hz
                      'ws': [8, 102],       # Stopband 0.5 Hz
                      'gpass': 3,           # 3dB corner at pass band
                      'gstop': 60,          # 60dB min. attenuation at stop band•
                      'analog': False,      # Digital filter
                      'fs': fs}

    ford, wn = sp_sig.buttord(**buttord_params)

    # Design the filter using second-order sections to ensure better stability
    sos = sp_sig.butter(ford, wn, btype='bandpass', output='sos', fs=fs)

    # Apply zero-phase forward/backward filter signal along the time axis
    signal = sp_sig.sosfiltfilt(sos, signal, axis=0)

    return signal


def apply(data_dict):

    ### Create kernel family
    print('- Constructing line-length kernel')
    Fs = signal_dict.get_fs(data_dict)

    # Get a view of the sample_dict after resampling
    sample_dict = signal_dict.subset(data_dict, channel=[0])

    # Create a placeholder linelength dictionary
    spk_dict = _create_signal_dict(
            sample_dict['sample'], data_dict['channel'], kernel_dict)
    spk_dict['signal'] = np.zeros_like(spk_dict['signal'], dtype=np.bool)

    bp_sig = apply_bpass(data_dict['signal'], Fs)
    bp_sig_zs = sp_stats.zscore(bp_sig, axis=0)

    for thr_i, thr in enumerate(kernel_dict['threshold']):
        spk_dict['signal'][:, thr_i, :] = bp_sig_zs > thr

    return spk_dict
