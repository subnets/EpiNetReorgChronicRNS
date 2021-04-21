"""
Utilities to blank stimulation pulses.

Author: Ankit N. Khambhati
Last Updated: 2019/07/05
"""

import numpy as np
import scipy.stats as stats


def nan_hold_sample(data_dict, hold_sample=10, pad_sample=0):
    assert hold_sample == int(hold_sample)
    assert pad_sample == int(pad_sample)

    # Find where the signal derivatives are non-zero
    thr = np.diff(data_dict['signal'].mean(axis=1))
    thr[thr != 0] = 1
    ix_on = np.flatnonzero(np.diff(thr) < 0) + 1
    ix_off = np.flatnonzero(np.diff(thr) > 0) + 1

    # Abort if there are no onsets or no offsets
    if (len(ix_on) == 0) and (len(ix_off) > 0):
        ix_on = np.array([0])
    if (len(ix_off) == 0) and (len(ix_on) > 0):
        ix_off = np.array([data_dict['signal'].shape[0]-1])
    if (len(ix_on) == 0) and (len(ix_off) == 0):
        return None

    # Handle any mismatch cases where there are uneven numbers of onsets or offsets
    if ix_on[0] > ix_off[0]:
        ix_off = ix_off[1:]
    if ix_off[-1] < ix_on[-1]:
        ix_on = ix_on[:-1]
    assert len(ix_on) == len(ix_off)

    # NaN each detected hold_signal
    detect_dict = {'onset_ix': [],
                   'offset_ix': []}
    for i1, i2 in zip(ix_on, ix_off):
        if (i2 - i1) < hold_sample:
            continue
        i2 += int(pad_sample)

        # Make sure that the i1, i2 range is within the signal shape
        if (i2+1) >= data_dict['signal'].shape[0]:
            data_dict['signal'][i1:, :] = np.nan
            continue

        detect_dict['onset_ix'].append(i1)
        detect_dict['offset_ix'].append(i2)

        for ch in range(data_dict['signal'].shape[1]):
            m, yint, _, _, _ = stats.linregress([i1-1, i2+1],
                    [data_dict['signal'][i1-1, ch],
                     data_dict['signal'][i2+1, ch]])
            data_dict['signal'][i1:i2+1, ch] = m*np.arange(i1, i2+1) + yint

    return detect_dict
