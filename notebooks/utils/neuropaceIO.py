"""
Library to manipulate neuropace binary data files

Created by: Ankit N. Khambhati
Last Updated: 2019/07/09
"""

import glob
import warnings

import numpy as np
import pandas as pd

BINARY_DTYPE = np.int16
DEVICE_OFFSET = 512
BINARY_INTERLEAVE = True  # Other option is block-wise format
N_DEVICE_CHAN = 4


def read_dat(path_dat, n_chan, n_sample):
    """
    Read DAT Binary file.

    Parameters
    ----------
        path_dat: str
            Path to DAT file.

        n_chan: int
            Number of <valid> channels in the file, probably shouldn't be
            more than N_DEVICE_CHAN.

        n_sample: int
            Number of samples per valid channel in the file.

    Returns
    -------
        data_arr: np.ndarray (dtype: np.float) [n_sample x n_chan]
    """

    if n_chan > N_DEVICE_CHAN:
        raise Exception('Requesting more channels than the device supports.')

    # Load the raw binary file
    raw = np.fromfile(path_dat, dtype=BINARY_DTYPE)

    # Do some error checking of the raw data size
    raw_sample_per_chan = len(raw) / n_chan
    if raw_sample_per_chan != np.round(raw_sample_per_chan):
        raise Exception(
            'Raw data does not contain an even number of samples per channel.')
    if np.abs(raw_sample_per_chan - n_sample) > 1:
        raise Exception(
            'Raw data does not match expected length within one sample.')
    n_sample = int(raw_sample_per_chan)

    # Re-format the raw data
    data_arr = np.nan * np.zeros((n_sample, n_chan))
    for ii in range(n_chan):
        if BINARY_INTERLEAVE:
            data_arr[:, ii] = raw[ii::n_chan]
        else:
            data_arr[:, ii] = raw[(ii * n_sample):((ii + 1) * n_sample)]
    data_arr -= DEVICE_OFFSET

    return data_arr


def find_dat(base_path, subj_name, file_id):
    """
    Find the path to the DAT file within a central repository.
    """

    path_dat = glob.glob('{}/{}_*/{}'.format(base_path, subj_name,
                                                file_id))

    if len(path_dat) == 0:
        warnings.warn('Could not find {} for {}_{}.'.format(
            file_id, subj_name, subj_id))
        return None

    if len(path_dat) > 1:
        warnings.warn('Found multiple {} for {}_{}.'.format(
            file_id, subj_name, subj_id))
        return None

    if len(path_dat) == 1:
        return path_dat[0]


def get_ieeg_data_dict(df_catalog, base_path):
    """
    Generate a data dictionary associated with the entries of df_catalog.

    Parameters
    ----------
        df_catalog: pandas.DataFrame
            A DataFrame from select rows of the master database (csv).
            Catalog will be used to generate a complete data_dict of the
            selected files.

        base_path: str
            Base location of the raw data.

    Returns
    -------
        data_dict: dictionary
            'signal': Contains raw signal with dimensions [T x N]
            'sample': Contains metadata related to sample
            'channel': Contains metadata related to channel
            'axis_ord': Order of axis dimensions in `signal'
    """

    assert type(df_catalog) == pd.DataFrame

    # Setup the output dictionary
    data_dict = {}
    data_dict['signal'] = None
    data_dict['sample'] = {
        'timestamp': None,
        'filename': None,
    }
    data_dict['channel'] = {
        'label': None,
    }
    data_dict['axis_ord'] = np.array(['sample', 'channel'])

    # Setup channel information based on the supplied catalog dataframe
    lbl_full = np.unique([
        df_catalog['Ch {} name'.format(ii + 1)].unique()
        for ii in range(N_DEVICE_CHAN)
    ])
    data_dict['channel']['label'] = lbl_full

    # Setup placeholders based on the supplied catalog dataframe
    data_dict['sample']['timestamp'] = np.array([])
    data_dict['sample']['filename'] = np.array([])
    data_dict['signal'] = np.empty((0, len(lbl_full)))

    # Iterate over catalog and add data to the dictionary
    for sel_ii, sel in df_catalog.iterrows():
        n_s = sel['Sampling rate'] * sel['ECoG length']
        n_c = sel['Waveform count']
        path_bin = find_dat(base_path, sel['NP_code'],
                            sel['Filename'])

        n_s = np.round(n_s)
        try:
            data = read_dat(path_bin, n_c, n_s)
            n_s = data.shape[0]
        except Exception as E:
            print(E)
            warnings.warn('Skip loading {} for {}'.format(
                sel['Filename'], sel['NP_code']))
            continue

        # Cross-reference the selected catalog enabled channels with data_dict
        data_full = np.nan * np.zeros((n_s, len(lbl_full)))
        ix = 0
        for ii in range(N_DEVICE_CHAN):
            if sel['Ch {} enabled'.format(ii + 1)] != 'On':
                continue
            ch_ix = np.flatnonzero(
                lbl_full == sel['Ch {} name'.format(ii + 1)])

            data_full[:, ch_ix] = data[:, [ix]]
            ix += 1

        # Update signal with data
        data_dict['signal'] = np.concatenate(
            (data_dict['signal'], data_full), axis=0)

        # Update sample
        data_dict['sample']['timestamp'] = np.concatenate(
            (data_dict['sample']['timestamp'],
             pd.Timestamp(sel['Raw UTC Timestamp']).value * 1e-9 +
             np.arange(n_s) / sel['Sampling rate']),
            axis=0)
        data_dict['sample']['filename'] = np.concatenate(
            (data_dict['sample']['filename'], np.repeat(sel['Filename'], n_s)),
            axis=0)

    return data_dict


def get_hourly_data_dict(csv_path):
    """
    Generate a Pandas DataFrame containing hourly count data.

    Parameters
    ----------
        csv_path: str
            Location of CSV containing hourly count data from NeuroPace.

    Returns
    -------
        df_hour: pandas.DataFrame
    """

    df_hour = pd.read_csv(csv_path, skiprows=3)
    df_hour = df_hour.rename(columns={'utc_start_time': 'Raw UTC Timestamp'})
    df_hour['Raw UTC Timestamp'] = pd.to_datetime(df_hour['Raw UTC Timestamp'])
    df_hour = df_hour.set_index('Raw UTC Timestamp', drop=True)

    return df_hour
