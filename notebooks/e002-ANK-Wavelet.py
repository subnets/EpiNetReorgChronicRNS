"""
Functions for applying wavelet analyses to NeuroPace data.

Created by: Ankit N. Khambhati
Updated: 2019/03/19
"""


import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings('ignore')

exec(open('__init__.py').read())
rs_path = path['RSRCH'](2)
df_meta = path['CORE']['RNS']['METADATA']
df_catalog = path['CORE']['RNS']['CATALOG']
df_catalog['Timestamp'] = pd.to_datetime(
    df_catalog['Timestamp'], errors='coerce')
df_catalog = df_catalog.dropna()

import importlib
e001_LL = importlib.import_module('e001-ANK-Line_Length')
e000_stim = importlib.import_module('e000-ANK-Stim_Blank')
PRE_TRIGGER_DUR = 30


def wavelet_name(subj_id, subj_fname):
    return '{}/Wavelet.{}.{}'.format(rs_path, subj_id, subj_fname)


def wavelet_trigger_name(subj_id, subj_fname, trigger_name):
    return '{}/{}/Wavelet.{}.{}'.format(rs_path, trigger_name, subj_id, subj_fname)


def wavelet_extraction(catalog_index):
    sel = df_catalog.iloc[[catalog_index]]
    out_name = wavelet_name(sel['NP_code'].iloc[0], sel['Filename'].iloc[0])

    out_path = '{}.npz'.format(out_name)
    log_path = '{}.log'.format(out_name)
    if os.path.exists(out_path):
        return None

    sys.stdout = open(log_path, 'w')

    dd = utils.neuropaceIO.get_ieeg_data_dict(sel, path['CORE']['RNS']['BASE'])
    Fs = utils.signal_dict.get_fs(dd)
    utils.stim_blank.nan_hold_sample(
        dd, hold_sample=10, pad_sample=int(Fs * 0.1))
    dd_wavelet = utils.apply_wavelet.apply(dd)

    np.savez(out_path, **dd_wavelet)


def resave_wavelet_extraction_as_mat(catalog_index,  trigger_name='Scheduled'):
    sel = df_catalog.iloc[[catalog_index]]
    if sel['ECoG trigger'].iloc[0] != trigger_name:
        return None

    npz_name = '{}.npz'.format(wavelet_name(sel['NP_code'].iloc[0], sel['Filename'].iloc[0]))
    mat_name = wavelet_trigger_name(sel['NP_code'].iloc[0], sel['Filename'].iloc[0], trigger_name)

    out_path = '{}.mat'.format(mat_name)
    log_path = '{}.log'.format(mat_name)
    if os.path.exists(out_path):
        return None

    data = np.load(npz_name)
    sp_io.savemat(out_path, data)


def plv_pretrigger_name(subj_id):
    return '{}/PLV.Pre_Trigger.{}'.format(rs_path, subj_id)


def pre_trigger_PLV(np_code):
    sel_clip = df_catalog[df_catalog['NP_code'] == np_code]
    sel_clip = sel_clip[sel_clip['ECoG Pre-trigger length'] >= PRE_TRIGGER_DUR]

    out_path = '{}.npz'.format(plv_pretrigger_name(np_code))
    log_path = '{}.log'.format(plv_pretrigger_name(np_code))
    #if os.path.exists(out_path):
    #    return None

    sys.stdout = open(log_path, 'w')

    plv_dict = {
        'signal': [],
        'sample': {
            'timestamp': [],
            'filename': [],
            'trigger': []
        },
        'wavelet': {},
        'channel': {
            'label': []
        },
        'axis_ord': np.array(['sample', 'wavelet', 'channel'])
    }

    triu_ix, triu_iy = np.triu_indices(4, k=1)

    for sel_i, sel in enumerate(sel_clip.iterrows()):
        print('{} of {}'.format(sel_i + 1, len(sel_clip)))

        try:
            df = utils.signal_dict.load_data_dict('{}.npz'.format(
                wavelet_name(sel[1]['NP_code'], sel[1]['Filename'])))
        except Exception as error:
            print('{} -- {}'.format(sel_i, error))
            continue

        # Check all channels are enabled
        enabled = [
            True if sel[1]['Ch {} enabled'.format(ch)] == 'On' else False
            for ch in range(1, 5)
        ]
        if not all(enabled):
            print('{} -- All channels not enabled'.format(sel_i))
            continue

        # Get the pre-trigger epoch
        Fs = utils.signal_dict.get_fs(df)
        n_pretrig = int(Fs * PRE_TRIGGER_DUR)
        df = utils.signal_dict.subset(df, sample=slice(0, n_pretrig))

        # Compute phase-locking during the whole pre-trigger epoch
        df_plv = utils.apply_wavelet.calc_plv(df, cross_freq=False, imag=False)

        # Add all necessary info to the plv_dict
        plv_dict['signal'].append(
            df_plv['signal'][0, ...][:, triu_ix, triu_iy])
        plv_dict['sample']['timestamp'].append(
            df_plv['sample']['timestamp'][0])
        plv_dict['sample']['filename'].append(df_plv['sample']['filename'][0])
        plv_dict['sample']['trigger'].append(sel[1]['ECoG trigger'])

    plv_dict['signal'] = np.array(plv_dict['signal'])
    plv_dict['sample']['timestamp'] = np.array(plv_dict['sample']['timestamp'])
    plv_dict['sample']['filename'] = np.array(plv_dict['sample']['filename'])
    plv_dict['sample']['trigger'] = np.array(plv_dict['sample']['trigger'])
    plv_dict['wavelet'] = df['wavelet']

    for trix, triy in zip(triu_ix, triu_iy):
        plv_dict['channel']['label'].append('{}.{}'.format(
            df['channel']['label'][trix], df['channel']['label'][triy]))
    plv_dict['channel']['label'] = np.array(plv_dict['channel']['label'])

    np.savez(out_path, **plv_dict)


def plvdespiked_pretrigger_name(subj_id):
    return '{}/PLVDespiked.Pre_Trigger.{}'.format(rs_path, subj_id)


def pre_trigger_PLV_despiked(np_code):
    sel_clip = df_catalog[df_catalog['NP_code'] == np_code]
    sel_clip = sel_clip[sel_clip['ECoG Pre-trigger length'] >= PRE_TRIGGER_DUR]

    out_path = '{}.npz'.format(plvdespiked_pretrigger_name(np_code))
    log_path = '{}.log'.format(plvdespiked_pretrigger_name(np_code))
    #if os.path.exists(out_path):
    #    return None

    sys.stdout = open(log_path, 'w')

    plv_dict = {
        'signal': [],
        'sample': {
            'timestamp': [],
            'filename': [],
            'trigger': []
        },
        'wavelet': {},
        'channel': {
            'label': []
        },
        'axis_ord': np.array(['sample', 'wavelet', 'channel'])
    }

    triu_ix, triu_iy = np.triu_indices(4, k=1)

    for sel_i, sel in enumerate(sel_clip.iterrows()):
        print('{} of {}'.format(sel_i + 1, len(sel_clip)))

        try:
            df = utils.signal_dict.load_data_dict('{}.npz'.format(
                wavelet_name(sel[1]['NP_code'], sel[1]['Filename'])))
            df_spk = utils.signal_dict.load_data_dict('{}.npz'.format(
                e001_LL.spiketimes_name(sel[1]['NP_code'], sel[1]['Filename'])))
        except Exception as error:
            print('{} -- {}'.format(sel_i, error))
            continue

        # Check all channels are enabled
        enabled = [
            True if sel[1]['Ch {} enabled'.format(ch)] == 'On' else False
            for ch in range(1, 5)
        ]
        if not all(enabled):
            print('{} -- All channels not enabled'.format(sel_i))
            continue

        # Get the pre-trigger epoch
        Fs = utils.signal_dict.get_fs(df)
        n_pretrig = int(Fs * PRE_TRIGGER_DUR)
        df = utils.signal_dict.subset(df, sample=slice(0, n_pretrig))

        # Despike
        thr_ix = np.flatnonzero(df_spk['threshold']['name'] == 4)[0]
        spk_inds = np.array(df_spk['signal'][thr_ix, 0, :])
        pad_len = int(Fs*0.25)
        for ch_ix in range(len(spk_inds)):
            for ind in spk_inds[ch_ix]:
                ind = int(np.round(ind*Fs))
                if ((ind+pad_len) > df['signal'].shape[0]):
                    continue
                if ((ind-pad_len < 0)):
                    continue
                df['signal'][ind-pad_len:ind+pad_len, :, ch_ix] = np.nan

        # Compute phase-locking during the whole pre-trigger epoch
        df_plv = utils.apply_wavelet.calc_plv(df, cross_freq=False, imag=False)

        # Add all necessary info to the plv_dict
        plv_dict['signal'].append(
            df_plv['signal'][0, ...][:, triu_ix, triu_iy])
        plv_dict['sample']['timestamp'].append(
            df_plv['sample']['timestamp'][0])
        plv_dict['sample']['filename'].append(df_plv['sample']['filename'][0])
        plv_dict['sample']['trigger'].append(sel[1]['ECoG trigger'])

    plv_dict['signal'] = np.array(plv_dict['signal'])
    plv_dict['sample']['timestamp'] = np.array(plv_dict['sample']['timestamp'])
    plv_dict['sample']['filename'] = np.array(plv_dict['sample']['filename'])
    plv_dict['sample']['trigger'] = np.array(plv_dict['sample']['trigger'])
    plv_dict['wavelet'] = df['wavelet']

    for trix, triy in zip(triu_ix, triu_iy):
        plv_dict['channel']['label'].append('{}.{}'.format(
            df['channel']['label'][trix], df['channel']['label'][triy]))
    plv_dict['channel']['label'] = np.array(plv_dict['channel']['label'])

    np.savez(out_path, **plv_dict)


def resample_PLV(np_code, remove_blank=False, trigger='Scheduled', despiked=False):
    X_PLV = load_tensor(
                np_code,
                func_name=plv_pretrigger_name if ~despiked else plvdespiked_pretrigger_name,
                trigger=trigger)
    XGLOBL = np.nanmedian(X_PLV['signal'], axis=-1)
    XINTRA = np.nanmedian(X_PLV['signal'][:, :, [0,-1]], axis=-1)
    XINTER = np.nanmedian(X_PLV['signal'][:, :, 1:-1], axis=-1)

    BANDS = {'Theta (4-8Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 4) &
                                             (X_PLV['wavelet']['freqs'] < 8)),
             'Alpha (8-15Hz)':  np.flatnonzero((X_PLV['wavelet']['freqs'] >= 8) &
                                               (X_PLV['wavelet']['freqs'] < 15)),
             'Beta (15-30Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 15) &
                                              (X_PLV['wavelet']['freqs'] < 30)),
             'Gamma (30-100Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 30) &
                                              (X_PLV['wavelet']['freqs'] < 100)),
             'LowG (30-70Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 30) &
                                              (X_PLV['wavelet']['freqs'] < 70)),
             'HighG (70-100Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 70) &
                                                (X_PLV['wavelet']['freqs'] < 100)),
             'Broadband (4-100Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 4) &
                                                   (X_PLV['wavelet']['freqs'] < 100))
    }

    W = []
    for band in BANDS:
        W.append(pd.DataFrame.from_dict(
            {'PLV_Global': np.nanmax(XGLOBL[:, BANDS[band]], axis=1),
             'PLV_Intra': np.nanmax(XINTRA[:, BANDS[band]], axis=1),
             'PLV_Inter': np.nanmax(XINTER[:, BANDS[band]], axis=1),
             'PLV_Band': np.array([band]*len(XGLOBL)),
             'Timestamp': X_PLV['sample']['timestamp'],
             'Filename': X_PLV['sample']['filename']}).set_index('Timestamp')),
    W = pd.concat(W, axis=0)

    df_blnk = e000_stim.load_stim_detect(np_code)

    n_exclude = []
    if remove_blank:
        val_index = W['Filename'].isin(df_blnk['Filename'])
        W = W[~val_index]
        n_exclude = [(~val_index).sum(), len(val_index)]

    return W, n_exclude


def resample_PLV_anatomical(np_code, remove_blank=False, trigger='Scheduled', despiked=False):
    X_PLV = load_tensor(
                np_code,
                func_name=plv_pretrigger_name if ~despiked else plvdespiked_pretrigger_name,
                trigger=trigger)

    if np_code in path['CORE']['RNS']['NP_LOC']:
        isHIPP = (
            ('hip' in X_PLV['channel']['label'][path['CORE']['RNS']['NP_LOC'][np_code]['SOZ-SOZ'][0]].split('.')[0]) &
            ('hip' in X_PLV['channel']['label'][path['CORE']['RNS']['NP_LOC'][np_code]['SOZ-SOZ'][0]].split('.')[0]))
        if isHIPP:
            XHIPP = X_PLV['signal'][:, :, path['CORE']['RNS']['NP_LOC'][np_code]['SOZ-SOZ'][0]]
            XNEO = np.nan*np.zeros_like(X_PLV['signal'][:, :, 0])
        else:
            XNEO = X_PLV['signal'][:, :, path['CORE']['RNS']['NP_LOC'][np_code]['SOZ-SOZ'][0]]
            XHIPP = np.nan*np.zeros_like(X_PLV['signal'][:, :, 0])
        X_SOZ_SOZ = np.nanmedian(X_PLV['signal'][:, :, path['CORE']['RNS']['NP_LOC'][np_code]['SOZ-SOZ']], axis=-1)
        X_nSOZ_nSOZ = np.nanmedian(X_PLV['signal'][:, :, path['CORE']['RNS']['NP_LOC'][np_code]['nSOZ-nSOZ']], axis=-1)
        X_SOZ_nSOZ = np.nanmedian(X_PLV['signal'][:, :, path['CORE']['RNS']['NP_LOC'][np_code]['SOZ-nSOZ']], axis=-1)
    else:
        isHIPP = np.array([
            ('hip' in ch.split('.')[0].lower()) &
            ('hip' in ch.split('.')[1].lower())
            for ch in X_PLV['channel']['label']])[[0,-1]]

        if True in isHIPP:
            XHIPP = np.nanmedian(X_PLV['signal'][:, :, [0, -1]][:, :, isHIPP], axis=-1)
        else:
            XHIPP = np.nan*np.zeros_like(X_PLV['signal'][:, :, 0])

        if True in ~isHIPP:
            XNEO = np.nanmedian(X_PLV['signal'][:, :, [0, -1]][:, :, ~isHIPP], axis=-1)
        else:
            XNEO = np.nan*np.zeros_like(X_PLV['signal'][:, :, 0])

        X_SOZ_SOZ = np.nan*np.zeros_like(X_PLV['signal'][:, :, 0])
        X_nSOZ_nSOZ = np.nan*np.zeros_like(X_PLV['signal'][:, :, 0])
        X_SOZ_nSOZ = np.nan*np.zeros_like(X_PLV['signal'][:, :, 0])

    BANDS = {'Theta (4-8Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 4) &
                                             (X_PLV['wavelet']['freqs'] < 8)),
             'Alpha (8-15Hz)':  np.flatnonzero((X_PLV['wavelet']['freqs'] >= 8) &
                                               (X_PLV['wavelet']['freqs'] < 15)),
             'Beta (15-30Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 15) &
                                              (X_PLV['wavelet']['freqs'] < 30)),
             'Gamma (30-100Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 30) &
                                              (X_PLV['wavelet']['freqs'] < 100)),
             'LowG (30-70Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 30) &
                                              (X_PLV['wavelet']['freqs'] < 70)),
             'HighG (70-100Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 70) &
                                                (X_PLV['wavelet']['freqs'] < 100)),
             'Broadband (4-100Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 4) &
                                                   (X_PLV['wavelet']['freqs'] < 100))
    }

    W = []
    for band in BANDS:
        W.append(pd.DataFrame.from_dict(
            {'PLV_HIPP': np.nanmax(XHIPP[:, BANDS[band]], axis=1),
             'PLV_NEO': np.nanmax(XNEO[:, BANDS[band]], axis=1),
             'PLV_SOZ_SOZ': np.nanmax(X_SOZ_SOZ[:, BANDS[band]], axis=1),
             'PLV_nSOZ_nSOZ': np.nanmax(X_nSOZ_nSOZ[:, BANDS[band]], axis=1),
             'PLV_SOZ_nSOZ': np.nanmax(X_SOZ_nSOZ[:, BANDS[band]], axis=1),
             'PLV_Band': np.array([band]*len(XHIPP)),
             'Timestamp': X_PLV['sample']['timestamp'],
             'Filename': X_PLV['sample']['filename']}).set_index('Timestamp')),
    W = pd.concat(W, axis=0)

    df_blnk = e000_stim.load_stim_detect(np_code)

    n_exclude = []
    if remove_blank:
        val_index = W['Filename'].isin(df_blnk['Filename'])
        W = W[~val_index]
        n_exclude = [(~val_index).sum(), len(val_index)]

    return W, n_exclude


def resample_PLV_LOC(np_code, remove_blank=False, trigger='Scheduled', despiked=False, skip_noloc=False):
    X_PLV = load_tensor(
                np_code,
                func_name=plv_pretrigger_name if ~despiked else plvdespiked_pretrigger_name,
                trigger=trigger)
    if np_code in path['CORE']['RNS']['NP_LOC']:
        SOZ_SOZ = np.nanmedian(X_PLV['signal'][:, :, path['CORE']['RNS']['NP_LOC'][np_code]['SOZ-SOZ']], axis=-1)
        nSOZ_nSOZ = np.nanmedian(X_PLV['signal'][:, :, path['CORE']['RNS']['NP_LOC'][np_code]['nSOZ-nSOZ']], axis=-1)
        SOZ_nSOZ = np.nanmedian(X_PLV['signal'][:, :, path['CORE']['RNS']['NP_LOC'][np_code]['SOZ-nSOZ']], axis=-1)
    else:
        if skip_noloc:
            W = pd.DataFrame.from_dict(
                    {'SOZ-SOZ': [],
                     'nSOZ-nSOZ': [],
                     'SOZ-nSOZ': [],
                     'PLV_Band': [], 
                     'Timestamp': [],
                     'Filename': []})
            n_exclude = []
            return W, n_exclude
        else:
            if np_code in path['CORE']['RNS']['NP_Ref'][path['CORE']['RNS']['NP_Ref']['N_Vs_MT'] == 'M']['NP_code'].to_list():
                SOZ_SOZ = np.nanmedian(X_PLV['signal'][:, :, [0,-1]], axis=-1)
            else:
                SOZ_SOZ = np.nanmedian(X_PLV['signal'][:, :, :], axis=-1)
            nSOZ_nSOZ = np.nan*np.zeros_like(SOZ_SOZ)
            SOZ_nSOZ = np.nan*np.zeros_like(SOZ_SOZ)

    BANDS = {'Theta (4-8Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 4) &
                                             (X_PLV['wavelet']['freqs'] < 8)),
             'Alpha (8-15Hz)':  np.flatnonzero((X_PLV['wavelet']['freqs'] >= 8) &
                                               (X_PLV['wavelet']['freqs'] < 15)),
             'Beta (15-30Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 15) &
                                              (X_PLV['wavelet']['freqs'] < 30)),
             'Gamma (30-100Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 30) &
                                              (X_PLV['wavelet']['freqs'] < 100)),
             'LowG (30-70Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 30) &
                                              (X_PLV['wavelet']['freqs'] < 70)),
             'HighG (70-100Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 70) &
                                                (X_PLV['wavelet']['freqs'] < 100)),
             'Broadband (4-100Hz)': np.flatnonzero((X_PLV['wavelet']['freqs'] >= 4) &
                                                   (X_PLV['wavelet']['freqs'] < 100))
    }

    W = []
    for band in BANDS:
        W.append(pd.DataFrame.from_dict(
            {'SOZ-SOZ': np.nanmax(SOZ_SOZ[:, BANDS[band]], axis=1),
             'nSOZ-nSOZ': np.nanmax(nSOZ_nSOZ[:, BANDS[band]], axis=1),
             'SOZ-nSOZ': np.nanmax(SOZ_nSOZ[:, BANDS[band]], axis=1),
             'PLV_Band': np.array([band]*len(X_PLV['signal'])),
             'Timestamp': X_PLV['sample']['timestamp'],
             'Filename': X_PLV['sample']['filename']}).set_index('Timestamp')),
    W = pd.concat(W, axis=0)

    df_blnk = e000_stim.load_stim_detect(np_code)

    n_exclude = []
    if remove_blank:
        val_index = W['Filename'].isin(df_blnk['Filename'])
        W = W[~val_index]
        n_exclude = [(~val_index).sum(), len(val_index)]

    return W, n_exclude


if __name__ == '__main__':
    from multiprocessing import Pool

    """
    try:
        task_id = int(os.environ['SGE_TASK_ID']) - 1
    except:
        task_id = int(sys.argv[1])
    """
    pool = Pool(10)
    pool.map(pre_trigger_PLV_despiked, list(df_catalog['NP_code'].unique()))

    """
    #wavelet_extraction(int(task_id))
    #pre_trigger_PLV(df_catalog['NP_code'].unique()[task_id])

    for task_id in range(len(df_catalog)):
        print(task_id)
        resave_wavelet_extraction_as_mat(task_id, trigger_name='Scheduled')
        resave_wavelet_extraction_as_mat(task_id, trigger_name='Long_Episode')
    """
