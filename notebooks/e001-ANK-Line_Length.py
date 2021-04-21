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
rs_path = path['RSRCH'](1)
df_meta = path['CORE']['RNS']['METADATA']
df_catalog = path['CORE']['RNS']['CATALOG']
df_catalog['Timestamp'] = pd.to_datetime(
    df_catalog['Timestamp'], errors='coerce')
df_catalog = df_catalog.dropna()

import importlib
e000_stim = importlib.import_module('e000-ANK-Stim_Blank')

PRE_TRIGGER_DUR = 30


def linelength_name(subj_id, subj_fname):
    return '{}/LineLength.{}.{}'.format(rs_path, subj_id, subj_fname)


def spiketimes_name(subj_id, subj_fname):
    return '{}/SpikeTimes.{}.{}'.format(rs_path, subj_id, subj_fname)


def linelength_pretrigger_name(subj_id):
    return '{}/LL_FEAT.Pre_Trigger.{}'.format(rs_path, subj_id)


def linelength_extraction(catalog_index):
    sel = df_catalog.iloc[[catalog_index]]
    out_name = linelength_name(sel['NP_code'].iloc[0], sel['Filename'].iloc[0])

    out_path = '{}.npz'.format(out_name)
    log_path = '{}.log'.format(out_name)
    if os.path.exists(out_path):
        return None

    sys.stdout = open(log_path, 'w')

    dd = utils.neuropaceIO.get_ieeg_data_dict(sel, path['CORE']['RNS']['BASE'])
    Fs = utils.signal_dict.get_fs(dd)
    utils.stim_blank.nan_hold_sample(
        dd, hold_sample=10, pad_sample=int(Fs * 0.1))
    dd_linelength = utils.apply_linelength.apply(dd)

    np.savez(out_path, **dd_linelength)


def spiketimes_extraction(catalog_index):
    sel = df_catalog.iloc[[catalog_index]]
    inp_name = linelength_name(sel['NP_code'].iloc[0], sel['Filename'].iloc[0])
    out_name = spiketimes_name(sel['NP_code'].iloc[0], sel['Filename'].iloc[0])

    out_path = '{}.npz'.format(out_name)
    log_path = '{}.log'.format(out_name)
    #if os.path.exists(out_path):
    #    return None

    sys.stdout = open(log_path, 'w')

    try:
        df = utils.signal_dict.load_data_dict('{}.npz'.format(inp_name))
    except Exception as error:
        print('{} -- {}'.format(sel_i, error))
        return None
    Fs = utils.signal_dict.get_fs(df)


    # Threshold to obtain event counts
    utils.signal_dict.zscore(df, 'sample', method='robust')
    ix_on = []
    thresholds = [*range(1,21)]
    for ZZZ in thresholds:
        thr = np.abs(df['signal'])
        thr[np.abs(df['signal']) > ZZZ] = 1
        thr[np.abs(df['signal']) <= ZZZ] = 0
        
        ix_on_thr = []
        for ll in range(thr.shape[1]):
            ix_on_ll = []
            for ch in range(thr.shape[2]):
                thr_ix = np.flatnonzero(np.diff(thr[:, ll, ch], axis=0) > 0)
                thr_ix = thr_ix / Fs
                
                # Filter thr_ix to avoid double counting LL threshold crossings within a 0.5s window
                ix_ch = []
                if len(thr_ix) > 0:
                    ix_ch.append(thr_ix[0])
                
                if len(thr_ix) > 1:
                    for ix in thr_ix[1:]:
                        if (ix-ix_ch[-1]) > 0.5:
                            ix_ch.append(ix)
                ix_on_ll.append(ix_ch)
            ix_on_thr.append(ix_on_ll)
        ix_on.append(ix_on_thr)
    ix_on = np.array(ix_on, dtype=object)

    # Create a dict to hold this info
    spk_dict = {
        'signal': ix_on,
        'threshold': {'name': np.array(thresholds)},
        'kernel': df['kernel'],
        'channel': df['channel'],
        'axis_ord': np.array(['threshold', 'kernel', 'channel'])
        }

    np.savez(out_path, **spk_dict)


def pre_trigger_LL_FEAT(np_code):
    sel_clip = df_catalog[df_catalog['NP_code'] == np_code]
    sel_clip = sel_clip[sel_clip['ECoG Pre-trigger length'] >= PRE_TRIGGER_DUR]

    out_path = '{}.npz'.format(linelength_pretrigger_name(np_code))
    log_path = '{}.log'.format(linelength_pretrigger_name(np_code))
    #if os.path.exists(out_path):
    #    return None

    sys.stdout = open(log_path, 'w')

    ll_dict = {
        'signal': [],
        'sample': {
            'timestamp': [],
            'filename': [],
            'trigger': []
        },
        'kernel': {},
        'channel': {
            'label': []
        },
        'axis_ord': np.array(['sample', 'kernel', 'channel'])
    }

    for sel_i, sel in enumerate(sel_clip.iterrows()):
        print('{} of {}'.format(sel_i + 1, len(sel_clip)))

        try:
            df = utils.signal_dict.load_data_dict('{}.npz'.format(
                linelength_name(sel[1]['NP_code'], sel[1]['Filename'])))
            df_spk = utils.signal_dict.load_data_dict('{}.npz'.format(
                spiketimes_name(sel[1]['NP_code'], sel[1]['Filename'])))
        except Exception as error:
            print('{} -- {}'.format(sel_i, error))
            continue

        # Get the pre-trigger epoch
        Fs = utils.signal_dict.get_fs(df)
        n_pretrig = int(Fs * PRE_TRIGGER_DUR)
        df = utils.signal_dict.subset(df, sample=slice(0, n_pretrig))

        # Compute mean/variance Line-Length statistic during the whole pre-trigger epoch
        ll_mean = np.nanmean(df['signal'], axis=0)
        ll_var = np.nanvar(df['signal'], axis=0)
        ll_feat = np.concatenate((np.expand_dims(ll_mean, axis=0),
                                  np.expand_dims(ll_var, axis=0)), axis=1)
        ll_name = ['mean', 'variance']

        # Grab SpikeTimes counts
        for thr_ii in range(len(df_spk['threshold']['name'])):
            spk_cnt = np.array([[sum(np.array(df_spk['signal'][thr_ii, ll, ch]) < PRE_TRIGGER_DUR)
                                 for ch in range(df_spk['signal'].shape[2])]
                                for ll in range(df_spk['signal'].shape[1])])
            spk_cnt = spk_cnt / PRE_TRIGGER_DUR
            ll_feat = np.concatenate((ll_feat, np.expand_dims(spk_cnt, axis=0)), axis=1)
            ll_name.append('Z={}'.format(df_spk['threshold']['name'][thr_ii]))
            
        if ll_feat.shape[-1] != 4:
            continue

        # Add all necessary info to the plv_dict
        ll_dict['signal'].append(ll_feat[0, ...])
        ll_dict['sample']['timestamp'].append(df['sample']['timestamp'][-1])
        ll_dict['sample']['filename'].append(df['sample']['filename'][-1])
        ll_dict['sample']['trigger'].append(sel[1]['ECoG trigger'])

    ll_dict['signal'] = np.array(ll_dict['signal'])
    ll_dict['sample']['timestamp'] = np.array(ll_dict['sample']['timestamp'])
    ll_dict['sample']['filename'] = np.array(ll_dict['sample']['filename'])
    ll_dict['sample']['trigger'] = np.array(ll_dict['sample']['trigger'])
    ll_dict['kernel'] = {'name': np.array(ll_name)}
    ll_dict['channel']['label'] = np.array(df['channel']['label'])

    np.savez(out_path, **ll_dict)


def resample_LL(np_code, remove_blank=False, trigger='Scheduled'):
    X_SPK = load_tensor(np_code, func_name=linelength_pretrigger_name, trigger=trigger)
    X_SPK_FANO = X_SPK['signal'][:,1,:] /  X_SPK['signal'][:,0,:]
    X_SPK_COVR = np.sqrt(X_SPK['signal'][:,1,:]) /  X_SPK['signal'][:,0,:]
    X_SPK['signal'] = np.concatenate(
            (X_SPK['signal'], np.expand_dims(X_SPK_FANO, axis=1),
             np.expand_dims(X_SPK_COVR, axis=1)), axis=1)
    X_SPK['kernel']['name'] = np.concatenate((X_SPK['kernel']['name'], np.array(['Fano', 'CoVR'])))

    W = pd.DataFrame(np.nanmax(X_SPK['signal'][:,:,:], axis=-1),
                     columns=X_SPK['kernel']['name'],
                     index=X_SPK['sample']['timestamp'])

    df_blnk = e000_stim.load_stim_detect(np_code)

    n_exclude = []
    if remove_blank:
        W_file = pd.DataFrame(X_SPK['sample']['filename'],
                              columns=['Filename'],
                              index=X_SPK['sample']['timestamp'])
        val_index = W_file['Filename'].isin(df_blnk['Filename'])
        W = W[~val_index]
        n_exclude = [(~val_index).sum(), len(val_index)]

    return W, n_exclude


if __name__ == '__main__':

    """
    try:
        task_id = int(os.environ['SGE_TASK_ID']) - 1
    except:
        task_id = int(sys.argv[1])

    N_BUFFER = 16

    N_CAT = len(df_catalog)
    IX_PER_BUFFER = int(np.ceil(N_CAT/N_BUFFER))

    for cat_ix in np.arange(
            task_id*IX_PER_BUFFER,
            (task_id+1)*IX_PER_BUFFER):
        linelength_extraction(int(cat_ix))

    NP = df_catalog['NP_code'].unique()
    N_CAT = len(NP)
    IX_PER_BUFFER = int(np.ceil(N_CAT/N_BUFFER))

    for cat_ix in np.arange(
            task_id*IX_PER_BUFFER,
            (task_id+1)*IX_PER_BUFFER):
        pre_trigger_LL_FEAT(NP[cat_ix])
    """

    from multiprocessing import Pool
    pp = Pool(30)

    try:
        task_id = int(os.environ['SGE_TASK_ID']) - 1
    except:
        pass
    #output = pp.map(spiketimes_extraction, range(len(df_catalog)))
    output = pp.map(pre_trigger_LL_FEAT, np.array(df_catalog['NP_code'].unique()))
