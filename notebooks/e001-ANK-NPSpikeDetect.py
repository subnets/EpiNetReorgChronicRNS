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
rs_path = path['RSRCH'](5)
df_meta = path['CORE']['RNS']['METADATA']
df_catalog = path['CORE']['RNS']['CATALOG']
df_catalog['Timestamp'] = pd.to_datetime(
    df_catalog['Timestamp'], errors='coerce')
df_catalog = df_catalog.dropna()

import importlib
e000_stim = importlib.import_module('e000-ANK-Stim_Blank')

PRE_TRIGGER_DUR = 30


def npspkdetect_name(subj_id, subj_fname):
    return '{}/NPSpkDetect.{}.{}'.format(rs_path, subj_id, subj_fname)


def npspkdetect_pretrigger_name(subj_id):
    return '{}/NPSPKDETECT_FEAT.Pre_Trigger.{}'.format(rs_path, subj_id)


def npspkdetect_extraction(catalog_index):
    sel = df_catalog.iloc[[catalog_index]]
    out_name = npspkdetect_name(sel['NP_code'].iloc[0], sel['Filename'].iloc[0])

    out_path = '{}.npz'.format(out_name)
    log_path = '{}.log'.format(out_name)
    if os.path.exists(out_path):
        return None

    sys.stdout = open(log_path, 'w')

    dd = utils.neuropaceIO.get_ieeg_data_dict(sel, path['CORE']['RNS']['BASE'])
    dd_npspkdetect = utils.apply_npspkdetect.apply(dd)

    np.savez(out_path, **dd_npspkdetect)


def pre_trigger_NPSPKDETECT_FEAT(np_code):
    sel_clip = df_catalog[df_catalog['NP_code'] == np_code]
    sel_clip = sel_clip[sel_clip['ECoG Pre-trigger length'] >= PRE_TRIGGER_DUR]

    out_path = '{}.npz'.format(npspkdetect_pretrigger_name(np_code))
    log_path = '{}.log'.format(npspkdetect_pretrigger_name(np_code))
    #if os.path.exists(out_path):
    #    return None

    sys.stdout = open(log_path, 'w')

    spk_dict = {
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
                npspkdetect_name(sel[1]['NP_code'], sel[1]['Filename'])))
        except Exception as error:
            print('{} -- {}'.format(sel_i, error))
            continue

        # Get the pre-trigger epoch
        Fs = utils.signal_dict.get_fs(df)
        n_pretrig = int(Fs * PRE_TRIGGER_DUR)
        df = utils.signal_dict.subset(df, sample=slice(0, n_pretrig))

        # Compute mean/variance Line-Length statistic during the whole pre-trigger epoch
        spk_mean = np.nanmean(df['signal'], axis=0)
        spk_feat = np.expand_dims(spk_mean, axis=0)
        spk_name = df['kernel']['threshold']

        if spk_feat.shape[-1] != 4:
            continue

        # Add all necessary info to the plv_dict
        spk_dict['signal'].append(spk_feat[0, ...])
        spk_dict['sample']['timestamp'].append(df['sample']['timestamp'][-1])
        spk_dict['sample']['filename'].append(df['sample']['filename'][-1])
        spk_dict['sample']['trigger'].append(sel[1]['ECoG trigger'])

    spk_dict['signal'] = np.array(spk_dict['signal'])
    spk_dict['sample']['timestamp'] = np.array(spk_dict['sample']['timestamp'])
    spk_dict['sample']['filename'] = np.array(spk_dict['sample']['filename'])
    spk_dict['sample']['trigger'] = np.array(spk_dict['sample']['trigger'])
    spk_dict['kernel'] = {'name': np.array(spk_name)}
    spk_dict['channel']['label'] = np.array(df['channel']['label'])

    np.savez(out_path, **spk_dict)


def resample_SPK(np_code, remove_blank=False, trigger='Scheduled'):
    X_SPK = load_tensor(np_code, func_name=npspkdetect_pretrigger_name, trigger=trigger)

    W = pd.DataFrame(np.nanmean(X_SPK['signal'][:,:,:], axis=-1),
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

    from multiprocessing import Pool
    pp = Pool(30)

    try:
        task_id = int(os.environ['SGE_TASK_ID']) - 1
    except:
        pass
    #output = pp.map(npspkdetect_extraction, range(len(df_catalog)))
    output = pp.map(pre_trigger_NPSPKDETECT_FEAT, np.array(df_catalog['NP_code'].unique()))
