"""
Routines for finding and blanking stim periods.

Created by: Ankit N. Khambhati
Updated: 2019/07/05
"""

import warnings
warnings.filterwarnings('ignore')

exec(open('__init__.py').read())
rs_path = path['RSRCH'](0)
df_meta = path['CORE']['RNS']['METADATA']
df_catalog = path['CORE']['RNS']['CATALOG']
df_catalog['Timestamp'] = pd.to_datetime(df_catalog['Timestamp'], errors='coerce')
df_catalog = df_catalog.dropna()


def stimblank_name(subj_id, subj_fname):
    return '{}/StimBlanks.{}.{}'.format(rs_path, subj_id, subj_fname)


def blank_detect():
    blank_catalog = {'NP_code': [],
                     'Filename': [],
                     'StimOnset_ix': [],
                     'StimOffset_ix': []}

    for ix in tqdm(range(len(df_catalog))):
        sel = df_catalog.iloc[[ix]]

        detect_dict = None
        try:
            dd = utils.neuropaceIO.get_ieeg_data_dict(sel, path['CORE']['RNS']['BASE'])
            Fs = utils.signal_dict.get_fs(dd)
            detect_dict = utils.stim_blank.nan_hold_sample(dd,
                    hold_sample=10, pad_sample=int(Fs*0.1))

            for i1, i2 in zip(detect_dict['onset_ix'], detect_dict['offset_ix']):
                blank_catalog['NP_code'].append(sel['NP_code'].iloc[0])
                blank_catalog['Filename'].append(sel['Filename'].iloc[0])
                blank_catalog['StimOnset_ix'].append(i1)
                blank_catalog['StimOffset_ix'].append(i2)

        except:
            continue

    blank_catalog = pd.DataFrame.from_dict(blank_catalog)
    blank_catalog.to_pickle('{}/Summary.Stim_Detect.pkl'.format(rs_path))


def load_stim_detect(np_code):
    df_blnk = pd.read_pickle('{}/Summary.Stim_Detect.pkl'.format(rs_path))
    return df_blnk[df_blnk['NP_code'] == np_code].reset_index(drop=True)


if __name__ == '__main__':
    print(' --- Detecting Signal Blanking Due to Stim ---')
    blank_detect()
