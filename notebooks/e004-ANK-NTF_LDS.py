"""
Functions for applying tensor-analysis to network features from raw recordings.

Created by: Ankit N. Khambhati
Updated: 2019/07/05
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings('ignore')

exec(open('__init__.py').read())
rs_path = path['RSRCH'](4)
rs1_path = path['RSRCH'](1)
rs2_path = path['RSRCH'](2)
df_meta = path['CORE']['RNS']['METADATA']
df_catalog = path['CORE']['RNS']['CATALOG']
df_catalog['Timestamp'] = pd.to_datetime(df_catalog['Timestamp'], errors='coerce')
df_catalog = df_catalog.dropna()
df_blnk = pd.read_pickle('{}/Summary.Stim_Detect.pkl'.format(rs1_path))
df_npref = path['CORE']['RNS']['NP_Ref']
df_nppc = path['CORE']['RNS']['NP_PC']

import tensortools as tt
import pickle as pkl
import scipy.stats as sp_stats

import importlib
e001_LL = importlib.import_module('e001-ANK-Line_Length')
e000_stim = importlib.import_module('e000-ANK-Stim_Blank')
e002_WV = importlib.import_module('e002-ANK-Wavelet')
PRE_TRIGGER_DUR = 30

NTF_PARAM = {'RANK': np.array([2,3]), #np.arange(1, 4, 1),
             'NTF_dict': {'beta': 1,
                          'init': 'rand'},
             'FIT_dict': {'method': 'β-Divergence',
                          'tol': 1e-4,
                          'min_iter': 1,
                          'max_iter': 4000,
                          'verbose': False},
             'N_FOLD': 10,
             'N_FULL': 50}


def ntf_name(subj_id, subj_fname):
    return '{}/NTF.{}.{}'.format(rs_path, subj_id, subj_fname)


def plv_pretrigger_name(subj_id):
    #return '{}/PLVDespiked.Pre_Trigger.{}'.format(rs2_path, subj_id)
    return '{}/PLV.Pre_Trigger.{}'.format(rs2_path, subj_id)


def gen_folds(n_obs):
    fold_size = int(np.floor(n_obs / NTF_PARAM['N_FOLD']))
    rand_obs = np.random.permutation(n_obs)
    rand_obs = rand_obs[:(fold_size*NTF_PARAM['N_FOLD'])].reshape(-1, NTF_PARAM['N_FOLD'])
    return rand_obs


def run_xval(np_id):
    out_name = ntf_name(np_id, 'XVAL')

    out_path = '{}.pkl'.format(out_name)
    log_path = '{}.log'.format(out_name) 
    if os.path.exists(out_path):
        return None

    tensor = load_tensor(np_id)
    n_obs = len(tensor['signal'])
    folds = gen_folds(n_obs)

    XVAL_DICT = {'RANK': [],
                 'FOLD': [],
                 'COST_TRAIN': [],
                 'COST_TEST': [],
                 'W': []}

    for rank in NTF_PARAM['RANK']:
        rank = int(rank)
        for fld in range(NTF_PARAM['N_FOLD']):
            stdout_orig = sys.stdout
            sys.stdout = open(log_path, 'a+')

            print('RANK: {} -- FOLD: {}'.format(rank, fld), flush=True)

            cost_train = np.nan
            cost_test = np.nan
            W = None

            try:
                train_fold = np.setdiff1d(np.arange(NTF_PARAM['N_FOLD']), [fld])
                df_train = tensor['signal'][folds[:, train_fold].reshape(-1)]
                df_test = tensor['signal'][folds[:, fld].reshape(-1)]

                ### Train fold
                model_train = tt.ncp_nnlds.init_model(
                    df_train,
                    rank=rank,
                    NTF_dict=NTF_PARAM['NTF_dict'],
                    REG_dict=None,
                    LDS_dict=None,
                    exog_input=None,
                    random_state=None)

                model_train = tt.ncp_nnlds.model_update(
                    df_train,
                    model_train,
                    exog_input=None,
                    mask=np.ones_like(df_train, dtype=bool),
                    fit_dict=NTF_PARAM['FIT_dict'])

                cost_train = model_train.status['obj']

                ### Test fold
                model_test = tt.ncp_nnlds.init_model(
                    df_test,
                    rank=rank,
                    NTF_dict=NTF_PARAM['NTF_dict'],
                    REG_dict=None,
                    LDS_dict=None,
                    exog_input=None,
                    random_state=None)

                model_test.model_param['NTF']['W'].factors[1] = model_train.model_param['NTF']['W'].factors[1]
                model_test.model_param['NTF']['W'].factors[2] = model_train.model_param['NTF']['W'].factors[2]

                model_test = tt.ncp_nnlds.model_update(
                    df_test,
                    model_test,
                    exog_input=None,
                    fixed_axes=[1,2],
                    mask=np.ones_like(df_test, dtype=bool),
                    fit_dict=NTF_PARAM['FIT_dict'])

                cost_test = model_test.status['obj']

                W = model_test.model_param['NTF']['W']

            except Exception as E:
                print('Exception: {}'.format(E))

            sys.stdout = stdout_orig

            XVAL_DICT['RANK'].append(rank)
            XVAL_DICT['FOLD'].append(fld)
            XVAL_DICT['COST_TRAIN'].append(cost_train)
            XVAL_DICT['COST_TEST'].append(cost_test)
            XVAL_DICT['W'].append(W)

    XVAL_DICT = pd.DataFrame.from_dict(XVAL_DICT)
    XVAL_DICT.to_pickle(out_path)


def get_optimal_rank(df_xval):
    from kneed import KneeLocator

    df_mean = df_xval.groupby('RANK').mean().reset_index()
    kneedle = KneeLocator(x=df_mean['RANK'], y=df_mean['COST_TEST'],
                          curve='convex', direction='decreasing')
    return kneedle.knee


def run_full(np_id):
    out_name = ntf_name(np_id, 'FULLDespiked')

    out_path = '{}.pkl'.format(out_name)
    log_path = '{}.log'.format(out_name) 
    #if os.path.exists(out_path):
    #    return None

    tensor = load_tensor(np_id)
    n_obs = len(tensor['signal'])

    FULL_DICT = {'RANK': [],
                 'RSS': [],
                 'AIC': [],
                 'BIC': [],
                 'R': [],
                 'W': []}

    for rank in NTF_PARAM['RANK']:
        for rep in range(NTF_PARAM['N_FULL']):
            stdout_orig = sys.stdout
            sys.stdout = open(log_path, 'a+')

            print('RANK: {} || REP: {}'.format(rank, rep), flush=True)

            # Preallocate dictionary items
            RSS = np.nan
            aic = np.nan
            bic = np.nan
            R = np.nan
            W = None
            rank = int(rank)

            # Try to run NTF
            try:
                ### Train fold
                model_train = tt.ncp_nnlds.init_model(
                    tensor['signal'],
                    rank=rank,
                    NTF_dict=NTF_PARAM['NTF_dict'],
                    REG_dict=None,
                    LDS_dict=None,
                    exog_input=None,
                    random_state=None)

                model_train = tt.ncp_nnlds.model_update(
                    tensor['signal'],
                    model_train,
                    exog_input=None,
                    mask=np.ones_like(tensor['signal'], dtype=bool),
                    fit_dict=NTF_PARAM['FIT_dict'])

                # Compute cost metrics and model evaluation measures
                W = model_train.model_param['NTF']['W']
                WF = W.full()
                K = rank*np.sum(WF.shape)

                RSS = np.nansum((WF - tensor['signal'])**2)
                aic = WF.size*np.log(RSS/WF.size) + 2*K + ((2*K*(K+1))/(WF.size-K-1))
                bic = WF.size*np.log(RSS/WF.size) + K*np.log(WF.size)
                R = sp_stats.pearsonr(WF.reshape(-1), tensor['signal'].reshape(-1))[0]

                # Normalize model
                l1 = np.ones(rank)
                for mm in range(3):
                    ll1 = np.linalg.norm(W.factors[mm], axis=0, ord=1)
                    W.factors[mm] /= ll1
                    l1 *= ll1
                W.factors[0] *= l1

            except Exception as E:
                print('Exception: {}'.format(E))

            sys.stdout = stdout_orig

            FULL_DICT['RANK'].append(rank)
            FULL_DICT['RSS'].append(RSS)
            FULL_DICT['AIC'].append(aic)
            FULL_DICT['BIC'].append(bic)
            FULL_DICT['R'].append(R)
            FULL_DICT['W'].append(W)

    FULL_DICT = pd.DataFrame.from_dict(FULL_DICT)
    FULL_DICT.to_pickle(out_path)


def get_optimal_full(np_id, metric='BIC', rank=None, despiked=False):
    if despiked:
        full_path = ntf_name(np_id, 'FULLDespiked') + '.pkl'
    else:
        full_path = ntf_name(np_id, 'FULL') + '.pkl'
    if not os.path.exists(full_path):
        return None
    df_full = pd.read_pickle(full_path)

    if rank is not None:
        df_full = df_full[df_full['RANK'] == rank].reset_index()

    return df_full.iloc[df_full[metric].idxmin()]['W']


def resample_NTF(np_code, remove_blank=False, metric='COST_TRAIN', rank=3, despiked=False):

    X_PLV = load_tensor(
            np_code,
            func_name=e002_WV.plv_pretrigger_name if ~despiked else e002_WV.plvdespiked_pretrigger_name,
            trigger='Scheduled')
    X_NTF = get_optimal_full(
                np_code, metric=metric, rank=rank, despiked=despiked)

    # W
    fac_dict = {}
    for rr in range(X_NTF.rank):
        fac_dict['Fac_{}'.format(rr)] = X_NTF.factors[0][:, rr]
    fac_dict['Timestamp'] = X_PLV['sample']['timestamp']
    fac_dict['Filename'] = X_PLV['sample']['filename']
    W_time = pd.DataFrame.from_dict(fac_dict).set_index('Timestamp')

    fac_dict = {}
    for rr in range(X_NTF.rank):
        fac_dict['Fac_{}'.format(rr)] = X_NTF.factors[1][:, rr]
    fac_dict['Frequency'] = X_PLV['wavelet']['freqs']
    W_freq = pd.DataFrame.from_dict(fac_dict).set_index('Frequency')

    fac_dict = {}
    for rr in range(X_NTF.rank):
        fac_dict['Fac_{}'.format(rr)] = X_NTF.factors[2][:, rr]
    fac_dict['Connections'] = ['\n'.join(lbl.split('.'))
            for lbl in X_PLV['channel']['label']]
    W_conn = pd.DataFrame.from_dict(fac_dict).set_index('Connections')


    df_blnk = e000_stim.load_stim_detect(np_code)

    n_exclude = []
    if remove_blank:
        val_index = W_time['Filename'].isin(df_blnk['Filename'])
        W_time = W_time[~val_index]
        n_exclude = [(~val_index).sum(), len(val_index)]

    W_freq = W_freq / np.linalg.norm(W_freq, axis=0, ord=1)
    W_conn = W_conn / np.linalg.norm(W_conn, axis=0, ord=1)

    return W_time, W_freq, W_conn, n_exclude


if __name__ == '__main__':
    from multiprocessing import Pool
    pp = Pool(30)

    try:
        task_id = int(os.environ['SGE_TASK_ID']) - 1
    except:
        pass
    #np_code = np.array(df_catalog['NP_code'].unique())[task_id]
    #run_xval(np_code)
    #run_full(np_code)

    #output = pp.map(run_xval, np.array(df_catalog['NP_code'].unique()))
    output = pp.map(run_full, np.array(df_catalog['NP_code'].unique()))
