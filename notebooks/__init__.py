"""
Generalized imports and initializations across notebooks

Ankit N. Khambhati
Last Updated: 2021/04/21
"""

### IO Imports
import glob
import os
import sys
import datetime 
from datetime import datetime
import h5py
### Plotting Setup
import matplotlib.pyplot as plt
### Data Imports
import numpy as np
import pandas as pd
import scipy.io as sp_io
import scipy.signal as sp_sig
import scipy.stats as sp_stats
import statsmodels.api as sm
import seaborn as sns
from datetime import timedelta 

import pyeisen
import pyfftw
### Project imports
import utils

sns.set_style(
    'ticks', {
        'axes.grid': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'xtick.bottom': True,
        'ytick.left': True,
        'xtick.color': 'k',
        'ytick.color': 'k'
    })


### Setup Paths
def _rsrch_hook(e_id):
    assert type(e_id) == int
    assert e_id < 100
    e_as_str = 'e{:03d}'.format(e_id)

    base = HOME + '/Remotes/RSRCH.2018.SpikeControl/{}'.format(e_as_str)
    os.makedirs(base, exist_ok=True)
    return base


path = {}
HOME = os.environ['HOTH']
path['CORE'] = {'RNS': {}}
path['RSRCH'] = _rsrch_hook

### Set paths
# NeuroPace RNS
CORE_RNS = HOME + '/Remotes/CORE.NeuroPace.vrao'
path['CORE']['RNS'][
    'BASE'] = '{}/NeuroPace UCSF PDMS Data Transfer #PHI'.format(CORE_RNS)
path['CORE']['RNS']['CATALOG'] = pd.read_csv(
    '{}/UCSF_ECoG_Catalog_v041619.csv'.format(CORE_RNS))
path['CORE']['RNS']['NP_Ref'] = pd.read_csv(
    #'{}/NP_Patient_Codes_v053119.csv'.format(CORE_RNS))
    '{}/NP_Patient_Codes_v030220.csv'.format(CORE_RNS))
path['CORE']['RNS']['NP_PC'] = pd.read_csv(
    '{}/NP_Param_Log_v053119.csv'.format(CORE_RNS))
path['CORE']['RNS']['NP_Outcome'] = pd.read_csv(
    '{}/NP_Outcomes_v053119.csv'.format(CORE_RNS))
path['CORE']['RNS']['METADATA'] = pd.read_excel(
    '{}/UCSF_RNS_MASTER_patient_summary_VR_v053119_abridged.xlsx'.format(CORE_RNS))
path['CORE']['RNS']['NP_LOC'] = pd.read_json(
    '{}/NP_SOZ_Leads_v123020.json'.format(CORE_RNS))

path['CORE']['RNS']['HOURLY'] = {}
path['CORE']['RNS']['HOURLY']['BASE'] = \
        '{}/NeuroPace UCSF HOURLY Data Transfer #PHI'.format(CORE_RNS)
for csv_pth in glob.glob('{}/*.csv'.format(path['CORE']['RNS']['HOURLY']['BASE'])):
    np_num_id = csv_pth.split('/')[-1].split('_')[0]
    sel_ix = path['CORE']['RNS']['NP_Ref']['NP_Patient_ID'] == float(np_num_id)
    np_id = path['CORE']['RNS']['NP_Ref'][sel_ix]['NP_code']
    if len(np_id) == 1:
        np_id = np_id.iloc[0]
        path['CORE']['RNS']['HOURLY'][np_id] = csv_pth

# Automagically Merge the NP codes
path['CORE']['RNS']['CATALOG'] = pd.merge(
        path['CORE']['RNS']['CATALOG'],
        path['CORE']['RNS']['NP_Ref'][['NP_code', 'NP_Patient_ID']],
        left_on='Patient ID', right_on='NP_Patient_ID').drop_duplicates()

path['CORE']['RNS']['METADATA'] = pd.merge(
        path['CORE']['RNS']['METADATA'],
        path['CORE']['RNS']['NP_Ref'][['NP_code', 'Initials']],
        left_on='Initials', right_on='Initials').drop_duplicates()

# Convert to Datetimes
path['CORE']['RNS']['NP_Ref']['Date_First_Implant'] = pd.to_datetime(path['CORE']['RNS']['NP_Ref']['Date_First_Implant'])
path['CORE']['RNS']['NP_Ref']['Date_Stim_On'] = pd.to_datetime(path['CORE']['RNS']['NP_Ref']['Date_Stim_On'])
path['CORE']['RNS']['NP_Outcome']['Date_Visit'] = pd.to_datetime(path['CORE']['RNS']['NP_Outcome']['Date_Visit'])
path['CORE']['RNS']['CATALOG']['Timestamp'] = pd.to_datetime(path['CORE']['RNS']['CATALOG']['Timestamp'], errors='coerce')

# Compute the recording span
df_sched = path['CORE']['RNS']['CATALOG'][path['CORE']['RNS']['CATALOG']['ECoG trigger'] == 'Scheduled']
df_recspan = path['CORE']['RNS']['CATALOG'].groupby(['NP_code']).apply(
        lambda x: len(x.set_index('Timestamp').resample('1d').mean())).reset_index()
df_recspan = df_recspan.rename(columns={0: 'Days_Recording'})
df_recspan_gap = path['CORE']['RNS']['CATALOG'].groupby(['NP_code']).apply(
        lambda x: x.set_index('Timestamp').resample('1d').mean().isna().mean()).reset_index()
df_recspan_gap = df_recspan_gap[['NP_code', 'Patient ID']].rename(columns={'Patient ID': 'Data Gaps'})

# Get the last outcome measurement
df_lastvisit = path['CORE']['RNS']['NP_Outcome'].loc[path['CORE']['RNS']['NP_Outcome'].groupby('NP_code')['Date_Visit'].idxmax()]

# Get timestamp of last recorded clip
latest_clip_date = path['CORE']['RNS']['CATALOG'].groupby(['NP_code']).apply(
    lambda x: x['Timestamp'].max()).reset_index()
latest_clip_date = latest_clip_date.rename(columns={0: 'Latest_Clip'})
latest_clip_date['Latest_Clip'] = pd.to_datetime(latest_clip_date['Latest_Clip'])

# Merge latest dates with longitudinal outcome data
path['CORE']['RNS']['NP_Ref'] = pd.merge(path['CORE']['RNS']['NP_Ref'], df_lastvisit, on='NP_code')
path['CORE']['RNS']['NP_Ref'] = pd.merge(path['CORE']['RNS']['NP_Ref'], latest_clip_date, on='NP_code')
path['CORE']['RNS']['NP_Ref'] = pd.merge(path['CORE']['RNS']['NP_Ref'], path['CORE']['RNS']['METADATA'],
        left_on='Initials_x', right_on='Initials')
path['CORE']['RNS']['NP_Ref']['NP_code'] = path['CORE']['RNS']['NP_Ref']['NP_code_x']
path['CORE']['RNS']['NP_Ref']['Initials'] = path['CORE']['RNS']['NP_Ref']['Initials_x']

# Calculate the Days Follow Up
path['CORE']['RNS']['NP_Ref']['Days_FollowUp'] = (path['CORE']['RNS']['NP_Ref']['Date_Visit'] -
        path['CORE']['RNS']['NP_Ref']['Date_First_Implant']).dt.round('1d')
path['CORE']['RNS']['NP_Ref'] = pd.merge(path['CORE']['RNS']['NP_Ref'],
        df_recspan, on='NP_code')
path['CORE']['RNS']['NP_Ref'] = pd.merge(path['CORE']['RNS']['NP_Ref'],
        df_recspan_gap, on='NP_code')

# Assign responder types based on VRao cutoffs
path['CORE']['RNS']['NP_Ref']['Pct_Seizure_Reduction_Last_Visit'] = \
        100*(path['CORE']['RNS']['NP_Ref']['Sz_per_Week'] / path['CORE']['RNS']['NP_Ref']['Baseline_Sz_per_Week'] - 1)
path['CORE']['RNS']['NP_Ref'].loc[path['CORE']['RNS']['NP_Ref']['Pct_Seizure_Reduction_Last_Visit'] > -50, 'Responder_Type'] = 'NR'
path['CORE']['RNS']['NP_Ref'].loc[path['CORE']['RNS']['NP_Ref']['Pct_Seizure_Reduction_Last_Visit'] <= -50, 'Responder_Type'] = 'IR'
path['CORE']['RNS']['NP_Ref'].loc[path['CORE']['RNS']['NP_Ref']['Pct_Seizure_Reduction_Last_Visit'] <= -90, 'Responder_Type'] = 'SR'

# Exclusions
path['CORE']['RNS']['NP_Ref'] = path['CORE']['RNS']['NP_Ref'][
    (path['CORE']['RNS']['NP_Ref']['Pct_Seizure_Reduction_Last_Visit'] >= -100) & 
    (path['CORE']['RNS']['NP_Ref']['Pct_Seizure_Reduction_Last_Visit'] <= 100)]
path['CORE']['RNS']['NP_Ref'] = path['CORE']['RNS']['NP_Ref'][ 
        path['CORE']['RNS']['NP_Ref']['Days_FollowUp'] >= '90d']

# Clean-Up
path['CORE']['RNS']['NP_Ref'] = path['CORE']['RNS']['NP_Ref'].drop(
        columns=['NP_code_x', 'NP_code_y', 'Initials_x', 'Initials_y',
            'MRN', 'NP_Patient_ID', '% Change Seizure Frequency* (LOCF)'])


def extract_key_dates(np_code):
    df_npref = path['CORE']['RNS']['NP_Ref']
    date_implant = pd.to_datetime(df_npref[df_npref['NP_code'] == np_code]['Date_First_Implant']).iloc[0]    
    date_stimon = pd.to_datetime(df_npref[df_npref['NP_code'] == np_code]['Date_Stim_On']).iloc[0]
    date_ieffect = date_implant + pd.Timedelta('90D')

    return date_implant, date_stimon, date_ieffect


def load_tensor(np_id, func_name=None, trigger='Scheduled'):
    df = utils.signal_dict.load_data_dict(
        '{}.npz'.format(func_name(np_id)))
    df = utils.signal_dict.subset(
            df, sample=np.flatnonzero(df['sample']['trigger'] == trigger))

    bad_ix = np.unique(np.nonzero(np.isnan(df['signal']))[0])
    good_ix = np.setdiff1d(np.arange(df['signal'].shape[0]), bad_ix)
    df = utils.signal_dict.subset(df, sample=good_ix)

    sel_ctlg = path['CORE']['RNS']['CATALOG'][path['CORE']['RNS']['CATALOG']['NP_code'] == np_id]
    sel_ctlg = sel_ctlg[sel_ctlg['Filename'].isin(df['sample']['filename'])]
    sel_ctlg = sel_ctlg.reset_index()

    df['sample']['timestamp'] = pd.to_datetime(sel_ctlg['Raw UTC Timestamp'])
    df['sample']['timestamp'] = df['sample']['timestamp'].reset_index(drop=True)

    return df

