"""
Library to manipulate device parameter files

Created by: Ankit N. Khambhati
Last Updated: 2019/07/09
"""

import glob
import warnings

import numpy as np
import pandas as pd


POLARITY_DICT = {'Cathodal': 
                          ['(----)(++++)(0)',
                           '(----)(0000)(+)',
                           '(++++)(----)(0)',
                           '(0---)(0000)(+)',
                           '(00--)(++++)(0)',
                           '(0000)(----)(+)',
                           '(-0-0)(0-0-)(+)',
                           '(0-0-)(-0-0)(+)',
                           '(00++)(----)(0)',
                           '(00--)(++++)(0)'],
                   'Bipolar_Wide': 
                          ['(++--)(0000)(0)',
                           '(0000)(--++)(+)',
                           '(0000)(--++)(0)',
                           '(0000)(++--)(0)',
                           '(--++)(0000)(0)',
                           '(0000)(++--)(+)',
                           '(0000)(--++)(+)'],
                   'Bipolar_Narrow':
                          ['(+-+-)(-+-+)(0)',
                           '(+-+-)(0000)(0)',
                           '(-+-+)(+-+-)(0)',
                           '(-+-+)(0000)(0)',
                           '(0000)(0+-+)(0)',
                           '(0000)(0-+-)(0)',
                           '(0000)(+-+-)(0)',
                           '(0000)(-+-+)(0)',
                           '(0000)(00+-)(0)',
                           '(0+-+)(0000)(0)',
                           '(0-+-)(0000)(0)',
                           '(0000)(00+-)(0)',
                           '(0000)(00-+)(0)'],
                   'Bipolar_Mix':
                         ['(--++)(+-+-)(0)',
                          '(--++)(----)(+)']
                  }

POLARITY_ARR = np.array([[key, val]
        for key in POLARITY_DICT
        for val in POLARITY_DICT[key]])

N_TX = 5
N_BURST = 2


def refactor_data_dict(df_nppc):
    """
    Refactor the DataFrame containing device sense/stim parameter
    configuration. This function will generate a row corresponding to the
    parameters used for each therapy and burst delivery.

    Parameters
    ----------
        df_nppc: pandas DataFrame
            Derived from the masterlog file and contains device parameters.

    Returns
    -------
        df_refactor: pandas DataFrame
    """
 
    refactor_nppc = {'Raw UTC Timestamp': [],
                     'Therapy_ID': [],
                     'Burst_ID': [],
                     'PARAM_Detector': [],
                     'PARAM_Polarity': [],
                     'PARAM_Current': [],
                     'PARAM_Freq': [],
                     'PARAM_PW': [],
                     'PARAM_Duration': [],
                     'PARAM_CD': []}

    for df_ii, sel_df in df_nppc.iterrows():
        for tx in (np.arange(N_TX) + 1):
            for bb in (np.arange(N_BURST) + 1):
                tx_str = 'Tx{}_B{}'.format(tx, bb)

                sel_pol = sel_df[tx_str]
                if sel_pol != 'OFF':
                    sel_name = np.unique(POLARITY_ARR[POLARITY_ARR[:, 1] == sel_pol, 0])
                    if len(sel_name) == 0:
                        sel_pol = 'Other'
                    if len(sel_name) == 1:
                        sel_pol =  sel_name[0]
                    if len(sel_name) > 1:
                        raise Exception('Errored.')

                refactor_nppc['Raw UTC Timestamp'].append(pd.to_datetime(sel_df['Programming_Date']))
                refactor_nppc['Therapy_ID'].append(tx_str.split('_')[0])
                refactor_nppc['Burst_ID'].append(tx_str.split('_')[1])
                refactor_nppc['PARAM_Detector'].append('{} {}'.format(
                    sel_df['Detection_A'], sel_df['Detection_B']).upper())
                refactor_nppc['PARAM_Polarity'].append(sel_pol)
                for key in sel_df.keys():
                    key_pfix = key.split('_' + tx_str)
                    if len(key_pfix) != 2:
                        continue
                    val = sel_df[key]
                    if  val == 'OFF':
                        val = 0
                    else:
                        val = float(val.split(' ')[0])
                    refactor_nppc['PARAM_' + key_pfix[0]].append(val)
    refactor_nppc = pd.DataFrame(refactor_nppc)
    refactor_nppc = refactor_nppc.sort_values(
            by=['Raw UTC Timestamp', 'Therapy_ID', 'Burst_ID']).reset_index(drop=True)
    refactor_nppc = refactor_nppc.set_index('Raw UTC Timestamp', drop=True)

    return refactor_nppc
