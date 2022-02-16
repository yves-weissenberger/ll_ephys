from typing import Tuple, Any
import re
from dataclasses import dataclass


import numpy as np
import pandas as pd

from .process_data.proc_beh import build_poke_df
from .process_data.proc_neural import get_mean_resps
from .load import load_data

@dataclass
class session_data:
    task_1_resps: None
    task_2_resps: None
    task_1_resps_by_direction: None
    task_2_resps_by_direction: None
    session_dataframe: None

def load_inpoke_filtered_data(root_path: str) -> Tuple[np.ndarray, np.ndarray]:
    out = load_data(root_path,align_to='task')
    spkT,spkC,single_units,events,lines,aligner = out
    # spkT_aligned = aligner.A_to_B(spkT)
    #valid_spikes = np.where(~np.isnan(spkT))[0]
    # spkC_task = spkC[valid_spikes]
    # spkT_task = spkT[valid_spikes]

    df = build_poke_df(lines,events)
    poke_dict_t1, poke_dict_t2 = build_poke_dict_with_inpokes_and_outpokes(df)
    all_resps_task1_single_trials,_ = get_all_resps_inpokes_and_outpokes(aligner, poke_dict_t1,
                                                   single_units, spkT,
                                                   spkC)

    all_resps_task2_single_trials,_ = get_all_resps_inpokes_and_outpokes(aligner, poke_dict_t2,
                                                   single_units, spkT,
                                                   spkC)

    mean_resps_task1,_ = get_mean_resps(all_resps_task1_single_trials)
    mean_resps_task2,_ = get_mean_resps(all_resps_task2_single_trials)

    return mean_resps_task1, mean_resps_task2


def build_poke_dict_with_inpokes_and_outpokes(df: pd.DataFrame) -> Tuple[dict, dict]:
    
    poke_dict_t1 ={}
    poke_dict_t2 = {}
    for port_nr in np.unique(df['port'].values):
        for task_nr in range(2):
            task_nr = str(task_nr)
            #print(task_nr)
            v = df.loc[(df['port']==port_nr) &
                       (df['correct']==True) & 
                       #(df['next_correct']==True) &
                       #(df['reward']==True) &
                       (df['port_repeat']==False) & 
                       (df['RT']<1600) &
                       (df['task_nr']==task_nr)
                      ][['inpoke_time','outpoke_time']].values
            #v = np.array(v).astype('float')
            if task_nr=='0':
                #print(task_nr,len(v),str(port_nr))
                poke_dict_t1[str(port_nr)] = np.array([[int(i[0]), int(i[1])] for i in v])
            else:
                poke_dict_t2[str(port_nr)] = np.array([[int(i[0]), int(i[1])] for i in v])


            if port_nr==8:
                poke_dict_t1['task_nr'] = str(task_nr)
                poke_dict_t2['task_nr'] = str(task_nr)
                poke_dict_t1['graph_type'] = df.loc[df['task_nr']==task_nr]['graph_type'].values[0]
                poke_dict_t2['graph_type'] = df.loc[df['task_nr']==task_nr]['graph_type'].values[0]
                poke_dict_t1['seq'] = df.loc[df['task_nr']==task_nr]['current_sequence'].values[0]
                poke_dict_t2['seq'] = df.loc[df['task_nr']==task_nr]['current_sequence'].values[0]


    return poke_dict_t1, poke_dict_t2



def get_all_resps_inpokes_and_outpokes(aligner, poke_dict, single_units,spkT, spkC):
    """ This code gets the average response of all cells to pokes in a single task block
    """
    all_resps = []
    all_resps1 = []
    all_resps2 = []
    
    for unit in single_units:  # [n_:n_+1]: loop over all cells
        
        spk_unit = spkT[np.where(spkC==unit)[0]]  # select all spikes that belong to this cell
        
        resps = [[] for _ in range(9)]
        resps1 = [[] for _ in range(9)]
        resps2 = [[] for _ in range(9)]
        for key,vals in poke_dict.items():  # loop over pokes
            if re.findall('[0-9]',key):  # ignore dictionary items that are metadata like the sequence and graph time
                aligned_T_in = aligner.B_to_A(vals[:,0])  # align pokes into spike times
                aligned_T_out = aligner.B_to_A(vals[:,1])  # align pokes into spike times

                # get the spikes that are in bounds for position encoding
                pks_unit_in_bounds = np.where(np.logical_not(np.isnan(aligned_T_in+aligned_T_out)))[0]
                
                used_pks_in = aligned_T_in[pks_unit_in_bounds].astype('int')  # get pokes aligned with spike times
                used_pks_out = aligned_T_out[pks_unit_in_bounds].astype('int')  # get pokes aligned with spike times
                key = int(key)
                half_npks = int(len(used_pks_in)/2)
                # print(key,half_npks)
                for pk_ix,(tpk_in, tpk_out) in enumerate(zip(used_pks_in, used_pks_out)):  # loop over all pokes to a given port
                    
                    # this is a block of code to split the data in half, useful for looking at stability when you
                    # only have one task block
                    spike_locs = np.logical_and(spk_unit>(tpk_in),spk_unit<(tpk_out))
                    scaleF = (tpk_out - tpk_in)/30000.
                    nSpikes = len(np.where(spike_locs)[0])
                    firing_rate = scaleF*float(nSpikes)

                    if pk_ix <= half_npks:
                        resps2[key].append(firing_rate)
                        
                    else:
                        resps1[key].append(firing_rate)

                    resps[key].append(firing_rate)

                    
                    
        all_resps.append(resps.copy())
        all_resps1.append(resps1.copy())
        all_resps2.append(resps2.copy())
        
    return all_resps, [all_resps1,all_resps2]