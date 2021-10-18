import numpy as np
from scipy.stats import spearmanr
import os
import sys
import re
import pandas as pd
from classes import all_sessions,single_session_analysis
package_dir = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0]
sys.path.append(package_dir)
from mecll.process_data.proc_beh import get_all_transitions, get_transitions_state, build_poke_df, get_in_task_pokes
from mecll.process_data.proc_neural import get_all_resps, get_mean_resps


def get_response_correlation(set1,set2):
    rcorr = []
    for r1,r2 in zip(set1,set2):
        rcorr.append(np.corrcoef(r1,r2)[0,1])
    return np.array(rcorr)


def get_task_responses(df):
    """
    How to dynamically filter a pandas dataframe with functions
    """
    poke_dict_t1 ={}
    poke_dict_t2 = {}
    for port_nr in np.unique(df['port'].values):
        for task_nr in range(2):
            task_nr = str(task_nr)
            v = df.loc[(df['port']==port_nr) &
                    (df['correct']==True) & 
                    #(df['next_correct']==True) &
                    #(df['reward']==True) &
                    (df['port_repeat']==False) & 
                    #(df['RT']<1600) &
                    (df['task_nr']==task_nr)
                    ]['time'].values
            #v = np.array(v).astype('float')
            if task_nr=='0':
                print(task_nr,len(v),str(port_nr))
                poke_dict_t1[str(port_nr)] = [float(i) for i in v]
            else:
                poke_dict_t2[str(port_nr)] = [float(i) for i in v]
                
            
            if port_nr==8:
                poke_dict_t1['task_nr'] = str(task_nr)
                poke_dict_t2['task_nr'] = str(task_nr)
                poke_dict_t1['graph_type'] = df.loc[df['task_nr']==task_nr]['graph_type'].values[0]
                poke_dict_t2['graph_type'] = df.loc[df['task_nr']==task_nr]['graph_type'].values[0]
                poke_dict_t1['seq'] = df.loc[df['task_nr']==task_nr]['current_sequence'].values[0]
                poke_dict_t2['seq'] = df.loc[df['task_nr']==task_nr]['current_sequence'].values[0]
    return poke_dict_t1, poke_dict_t2


if __name__=='__main__':

    all_sess = all_sessions()
    ctr = 0
    for session_id,subject,date,session_path in all_sess:
        print(session_path)
        
        out = all_sess.load_task_session(session_path)
        spkT,spkC,single_units,events,lines,aligner = out
        


        #process task data
        all_poke_dict = get_in_task_pokes(lines,events)
        df = build_poke_df(lines,events)
        df['previous_port'][1:] = df['port'][:-1]
        df['previous_state'][1:] = df['state'][:-1]

        poke_dict_t1, poke_dict_t2 = get_task_responses(df)

        tmp_ = get_all_resps(aligner,poke_dict_t1,single_units,spkT,spkC,window0=6000,window1=6000)
        all_resps_g1_single_trial, (all_resps1_g1_single_trial,all_resps2_g1_single_trial)  = tmp_
        tmp_ = get_all_resps(aligner,poke_dict_t2,single_units,spkT,spkC,window0=6000,window1=6000)
        all_resps_g2_single_trial, (all_resps1_g2_single_trial,all_resps2_g2_single_trial) = tmp_


        all_resps_g1,_ = get_mean_resps(all_resps_g1_single_trial)
        all_resps_g2,_ = get_mean_resps(all_resps_g2_single_trial)


        all_resps1_g1,_ = get_mean_resps(all_resps1_g1_single_trial)
        all_resps1_g2,_ = get_mean_resps(all_resps1_g2_single_trial)

        all_resps2_g1,_ = get_mean_resps(all_resps2_g1_single_trial)
        all_resps2_g2,_ = get_mean_resps(all_resps2_g2_single_trial)

        #
        ccs_within1 = get_response_correlation(all_resps1_g1,all_resps2_g1)
        ccs_within2 = get_response_correlation(all_resps1_g2,all_resps2_g2)
        
        ccs_across1 = get_response_correlation(all_resps1_g1,all_resps1_g2)
        ccs_across2 = get_response_correlation(all_resps1_g2,all_resps1_g1)

        ccs_across_all = get_response_correlation(all_resps_g1,all_resps1_g2)

        sess = single_session_analysis(subject=subject,date=date,session_id=session_id)
        sess.add_data({'within_task_1_correlations': ccs_within1},single_units)
        sess.add_data({'within_task_2_correlations': ccs_within2},single_units)

        sess.add_data({'across_task_correlations_all': ccs_across1},single_units)
        sess.add_data({'across_task_correlations_1': ccs_across2},single_units)
        sess.add_data({'across_task_correlations_2': ccs_across_all},single_units)
