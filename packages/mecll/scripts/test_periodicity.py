import os
import sys
import re
from classes import all_sessions,single_session_analysis


import numba
import numpy as np
import pandas as pd
import scipy.optimize as op

package_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(package_dir)

from process_data.proc_beh import build_poke_df
from process_data.proc_neural import get_all_resps, get_mean_resps


def get_mean_resps(all_resps_single_trial):
    mus = []
    vs = []
    mu_task1 = []
    var_task1 = []
    for neuron in all_resps_single_trial:
        tmp_mu = []
        tmp_var = []
        for poke in neuron:
            tmp_mu.append(np.mean(poke))
            tmp_var.append(np.var(poke))
        mu_task1.append(tmp_mu)
        var_task1.append(tmp_var)

    return np.array(mu_task1),np.array(var_task1)

@numba.jit(nopython=True)
def fit_sin2(x,*params):
    #t_ = np.linspace(0,2*np.pi,9)
    y,t_ = params
    #y = np.array(params)
    #y -= np.mean(y)
    #y /=np.max(y)
    pred = np.cos(x[0]*t_ + x[1])
    cc = np_pearson_cor(pred,y)
    if np.isnan(cc): cc = -100
    return -cc


@numba.jit(nopython=True)
def np_pearson_cor(x, y):
    """ from 
        https://cancerdatascience.org/blog/posts/pearson-correlation/
    """
    xv = x - np.mean(x)
    yv = y - np.mean(y)
    xvss = np.sum(xv * xv)
    yvss = np.sum(yv * yv)
    result = np.dot(xv.T, yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return result[0][0]# np.maximum(np.minimum(result, 1.0), -1.0)


def get_poke_dicts(df):


    poke_dict_t1 ={}
    poke_dict_t2 = {}

    for port_nr in np.unique(df['port'].values):
        for task_nr in range(2):
            task_nr = str(task_nr)
            v = df.loc[(df['port']==port_nr) &
                    (df['correct']==True) & 
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





def fit_sine_waves(resps_state_task1,resps_state_task2,NSHUFF=1000):

    """
    
    Arguments:
    =======================

    resps_state_task1:     np.array


    resps_state_task2:     np.array
                        
    """


    resp_sets = [resps_state_task1,resps_state_task2]

    t_ = np.linspace(0,2*np.pi,9)
    params_ranges = [slice(0,4,.5),slice(0,2*np.pi,np.pi/9)]

    n_neurons = resps_state_task1.shape[0]
    print(n_neurons)

    res_sets = []
    ccs_sets = []
    cc_shuff_sets = []
    for neuron_ix in range(n_neurons):
        
        tmp_cc = []
        tmp_res = []
        tmp_cc_shuff = []
        for ix,resp_set in enumerate(resp_sets):    

            sys.stdout.write('\r running cell:{}'.format(neuron_ix))
            sys.stdout.flush()

            spks = resp_set[neuron_ix]
            res1 = op.brute(fit_sin2,params_ranges,args=(spks,t_),finish=None)

            
            cc1 = np.corrcoef(spks,np.cos(res1[0]*t_ + res1[1]))[0,1]

            cc1_shuff = []
            for _ in range(NSHUFF):
                spks_shuff = np.random.permutation(spks)
                res1_shuff = op.brute(fit_sin2,params_ranges,args=(spks_shuff,t_),finish=None)

                cc1_shuff.append(np.corrcoef(spks_shuff,np.cos(res1_shuff[0]*t_ + res1_shuff[1]))[0,1])

            tmp_cc.append(cc1)
            tmp_cc_shuff.append(cc1_shuff)
            tmp_res.append(res1)
        res_sets.append(tmp_res)
        ccs_sets.append(tmp_cc)
        cc_shuff_sets.append(tmp_cc_shuff)

    return np.array(res_sets), np.array(ccs_sets), np.array(cc_shuff_sets)
    
if __name__=='__main__':

    all_sess = all_sessions()
    ctr = 0

    #this time window is specified in the clock of the 
    params = {'window_pre':6000,
              'window_post':6000,
              'filter_by': 'corr_port_taskNr',
              'NSHUFF': 1000,
              'remove_space': True
              }

    for session_id,subject,date,session_path in all_sess:
        print(session_path)
        
        out = all_sess.load_task_session(session_path)
        spkT,spkC,single_units,events,lines,aligner,position = out
        df = build_poke_df(lines,events)

        poke_dict_t1, poke_dict_t2 = get_poke_dicts(df)


        #Here get the mean arrays of responses under the different criteria
        ##########################################################################################################
        tmp_ = get_all_resps(aligner,poke_dict_t1,single_units,spkT,spkC,window0=params['window_pre'],
                                                                         window1=params['window_post'])
        all_resps_task1_single_trial, (all_resps1_task1_single_trial,all_resps2_task1_single_trial)  = tmp_

        tmp_ = get_all_resps(aligner,poke_dict_t2,single_units,spkT,spkC,window0=params['window_pre'],
                                                                         window1=params['window_post'])
        all_resps_g2_single_trial, (all_resps1_g2_single_trial,all_resps2_g2_single_trial) = tmp_

        #here get the mean responses
        all_resps_task1,_ = get_mean_resps(all_resps_task1_single_trial)
        all_resps_g2,_ = get_mean_resps(all_resps_g2_single_trial)
        ##########################################################################################################


        #if de-mean th spatial responses
        if params['remove_space']:
            mean_resps_state = (all_resps_task1 + all_resps_g2)/2.
        else:
            mean_resps_state = np.zeros_like(all_resps_task1)

        resps_state_task1 = (all_resps_task1-mean_resps_state)[:,poke_dict_t1['seq']]
        resps_state_task2 = (all_resps_g2-mean_resps_state)[:,poke_dict_t2['seq']]
        #Save the session data
        sess = single_session_analysis(subject=subject,date=date,session_id=session_id)

        res_sets, ccs_sets, cc_shuff_sets = fit_sine_waves(resps_state_task1,resps_state_task2,NSHUFF=params['NSHUFF'])


        #this needs changing currently in the 
        print('\n')
        


        #include in results freq_task1, freq_task2, p_value_periodicity1, p_value_periodicity2
        param_string = "_used_params_" + repr(params)
        sess.add_data({'freq_task1_'+param_string: XXXXX},single_units,
                        save_format={'spatial_rate_maps'+param_string: '.npy'})

        sess.add_data({'freq_task2_'+param_string: XXXXX},single_units,
                        save_format={'spatial_rate_maps'+param_string: '.npy'})


        sess.add_data({'p_value_freq_task1_'+param_string: XXXXX},single_units,
                        save_format={'spatial_rate_maps'+param_string: '.npy'})


        sess.add_data({'p_value_freq_task2_'+param_string: XXXXX},single_units,
                        save_format={'spatial_rate_maps'+param_string: '.npy'})
