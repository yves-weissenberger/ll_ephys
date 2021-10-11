import numpy as np
import re

from ..utils import align_activity


def build_spike_array(spk_join,dicts):
    """ This function takes in a series of events and return
    """
    n_timepoints = 25
    n_neurons = len((list(set(spk_join[0]))))

    all_big_array = []

    for dict_ix in range(len(dicts)):
        
        big_array = np.zeros([n_neurons,n_timepoints*9])
        for state_nr in range(9):

            direction = dicts[dict_ix]['direction']
            trial_events = np.array(dicts[dict_ix][str(state_nr)])
            print(len(trial_events))
            if len(trial_events)>0:
                out = align_activity(trial_events,np.array([0,1000]),spk_join,fs=25)
                #if len
                #algined_rates.shape = [n_trials, n_neurons, n_timepoints]
                aligned_rates, t_out, min_max_stretch = out

                mean_aligned_rates = np.nanmean(aligned_rates,axis=0)
            else:
                mean_aligned_rates = np.zeros([n_neurons,n_timepoints]) + np.nan
            if direction=='-1':
                big_array[:,state_nr*n_timepoints:(state_nr+1)*n_timepoints] = np.fliplr(mean_aligned_rates)
            if direction=='1':
                big_array[:,(1+state_nr-1)*n_timepoints:(1+state_nr)*n_timepoints] = mean_aligned_rates

        all_big_array.append(big_array)

    return all_big_array

#def build_spike_array(spkT,spkC):
#    pass

def get_all_resps(aligner,poke_dict,single_units,spkT,spkC,window0=3000,window1=6000,get_time_mean=True):
    """ This code gets the average response of all cells to pokes in a single task block
    """
    all_resps = []
    all_resps1 = []
    all_resps2 = []
    scaleF = (window0+window1)/30000.
    
    for unit in single_units:#[n_:n_+1]:  #loop over all cells
        
        spk_unit = spkT[np.where(spkC==unit)[0]] #select all spikes that belong to this cell
        
        resps = [[] for _ in range(9)]
        resps1 = [[] for _ in range(9)]
        resps2 = [[] for _ in range(9)]
        for key,vals in poke_dict.items():  #loop over pokes
            if re.findall('[0-9]',key): #ignore dictionary items that are metadata like the sequence and graph time
                aligned_T = aligner.B_to_A(vals) #align pokes into spike times

                #get the spikes that are in bounds for position encoding
                pks_unit_in_bounds = np.where(np.logical_not(np.isnan(aligned_T)))[0]
                
                used_pks = aligned_T[pks_unit_in_bounds].astype('int') #get pokes aligned with spike times
                key = int(key)
                half_npks = int(len(used_pks)/2)
                #print(key,half_npks)
                for pk_ix,tpk in enumerate(used_pks):  #loop over all pokes to a given port
                    
                    #this is a block of code to split the data in half, useful for looking at stability when you
                    #only have one task block
                    
                    spike_locs = np.logical_and(spk_unit>(tpk-window0),spk_unit<(tpk+window1))
                    if get_time_mean:
                        nSpikes = len(np.where(spike_locs)[0])
                        firing_rate = scaleF*float(nSpikes)
                    else:
                        spikes = np.zeros(window0+window1)
                        spikes[spk_unit[spike_locs]-tpk] = 1
                        firing_rate = spikes.copy()
                    
                    if pk_ix<=half_npks:
                        resps2[key].append(firing_rate)
                        
                    else:
                        resps1[key].append(firing_rate)

                    resps[key].append(firing_rate)

                    
                    
        all_resps.append(resps.copy())
        all_resps1.append(resps1.copy())
        all_resps2.append(resps2.copy())
        
    return all_resps, [all_resps1,all_resps2]

def get_mean_resps(all_resps_single_trial):
    mus = []
    vs = []
    mu_g1 = []
    var_g1 = []
    for neuron in all_resps_single_trial:
        tmp_mu = []
        tmp_var = []
        for poke in neuron:
            tmp_mu.append(np.mean(poke))
            tmp_var.append(np.var(poke))
        mu_g1.append(tmp_mu)
        var_g1.append(tmp_var)

    return np.array(mu_g1),np.array(var_g1)


def get_activity_for_each_poke(df, spkT, spkC, single_units, aligner, window0=3000, window1=6000):
    """ 
    This function takes in dataframes like the one produced by proc_beh.build_poke_df
    and returns spiking of all neurons searched for in single_units. Need to update this to
    include indicies that are not nans

    """

    poke_dict ={}

    for port_nr in np.unique(df['port'].values):
        v = df.loc[(df['port']==port_nr)]['time'].values
        poke_dict[str(port_nr)] = [float(i) for i in v]


    #st = time.time()
    used_pokes = np.zeros(len(df))
    
    scaleF = (window0+window1)/30000.
    all_spk_store = []

    for unit in single_units:#
        spk_unit = spkT[np.where(spkC==unit)[0]] #select all spikes that belong to this cell
        spk_store = []
        for nr,row in df.iterrows():
            aligned_T = aligner.B_to_A(row['time'])
            
            if not np.isnan(aligned_T):

                tpk = aligned_T
                spike_locs = np.logical_and(spk_unit>(tpk-window0),spk_unit<(tpk+window1))
                nSpikes = len(np.where(spike_locs)[0])
                firing_rate = scaleF*float(nSpikes)
                spk_store.append(firing_rate)
                used_pokes[nr] = 1
            else:
                spk_store.append(np.nan)
        all_spk_store.append(spk_store)

    return np.array(all_spk_store)


def get_within_and_across_correlations(all_resps1_g1,all_resps2_g1,all_resps1_g2,all_resps2_g2,all_resps_g1,all_resps_g2):
    
    """ Get correlations between neural activity across behaivour on the two different graphs
        as well as across split halves of the same graph
    """

    ccs_within1 = []
    for r1,r2 in zip(all_resps1_g1,all_resps2_g1):
        ccs_within1.append(np.corrcoef(r1,r2)[0,1])

    ccs_within1 = np.array(ccs_within1)
    print(np.nanmean(ccs_within1))

    ccs_within2 = []
    for r1,r2 in zip(all_resps1_g2,all_resps2_g2):
        ccs_within2.append(np.corrcoef(r1,r2)[0,1])
    ccs_within2 = np.array(ccs_within2)

    print(np.nanmean(ccs_within2))


    ccs_within = (np.array(ccs_within1) + np.array(ccs_within2))/2.
    ccs_within = np.min(np.vstack([np.array(ccs_within1),np.array(ccs_within2)]),axis=0)


    #
    ccs_across = []
    for r1,r2 in zip(all_resps_g1,all_resps_g2):
        ccs_across.append(np.corrcoef(r1,r2)[0,1])

    ccs_across = np.array(ccs_across)
    print(np.nanmean(ccs_across))
    
    return ccs_within,ccs_across,(ccs_within1,ccs_within2)