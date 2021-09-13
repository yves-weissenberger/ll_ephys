import os
import re
import sys
import numpy as np
import pandas as pd
from .rsync import Rsync_aligner

def load_data(root_path):



    all_fs = os.listdir(root_path)

    task_path = os.path.join(root_path,[i for i in all_fs if '.txt' in i][0])
    lines = open(task_path,'r').readlines()


    #load neural data timestamps

    sync_path_spk = os.path.join(root_path,[i for i in all_fs if i=='timestamps.npy'][0])#'/Users/yves/Downloads/example_files/timestamps.npy'
    sync_messages_spk = np.load(sync_path_spk)[::2][:-1].astype('float')


    #load spiking data
    #ROOT = '/Users/yves/Downloads/example_files/'
    spkT = np.load(os.path.join(root_path,[i for i in all_fs if i=='spike_times.npy'][0])).flatten()
    spkC = np.load(os.path.join(root_path,[i for i in all_fs if i=='spike_clusters.npy'][0])).flatten()
    cluster_labels = pd.read_table(os.path.join(root_path,[i for i in all_fs if '.tsv' in i][0]))


    #select single units
    single_units = np.where(cluster_labels.KSLabel=='good')[0]
    #Extract the timestamps of the sync-pulses between the neural data and the behavioural data
    events = eval(lines[9][2:])
    bnc_ev = events['BNC_input']
    sync_messages_task =[int(re.findall(r' ([0-9]*)',l)[0]) for l in lines if str(bnc_ev)+'\n' in l and l[0]=='D']
    poke_event_ids = [events['poke_'+str(i)] for i in range(1,10)]
    #print(poke_event_ids)

    #align neural and behavioural timestamps
    aligner = Rsync_aligner(sync_messages_spk,np.array(sync_messages_task)[1:],units_A=1/30.,units_B=1)

    out = (spkT,spkC,single_units,events,lines,aligner)
    return out