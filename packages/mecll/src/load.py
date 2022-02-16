from typing import Tuple
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from mecll.retreat_preprocess import get_task_responses_by_direction

from .rsync import *
from .retreat_preprocess import get_task_response_by_direction



def get_n_samples_first_half(session_data_path:str ,divisor: int =384*2):
    """ Divisor is 384 channels * 2bits per int16"""
    pth = os.path.join(session_data_path,[i for i in os.listdir(session_data_path) if 'stitch_info' in i][0])
    with open(pth,'r') as f:
        lines = f.readlines()
    size1 = int(re.findall(r':\s([0-9]*)',lines[1])[0])
    return int(size1/divisor)


def find_task_file(root_path: str) -> str:
    for f in os.listdir(root_path):
        if 'task_file_' in f:
            return f


def load_data(root_path,align_to='task',camera_frame_rate=30):

    """
    This returns spike times in the reference frame of the recording 
    NOT in the frame of any of the behaviour hardware.


    Arguments:
    =====================


    align to:       str
                    must be either 'task' | 'OF'
    """

    all_fs = os.listdir(root_path)

    task_path = os.path.join(root_path,find_task_file(root_path))
    lines = open(task_path,'r').readlines()


    #load neural data timestamps


    #these are the sync messages that the spikes have
    sync_path_spk = os.path.join(root_path,[i for i in all_fs if i=='timestamps_'+align_to+'.npy'][0])
    sync_messages_spk = np.load(sync_path_spk)[::2][:-1].astype('float')


    #load spiking data
    #ROOT = '/Users/yves/Downloads/example_files/'
    spkT = np.load(os.path.join(root_path,[i for i in all_fs if i=='spike_times.npy'][0])).flatten()
    spkC = np.load(os.path.join(root_path,[i for i in all_fs if i=='spike_clusters.npy'][0])).flatten()
    cluster_labels = pd.read_table(os.path.join(root_path,[i for i in all_fs if 'KSLabel.tsv' in i][0]))


    #select single units
    single_units = np.where(cluster_labels.KSLabel=='good')[0]

    events = eval(lines[9][2:])
    bnc_ev = events['BNC_input']
    sync_messages_task =[int(re.findall(r' ([0-9]*)',l)[0]) for l in lines if str(bnc_ev)+'\n' in l and l[0]=='D']

    
    #load camera sync messages
    if align_to=='OF':
        cam_sync_path = os.path.join(root_path,'OF_camera_timestamps.csv')
        sync_df_of = pd.read_csv(cam_sync_path,header=None)
        sync_messages_cam = np.where((sync_df_of[0].values[1:] - sync_df_of[0].values[:-1])<0)[0].astype('float')


    if align_to=='OF':
        sample_offset = get_n_samples_first_half(root_path)
        sync_messages_ext = sync_messages_cam
        units_A = camera_frame_rate/30000
    else:
        sync_messages_ext = np.array(sync_messages_task)
        units_A = 1/30.
        sample_offset = 0
    #align neural and behavioural timestamps
    print(len(sync_messages_spk),len(sync_messages_ext))
    aligner = Rsync_aligner(sync_messages_spk+sample_offset,
                            sync_messages_ext,
                            units_A=units_A,
                            units_B=1)

    return spkT,spkC,single_units,events,lines,aligner



@dataclass
class preprocessed_dataset:
    neuron_response_table: None
    task_event_table: None
    seq0: None
    seq1: None
    graph_type0: None
    graph_type1: None
    firing_rate_maps: None

def load_preprocessed_data(selected_session: int, 
                            all_data_dir: str = '/Users/yves/Desktop/retreat_data_dir/data/'
                           ) -> preprocessed_dataset:
    all_data_folders = sorted([i for i in os.listdir(all_data_dir) if 'ks25' in i])
    root_dir = os.path.join(all_data_dir,all_data_folders[selected_session])

    # This is basically a big table (you can open it in excel) which contains
    # relevant information about each time the animal poked one of the ports
    task_event_df = pd.read_csv(os.path.join(root_dir,'task_event_table.csv'),index_col=0)

    #
    response_table = np.load(os.path.join(root_dir,'neuron_response_table.npy'))
    #alternatively to change the time window


    #not all cluster in spkC correspond to single units. Single units is an array of the clusters that are single units
    single_units = np.load(os.path.join(root_dir,'single_units.npy'))
    
    
    seq0 = np.array(eval(task_event_df.loc[task_event_df['task_nr']==0]['current_sequence'].values[0]))
    seq1 = np.array(eval(task_event_df.loc[task_event_df['task_nr']==1]['current_sequence'].values[0]))
    
    
    graph_type0 = task_event_df.loc[task_event_df['task_nr']==0]['graph_type'].values[0]
    graph_type1 = task_event_df.loc[task_event_df['task_nr']==0]['graph_type'].values[0]
    
    firing_rate_maps = get_task_responses_by_direction(task_event_df,
                                                       response_table)

    dataset = preprocessed_dataset(neuron_response_table = response_table,
                                   task_event_table = task_event_df,
                                   seq0 = seq0,
                                   seq1 = seq1,
                                   graph_type0 = graph_type0,
                                   graph_type1 = graph_type1,
                                   firing_rate_maps = firing_rate_maps
                                   )
    
    return dataset

