import os
import re
import sys
import numpy as np
import pandas as pd
from rsync import Rsync_aligner



def get_n_samples_first_half(session_data_path,divisor=384*2):
    """ Divisor is 384 channels * 2bits per int16"""
    pth = os.path.join(session_data_path,[i for i in os.listdir(session_data_path) if 'stitch_info' in i][0])
    with open(pth,'r') as f:
        lines = f.readlines()
    size1 = int(re.findall(r':\s([0-9]*)',lines[1])[0])
    return int(size1/divisor)


def find_task_file(root_path):
    for f in os.listdir(root_path):
        if 'task_file_' in f:
            return f


def load_data(root_path,align_to='task',camera_frame_rate=30):

    """ 
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

    aligner = Rsync_aligner(sync_messages_spk+sample_offset,
                            sync_messages_ext,
                            units_A=units_A,
                            units_B=1)

    out = [spkT,spkC,single_units,events,lines,aligner]
    return out