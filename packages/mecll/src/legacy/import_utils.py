import os
import sys
import re
import datetime
from datetime import datetime as dt


def get_video_timestamp(fname):
    """ e.g. _2021-07-25-120729"""
    date_string = re.findall(r'_([0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{6})',fname)[0]
    date_object = dt.strptime('2021-07-25-120729','%Y-%M-%d-%H%m%S')
    return date_object


def load_box_dataset_from_folder(root_path):

    #Load behavioural data
    root_path = '/Users/yves/Downloads/example_files4/'

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
    return lines,sync_path_spk,sync_messages_spk,spkT,spkC,cluster_labels,single_units