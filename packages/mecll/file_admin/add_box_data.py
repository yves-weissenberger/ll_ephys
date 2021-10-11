import re
import os
import sys
import numpy as np
from file_admin_utils import *


def get_behaviour_file(session_data_path,subject_folder):


    dt_spk = get_session_timestamp(session_data_path=session_data_path)
    reg_str = '.*?-(.*?-[0-9]*).txt'
    sessions = [i for i in sorted(os.listdir(subject_folder)) 
                if os.path.isfile(os.path.join(subject_folder,i))]
    dts_str_box = [re.findall(reg_str,i)[0] for i in sessions]
    dt_box = [datetime.datetime.strptime(i,'%Y-%m-%d-%H%M%S') for i in dts_str_box]

    min_ix,min_dt = closest_timestamp_index(dt_box,dt_spk)
    best_F = sessions[min_ix]
    return os.path.join(subject_folder,best_F),min_dt

if __name__ == '__main__':

    root_folder = '/Users/yves/team_mouse Dropbox/MEC_data/spike_sorted'
    behaviour_dir = '/Users/yves/Downloads/behavioural_data_for_matching_to_ephys_mec'

    folders = [foldr for foldr in os.listdir(root_folder)]

    for foldr in folders:
        print(foldr)
        full_folder = os.path.join(root_folder,foldr)
        if os.path.isdir(full_folder):
            subject = get_session_subject(foldr)
            subject_behaviour_folder = os.path.join(behaviour_dir,'mecLL_'+subject)
            beh_path, min_dt = get_behaviour_file(full_folder,subject_behaviour_folder)
            copy_file_to(source_path=beh_path,
                target_path=os.path.join(full_folder,os.path.split(beh_path)[1]))

        else:
            print("Not processing {}".format(foldr))




