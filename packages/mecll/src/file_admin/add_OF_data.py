import re
import os
import sys
import numpy as np
from file_admin_utils import *


""" 
Copy the extracted position data to the folder containing spikes etc
"""


def get_camera_sync_file(session_data_path,video_dir='/Users/yves/team_mouse Dropbox/data_storage/MEC_OF_video'):
    
    """ This will return the matching camera sync file """

    dt_spk = get_session_timestamp(session_data_path=session_data_path)
    subject = get_session_subject(session_data_path)
    
    reg_str = "20[0-3]{2}-[0-9]{2}-[0-9]{2}-[0-9]{6}"
    subj_f_of = [i for i in sorted(os.listdir(video_dir)) if subject in i and 'pinstate' in i and 'OF' in i]
    dts_str_of = [re.findall(reg_str,i)[0] for i in subj_f_of]
    dt_of = [datetime.datetime.strptime(i,'%Y-%m-%d-%H%M%S') for i in dts_str_of]
    
    #delta_t = np.abs(np.array(dt_of) - dt_spk)
    #min_ix = np.argmin(delta_t)
    min_ix, min_dt = closest_timestamp_index(dt_of,dt_spk)
    #print(np.min(delta_t))
    best_F = subj_f_of[min_ix]
    return os.path.join(video_dir,best_F), min_dt



def get_position_path(cam_sync_path):
    """ This is a hack """ 
    pa,pb = os.path.split(cam_sync_path)
    pb2 = pb.replace('_pinstate','').replace('.csv','.mp4') + '_positions.npy'
    return os.path.join(*[pa,'position_extracted_folder',pb2])


if __name__ == '__main__':

    root_folder = '/Users/yves/team_mouse Dropbox/MEC_data/spike_sorted'
    video_dir = '/Users/yves/team_mouse Dropbox/data_storage/MEC_OF_video'
    of_cam_filename = 'OF_camera_timestamps.csv'
    of_pos_filename = 'OF_positions.npy'
    folders = [foldr for foldr in os.listdir(root_folder)]

    for foldr in folders:
        print(foldr)
        full_folder = os.path.join(root_folder,foldr)
        if os.path.isdir(full_folder):
            subject = get_session_subject(foldr)
            cam_sync_path,min_dt_sync = get_camera_sync_file(full_folder)
            pos_path = get_position_path(cam_sync_path)
            try:
                copy_file_to(source_path=cam_sync_path,
                            target_path=os.path.join(full_folder,of_cam_filename))
            except Exception as e: 
                print(e)
            try:
                copy_file_to(source_path=pos_path,
                            target_path=os.path.join(full_folder,of_pos_filename))
            except Exception as e: 
                print(e)
        else:
            print("Not processing {}".format(foldr))




