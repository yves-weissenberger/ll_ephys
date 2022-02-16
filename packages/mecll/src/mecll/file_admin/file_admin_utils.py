import numpy as np
import os
import re
import sys
import shutil
import datetime

def get_camera_sync_file(session_data_path,video_dir='/Users/yves/team_mouse Dropbox/data_storage/MEC_OF_video'):
    
    """ This will return the matching camera sync file """

    dt_spk = get_session_timestamp(session_data_path)
    subject = re.findall('([0-9]*)_ks25',session_data_path)[0]

    

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

def get_of_timestamp(video_dir):
    """ get timestamp of open field data """
    return None

def closest_timestamp_index(dt_array,dt_target):
    delta_t = np.abs(np.array(dt_array) - dt_target)
    return np.argmin(delta_t), min(delta_t)



def get_session_date(session_data_path): return re.findall("20[0-3]{2}-[0-9]{2}-[0-9]{2}",session_data_path)[0]

def get_session_subject(session_data_path): return re.findall('([0-9]*)_ks25',session_data_path)[0]


def get_session_timestamp(lines=None,session_data_path=None,session_half='OF'):

    """
    Get an exact datetime timestamp for when the recording was started. Need to specify
    whether looking at the open field session (session_half=OF) or the other half
    
    """
    session_index = (session_half == 'OF')  #always have the box before the open field session, I think...
    if lines is None:
        assert session_data_path, 'You ,must either provide the session data path or the lines in stitch_info'
        lines = load_stitch_info(session_data_path)

    file_name = eval(lines[0])[session_index]
    reg_str = "20[0-3]{2}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}"
    dt_spk = datetime.datetime.strptime(re.findall(reg_str,file_name)[0],'%Y-%m-%d_%H-%M-%S')

    return dt_spk    


        



def load_stitch_info(session_data_path):
    """ returns the lines as a list from the stitch_info_file """
    stitch_name = [i for i in os.listdir(session_data_path) if 'stitch_info' in i][0]
    with open(os.path.join(session_data_path,stitch_name),'r') as f:
        lines = f.readlines()
    return lines




###

def copy_file_to(source_path,target_path):

    if os.path.isfile(target_path):
        raise Exception("WARNING YOU ARE OVERWRITING AN EXISING FILE ABORTING!!")

    with open(target_path, "wb") as nf:
            with open(source_path,'rb') as of:
                shutil.copyfileobj(of, nf)
    return 1
