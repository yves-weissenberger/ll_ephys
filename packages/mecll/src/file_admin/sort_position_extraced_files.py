import os
import sys
import re
import shutil
from functools import partial
from .subjects import subjects as subject_map

if __name__=='__main__':


    target_folder = '/Users/Yves/Desktop/folder_with_all_stuff/'
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)


    data_path = '/Users/yves/Downloads/behavioural_data_for_matching_to_ephys_mec'
    #subject_folders = map(partial(os.path.join,behaviour_dat_path),os.listdir(behaviour_dat_path))


    subjects = ['39951','39964','39955','39964','39967']
    for folder in subject_folders:
        check_s = [s in folder for s in subjects]
        if any(check_s):
            print(folder)
            s = subject_map.get(subjects[check_s.index(1)])
            for f in os.listdir(folder):
                original_path = os.path.join(folder,f)
                if os.path.isfile(original_path):
                    date = re.findall('.*?-(.*?)-[0-9]*.txt',f)[0].replace('-','')

                    subject_date_folder = '_'.join([s,date])
                    subject_date_folder_fullpath = os.path.join(target_folder,subject_date_folder)


                    #if this folder doesn't exist make it
                    if not os.path.isdir(subject_date_folder_fullpath): os.mkdir(subject_date_folder_fullpath)

                    target_filepath = os.path.join(subject_date_folder_fullpath,f)
                    #original_filepath = os.path.join()

                    with open(target_filepath, "ab") as nf:
                        with open(original_path,'rb') as of:
                            shutil.copyfileobj(of, nf)
                    #print('\ndone!')

                    
                        
                    
                    
                


