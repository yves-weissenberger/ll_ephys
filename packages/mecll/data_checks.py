import sys
import os
import re


def files_in_folder(files,folder_path):
    folder_files = os.listdir(folder_path)
    return list(map(lambda x: x in folder_files, files))

def file_specs_in_folder(file_specs,folder_path):
    folder_files = os.listdir(folder_path)
    return list(map(lambda x: any([re.findall(x,i) for i in folder_files]), file_specs))

def has_spike_data(folder_path):

    required_files =['cluster_KSLabel.tsv',
                     'spike_times.npy',
                     'spike_clusters.npy']

    return files_in_folder(required_files,folder_path)

def has_of_data(folder_path):
    
    conditions = (file_specs_in_folder(['pinstate','positions.npy'],folder_path) + 
                  files_in_folder(['timestamps_OF.npy'],folder_path) )
    return conditions

def has_task_data(folder_path):


    conditions = (file_specs_in_folder(['^(?!.*stitch_info).*\.txt.*$'],folder_path) + 
                  files_in_folder([ 'timestamps_task.npy'],folder_path) )

    return conditions


def has_metadata(folder_path):
    pass