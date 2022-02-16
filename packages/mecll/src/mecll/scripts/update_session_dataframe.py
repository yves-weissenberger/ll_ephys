import pandas as pd
import os
import sys
import re
import hashlib 
package_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(package_dir)

from data_checks import *
from file_admin import get_session_date, get_session_subject





from config import results_directory, data_directory

if __name__=="__main__":

    if 'session_info_df.csv' in os.listdir(results_directory):
        df = pd.read_csv(os.path.join(results_directory,'session_info_df.csv'))
        df.drop(df.filter(regex="Unnamed"),axis=1, inplace=True)

    else:
        raise Exception("Need to initialse the dataframe first")

    results_folders = os.listdir(data_directory)
    results_folders = filter(lambda x: '.DS_Store' not in x, results_folders)
    for pth_ in results_folders:
        
        date = get_session_date(pth_)
        subject = get_session_subject(pth_)

        id_ = int(hashlib.sha1(pth_.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
        if id_ not in df['id']:
            pth = os.path.join(data_directory,pth_)
            has_spk = all(has_spike_data(pth))
            has_of = all(has_of_data(pth))
            has_task = all(has_task_data(pth))

            dct = {'subject': subject,
                'date': date,
                'of': has_of,
                'task': has_task,
                'path':pth,
                'spk': has_spk,
                'id': id_
                    }

            df = df.append(dct,ignore_index=True)

    print(len(df))
    df.to_csv(os.path.join(results_directory,'session_info_df.csv'),)


