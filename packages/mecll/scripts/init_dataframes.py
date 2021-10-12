import os
import sys
import re
import pandas as pd


package_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(package_dir)

from config import results_directory



if __name__=='__main__':


    if 'session_info_df.csv' not in os.listdir(results_directory):
        session_df  = pd.DataFrame(columns=['subject','date','of','task','spk','path','id'])
        session_df.to_csv(os.path.join(results_directory,'session_info_df.csv'))


    if 'cell_analysis_df.csv' not in os.listdir(results_directory):
        cell_df = pd.DataFrame(columns=['cell_id','cluster_id','subject','date','session_id','cluster_settings'])
        cell_df.to_csv(os.path.join(results_directory,'cell_df.csv'))

    if 'session_analaysis_df.csv' not in os.listdir(results_directory):
        cell_df = pd.DataFrame(columns=['subject','date','session_id'])
        cell_df.to_csv(os.path.join(results_directory,'session_df.csv'))
