import os

results_directory = '/Users/yves/Desktop/mecll_tables'
data_directory = '/Users/yves/team_mouse Dropbox/MEC_data/spike_sorted'


cell_df_path = os.path.join(results_directory,'cell_df.csv')
session_index_df_path = os.path.join(results_directory,'session_info_df.csv')


#this is the directory in which intermediate results are saved
result_store_directory = os.path.join(results_directory,'res')