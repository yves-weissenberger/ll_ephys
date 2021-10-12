import numpy as np
import os
import sys
import re
import pandas as pd

package_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(package_dir)

from config import cell_df_path, session_index_df_path, results_directory,result_store_directory 
from load import load_data


import hashlib 



class single_session_analysis:
    """ 
    This is a helper class that implements the loading and saving of results of analyses
    that are common to all the different ones (e.g. writing intermediate results to file)
    or adding data to the dataframe.
    """
    def __init__(self,subject,date,session_id):

        self.result_store_directory = result_store_directory 
        if not os.path.isdir(result_store_directory): os.mkdir(result_store_directory)

        #if type()
        #self.df = pd.DataFrame(columns=['subject','date','session_id']+list(analyses))
        self.cell_df_path =cell_df_path
        self.subject = subject
        self.date = date
        self.session_id = session_id
        self.info_dict = {'subject': subject,
                          'date': date,
                          'session_id': session_id,
                          'cluster_settings': 'default'}

        
        

    def _unique_cell_id(self,subject,date,session_id,cluster_id):
        cell_string = '_'.join([str(subject),str(date),str(session_id),str(cluster_id),str(self.info_dict['cluster_settings'])])
        #id_ = int(hashlib.sha1(cell_string.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
        return cell_string


    def _get_cell_ids(self,data_dict):
        cell_ids = []
        for cluster_id in data_dict['cluster_id']:
            tmp = self._unique_cell_id(self.subject,self.date,self.session_id,cluster_id)
            cell_ids.append(tmp)
            
        return cell_ids

    def add_data(self,data_dict,cluster_ids):
        """ pass data to be added here in the form of a list of """
        
        new_data_dict  = {**data_dict, **self.info_dict}
        new_data_dict['cluster_id'] = cluster_ids
        new_data_dict['cell_id'] = self._get_cell_ids(new_data_dict)
        df_new_data = pd.DataFrame.from_dict(new_data_dict,orient='columns')
        #print(df_new_data)
        df_cells = pd.read_csv(cell_df_path,index_col=0)
        for k in data_dict.keys():
            if k not in df_cells.columns:
                df_cells[k] = None

            for cell_ix,cell_id in enumerate(new_data_dict['cell_id']):
                if cell_id not in df_cells['cell_id']:
                    #print(1)
                    d2 = dict([(k,self.info_dict[k]) if k in self.info_dict.keys() else (k,None) for k in df_cells.columns  ])
                    d2['cell_id'] = cell_id; d2['cluster_id'] = cluster_ids[cell_ix]
                    df_cells = df_cells.append(d2, ignore_index=True)

                df_cells.loc[df_cells['cell_id']==cell_id,k] = data_dict[k][cell_ix]

        new_df_cells = df_cells
                #df_cells.loc[df_cells]
        #df_cells.set_index('cell_id')
        #df_new_data.set_index('cell_id')
        #print(df_cells)
        
        
        #cols_to_use = df_new_data.columns.difference(df_cells.columns)
        #new_df_cells = pd.merge(df_new_data, df_cells, left_index=True, right_index=True, how='outer')
        
        
        #new_df_cells = pd.concat([df_cells,df_new_data]).drop_duplicates(subset='cell_id',keep='last').reset_index(drop=True)

        #new_df_cells = new_df_cells.groupby(['cell_id'],as_index=False).agg()#.agg(lambda x: ''.join(x.fillna(''))).reset_index(drop=True)
        #new_df_cells = df_cells.join(df_new_data)


        #df_cells.merge(df_new_data, 'outer', 'cell_id').groupby(lambda x: x.split('_')[0], axis=1).last()
        #new_df_cells = df_cells


        #print(cols_to_use)
        
        #new_df_cells = pd.concat([df_cells,df_new_data],how='outer',ignore_index=False)
        #new_df_cells = pd.concat([df_cells,df_new_data],
        #                        join="outer",
        #                        ignore_index=False,
        #                        keys=None,
        #                        levels=None,
        #                        names=None,
        #                        verify_integrity=False,
        #                        copy=True,)
        #print(new_df_cells)
        #new_df_cells = df_new_data.merge(df_cells, how='outer', on=['subject','date','cell_id']).drop_duplicates(keep='first')

        new_df_cells.to_csv(self.cell_df_path)


    def save_intermediate_results(self):
        return
        #os.mkdir()

    


class all_sessions:
    """ This is meant to be a helper class that is useful for dispatching
        analyses
    """
    def __init__(self,need_spk=True,need_of=True,need_task=True):
        self.df_index = pd.read_csv(session_index_df_path)
        
        self.need_spk=need_spk; self.need_of=need_of; self.need_task=need_task

    
    def __iter__(self):
        self._session_list = self.iter_sessions()
        return iter(self._session_list)
    
    def iter_sessions(self):


        session_paths = []
        for row_ix,session_row in self.df_index.iterrows():
            if ((session_row['spk'] or not self.need_spk) and
                (session_row['of'] or not self.need_of) and
                (session_row['task'] or not self.need_task)
            ):

                session_paths.append([session_row['id'],session_row['subject'],session_row['date'],session_row['path']])
        return session_paths

    def load_task_session(self,session_path):
        out = load_data(session_path,align_to='task')
        #spkT,spkC,single_units,events,lines,aligner = out
        return out


    def load_of_session(self,session_path):
        out = load_data(session_path,align_to='OF')
        #spkT,spkC,single_units,events,lines,aligner = out
        return out