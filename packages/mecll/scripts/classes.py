import numpy as np
import os
import sys
import re
import pandas as pd
import pickle

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

        self.results_store_directory = result_store_directory 
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

    def add_data(self,data_dict,cluster_ids,save_format={},params=None):
        """ pass data to be added here in the form of a list of 
        
        The basic idea with save list is that you can pass values of the dictionary
        that you don't want to explicitly store in the dataframe to the add_data function 
        in the save list. If they key is in this list, then a new folder will be created that
        will contain the data.

        Question: how do you pass the parameters of the analysis throuh?
        
        """
        
        new_data_dict  = {**data_dict, **self.info_dict}
        new_data_dict['cluster_id'] = cluster_ids
        new_data_dict['cell_id'] = self._get_cell_ids(new_data_dict)

        df_cells = pd.read_csv(cell_df_path,index_col=0)
        df_cells.set_index('cell_id')


        #for each key in the data dictionary
        for k in data_dict.keys():

            if k in save_format.keys():
                for cell_ix,(res,cell_id) in enumerate(zip(new_data_dict[k],new_data_dict['cell_id'])):
                    if save_format[k]=='.npy':
                        save_dir = k.replace('.','dot').replace(' ','_-_')
                        save_dir_path = os.path.join(self.results_store_directory,save_dir)
                        if not os.path.isdir(save_dir_path): os.mkdir(save_dir_path)
                        save_path = os.path.join(save_dir_path,cell_id)
                        np.save(save_path + '.npy',res)
                    else:
                        save_dir = k.replace('.','dot').replace(' ','_-_')
                        save_dir_path = os.path.join(self.results_store_directory,save_dir)
                        if not os.path.isdir(save_dir_path): os.mkdir(save_dir_path)
                        save_path = os.path.join(save_dir_path,cell_id)
                        with open(save_path,'w') as f:
                            pickle.dump(res,f)


                    data_dict[k][cell_ix] = save_path




            #if this key is not a column yet
            if 'Analysis_'+k not in df_cells.columns:
                df_cells['Analysis_'+k] = None  #initialise this column


            #for each cell in the new data
            for cell_ix,cell_id in enumerate(new_data_dict['cell_id']):
                #if the cell_id is not in the dictionary
                if cell_id not in list(df_cells['cell_id']):
                    
                    #create a dictionary with the results rom this analysis
                    d2 = dict([(kcol,self.info_dict[kcol]) if kcol in self.info_dict.keys() else (kcol,None) for kcol in df_cells.columns  ])
                    d2['cell_id'] = cell_id; d2['cluster_id'] = cluster_ids[cell_ix]
                    d2[k] = data_dict[k][cell_ix]
                    d2['Analysis_'+k] = d2[k]
                    del d2[k]


                    #and append it to the dataframe
                    df_cells = df_cells.append(d2,ignore_index=True)
                    df_cells.set_index('cell_id')
                else:
                    #otherwise add the information in the relevant place
                    #print(k,cell_ix,len(data_dict[k]),len(data_dict['cell_id']))
                    #print(data_dict[k][cell_ix])
                    df_cells.loc[df_cells['cell_id']==cell_id,'Analysis_'+k] = data_dict[k][cell_ix]

        new_df_cells = df_cells

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
        position = np.load(os.path.join(session_path,'OF_positions.npy'))
        #spkT,spkC,single_units,events,lines,aligner = out
        return out + [position]