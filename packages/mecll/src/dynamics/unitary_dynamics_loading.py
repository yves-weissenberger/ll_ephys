import os
import numpy as np
import pandas as pd

def load_data(selected_session:int, by_dir: bool=False, all_data_dir: str = '/Users/yves/Desktop/retreat_data_dir/data/'):
    
    """ Function to load data structured according to the format
        that the data were delivered to the retreat for
    """

    all_data_folders = sorted([i for i in os.listdir(all_data_dir) if 'ks25' in i])
    root_dir = os.path.join(all_data_dir,all_data_folders[selected_session])
    spkT = np.load(os.path.join(root_dir,'spkT_task.npy'))


    #This array is the same shape as spkT but shows which cluster each of the spikes in spkT belongs to
    spkC = np.load(os.path.join(root_dir,'spkC_task.npy'))

    #This is basically a big table (you can open it in excel) which contains
    #relevant information about each time the animal poked one of the ports
    task_event_df = pd.read_csv(os.path.join(root_dir,'task_event_table.csv'),index_col=0)

    #
    response_table = np.load(os.path.join(root_dir,'neuron_response_table.npy'))
    #alternatively to change the time window


    #not all cluster in spkC correspond to single units. Single units is an array of the clusters that are single units
    single_units = np.load(os.path.join(root_dir,'single_units.npy'))
    
    
    seq0 = np.array(eval(task_event_df.loc[task_event_df['task_nr']==0]['current_sequence'].values[0]))
    seq1 = np.array(eval(task_event_df.loc[task_event_df['task_nr']==1]['current_sequence'].values[0]))
    
    
    graph_type0 = task_event_df.loc[task_event_df['task_nr']==0]['graph_type'].values[0]
    graph_type1 = task_event_df.loc[task_event_df['task_nr']==0]['graph_type'].values[0]
    
    if by_dir:
        firing_rate_maps,frm_stability = get_task_responses_by_direction(task_event_df,response_table)
    else:
        firing_rate_maps,frm_stability = get_task_responses(task_event_df,response_table)
    
    return firing_rate_maps, frm_stability, task_event_df,seq0,seq1,graph_type0,graph_type1


def get_task_responses_by_direction(task_event_df: pd.DataFrame,response_table):
    """ 
    Use the columns of the task_event_df to filter neural activity. 
    In this example build separate firing rate maps for each of the
    tasks, selecting only trials where subjects poked the correct poke.
    
    
    """
    
    
    n_neurons = response_table.shape[1]
    n_ports = 9
    n_tasks = 2
    n_direction = 2
    
    #set variables to nan to not confuse missing data for no responses
    firing_rate_maps = np.zeros([n_neurons,n_ports,n_tasks,n_direction]) + np.nan
    #for each task
    for task in [0,1]:
        
        for port in range(n_ports):  #for each port
            
            for dix,direction in enumerate(np.unique(task_event_df['direction'].values)):

                #Select indices of pokes where...
                table_index = task_event_df.loc[(task_event_df['task_nr']==task) &  #task_nr was task
                                                (task_event_df['correct']==True) &  #the poke was to the correct port
                                                (task_event_df['port']==port) &       #the port poked was port
                                                (task_event_df['direction']==direction)
                                               ].index           
                #print(len(table_index))
                #get the average
                firing_rate_maps[:,int(port),int(task),dix] = np.nanmean(response_table[table_index],axis=0)
    print('warning not calculating stability just setting to 1')
    return firing_rate_maps, np.ones(n_neurons)
                                         
def get_task_responses(task_event_df: pd.DataFrame,response_table):
    """ 
    Use the columns of the task_event_df to filter neural activity. 
    In this example build separate firing rate maps for each of the
    tasks, selecting only trials where subjects poked the correct poke.
    
    
    """
    
    
    n_neurons = response_table.shape[1]
    n_ports = 9
    n_tasks = 2
    n_direction = 2
    
    #set variables to nan to not confuse missing data for no responses
    firing_rate_maps = np.zeros([n_neurons,n_ports,n_tasks]) + np.nan
    firing_rate_maps1 = np.zeros([n_neurons,n_ports,n_tasks]) + np.nan
    firing_rate_maps2 = np.zeros([n_neurons,n_ports,n_tasks]) + np.nan

    #for each task
    for task in [0,1]:
        
        for port in range(n_ports):  #for each port
            
            #for dix,direction in enumerate(np.unique(task_event_df['direction'].values)):

                #Select indices of pokes where...
            table_index = task_event_df.loc[(task_event_df['task_nr']==task) &  #task_nr was task
                                            (task_event_df['correct']==True) &  #the poke was to the correct port
                                            (task_event_df['port']==port)        #the port poked was port
                                            #(task_event_df['direction']==direction)
                                           ].index           
            #print(len(table_index))
            #get the average
            
            half = len(table_index) //2
            firing_rate_maps[:,int(port),int(task)] = np.nanmean(response_table[table_index],axis=0)
            firing_rate_maps1[:,int(port),int(task)] = np.nanmean(response_table[table_index[:half]],axis=0)
            firing_rate_maps2[:,int(port),int(task)] = np.nanmean(response_table[table_index[half:]],axis=0)
        
    ccs = []
    for n in range(n_neurons):
        cc = np.corrcoef(firing_rate_maps1[n].flatten(),firing_rate_maps2[n].flatten())[0,1]
        if np.isnan(cc): cc=-100
        ccs.append(cc)
    return firing_rate_maps,ccs