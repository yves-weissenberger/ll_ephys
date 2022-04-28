#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import os


import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import pandas as pd
import networkx as nx
#optional for nicer plots
import seaborn
seaborn.set(style='ticks',font_scale=1.5)


# # UPDATE THE PATH BELOW TO THE CODE FOLDER

# In[2]:


sys.path.append("/Users/yves/Desktop/retreat_data_dir/code/")
from mecll.task import plot_activity_on_graph


# # Load stuff

# In[3]:


os.listdir


# In[4]:


#These are the times (in units of the behaviour system bin running @1000Hz) at which spikes occurred

selected_session = 0

all_data_dir = '/Users/yves/Desktop/retreat_data_dir/data/'
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


# In[5]:


all_data_folders


# In[ ]:





# In[6]:


def get_task_responses(task_event_df,response_table):
    """ 
    Use the columns of the task_event_df to filter neural activity. 
    In this example build separate firing rate maps for each of the
    tasks, selecting only trials where subjects poked the correct poke.
    
    
    """
    
    
    n_neurons = response_table.shape[1]
    n_ports = 9
    n_tasks = 2
    
    #set variables to nan to not confuse missing data for no responses
    firing_rate_maps = np.zeros([n_neurons,n_ports,n_tasks]) + np.nan
    
    #for each task
    for task in [0,1]:
        
        for port in range(n_ports):  #for each port
            
            #Select indices of pokes where...
            table_index = task_event_df.loc[(task_event_df['task_nr']==task) &  #task_nr was task
                                            (task_event_df['correct']==True) &  #the poke was to the correct port
                                            (task_event_df['port']==port)       #the port poked was port
                                            #add your filter of interest here...
                                           ].index           

            #get the average
            firing_rate_maps[:,int(port),int(task)] = np.nanmean(response_table[table_index],axis=0)
    return firing_rate_maps
                                         


# In[7]:


firing_rate_maps = get_task_responses(task_event_df,response_table)


# In[8]:


#
ccs_across = []
for r1,r2 in zip(firing_rate_maps[:,:,0],firing_rate_maps[:,:,1]):
    ccs_across.append(np.corrcoef(r1,r2)[0,1])
    
ccs_across = np.array(ccs_across)
print(np.nanmean(ccs_across))


# In[9]:


task_event_df['target'].unique()


# In[10]:


eval(task_event_df.loc[task_event_df['task_nr']==0].iloc[0]['current_sequence'])


# In[11]:


#this is the map from the 
task_sequence_0 = eval(task_event_df.loc[task_event_df['task_nr']==0].iloc[0]['current_sequence'])
task_sequence_1 = eval(task_event_df.loc[task_event_df['task_nr']==1].iloc[0]['current_sequence'])

graph_type_0 = task_event_df.loc[task_event_df['task_nr']==0].iloc[0]['graph_type']
graph_type_1 = task_event_df.loc[task_event_df['task_nr']==1].iloc[0]['graph_type']


# In[12]:


resps_state_g1 = firing_rate_maps[:,:,0][:,task_sequence_0]
resps_state_g2 = firing_rate_maps[:,:,1][:,task_sequence_1]
plt.figure(figsize=(14,84))

n_plot = 40
start = 0
ctr = 0
n_units = firing_rate_maps.shape[0]
for i in range(20):

    spks1 = firing_rate_maps[:,:,0][i]
    spks2 = firing_rate_maps[:,:,1][i]
    mx = np.nanmax(np.concatenate([spks1,spks2]))
    mn = 0#np.nanmin(np.concatenate([spks1,spks2]))
    plt.subplot(n_plot,4,4*ctr+1)
    plt.title('Cell:{} |  ccs_across:{:.2f}'.format(i,ccs_across[i]))

    spks = spks1
    plot_activity_on_graph(task_sequence_0,graph_type_0,
                           spks=spks,order='poke',mx=mx,mn=mn)

    plt.subplot(n_plot,4,4*ctr+2)
    spks = spks2
    plot_activity_on_graph(task_sequence_1,graph_type_1,
                           spks=spks,order='poke',mx=mx,mn=mn)

    plt.subplot(n_plot,4,4*ctr+3)
    spks = resps_state_g1[i]
    plot_activity_on_graph(task_sequence_0,graph_type_0,
                           spks=spks,order='state',mx=mx,mn=mn)

    plt.subplot(n_plot,4,4*ctr+4)
    spks = resps_state_g2[i]
    plot_activity_on_graph(np.arange(9).tolist(),graph_type_1,
                           spks=spks,order='state',mx=mx,mn=mn)

    ctr +=1


# In[ ]:




