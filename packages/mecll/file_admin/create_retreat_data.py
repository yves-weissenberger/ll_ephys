import numpy as np
import os
import re
import sys
import shutil

package_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(package_dir)


from load import load_data
from process_data.proc_beh import get_in_task_pokes,build_poke_df



def build_neuron_response_table(df,spkT,spkC,single_units):
    """ Build a table of the responses of each neuron in a 
        400ms symmetric window around the time of inpokes. 
        Firing rate is in Hz
    """
    n_pokes = len(df)
    n_units = len(single_units)
    

    response_table = np.zeros([n_pokes,n_units])
    for unit_counter,unit_id in enumerate(single_units):
        unit_spikes = spkT[spkC==unit_id]
        for poke_ix,row in df.iterrows():
            #print(row['time'].values)
            response_table[poke_ix,unit_counter] = np.nansum(np.logical_and(unit_spikes>(row['time']-200),
                                                                  unit_spikes<(row['time']+200))
                                                                 )*10/4
    return response_table
            
            


if __name__=='__main__':

    retreat_folders = ['/Users/yves/team_mouse Dropbox/MEC_data/spike_sorted/2021-08-06_39964_ks25',

                       '/Users/yves/team_mouse Dropbox/MEC_data/spike_sorted/2021-07-31_39951_ks25'
                        ]
    retreat_target_dir = '/Users/yves/Desktop/retreat_data_dir/data'

    for folder in retreat_folders:
        suffix = os.path.split(folder)[1]
        target_dir = os.path.join(retreat_target_dir,suffix)
        if not os.path.isdir(target_dir): os.mkdir(target_dir)

        position = np.load(os.path.join(folder,'OF_positions.npy'))
        #clean up position data from when there was a hand or sth
        for i in range(1,position.shape[0]):
            if np.abs(position[i-1,:]-position[i]).sum()>100:
                position[i] = position[i-1]
        
        position_save_path = os.path.join(target_dir,'OF_positions.npy')
        np.save(position_save_path,position)



        #copy over data for the open field analysis
        out = load_data(folder,align_to='OF',camera_frame_rate=30)
        spkT,spkC,single_units,events,lines,aligner = out

        aligned_T = aligner.A_to_B(spkT)
        spks_unit_in_bounds = np.where(np.logical_not(np.isnan(aligned_T)))[0]
        spkC_OF = spkC[spks_unit_in_bounds].astype('uint16')
        spkT_OF = aligned_T[spks_unit_in_bounds].astype('uint64')
        np.save(os.path.join(target_dir,'spkC_OF.npy'),spkC_OF)
        np.save(os.path.join(target_dir,'spkT_OF.npy'),spkT_OF)
        np.save(os.path.join(target_dir,'single_units.npy'),single_units)




        #copy over data for the task stuff
        out = load_data(folder,align_to='task')
        spkT,spkC,single_units,events,lines,aligner = out
        aligned_T = aligner.A_to_B(spkT)
        spks_unit_in_bounds = np.where(np.logical_not(np.isnan(aligned_T)))[0]
        spkC_task = spkC[spks_unit_in_bounds].astype('uint16')
        spkT_task = aligned_T[spks_unit_in_bounds].astype('uint64')
        np.save(os.path.join(target_dir,'spkC_task.npy'),spkC_task)
        np.save(os.path.join(target_dir,'spkT_task.npy'),spkT_task)



        ##build and save table of poke data
        df = build_poke_df(lines,events)
        df.to_csv(os.path.join(target_dir,'task_event_table.csv'))
        response_table = build_neuron_response_table(df,spkT_task,spkC_task,single_units)
        np.save(os.path.join(target_dir,'neuron_response_table.npy'),response_table)



