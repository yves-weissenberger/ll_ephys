import numpy as np
import os
import re
import sys
import shutil

package_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(package_dir)


from load import load_data

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
        spkC_OF = spkC[spks_unit_in_bounds]
        spkT_OF = aligned_T[spks_unit_in_bounds].astype('int')
        np.save(os.path.join(target_dir,'spkC_OF.npy'),spkC_OF)
        np.save(os.path.join(target_dir,'spkT_OF.npy'),spkT_OF)
        np.save(os.path.join(target_dir,'single_units.npy'),single_units)




        #copy over data for the task stuff
        out = load_data(folder,align_to='task')
        spkT,spkC,single_units,events,lines,aligner = out
        aligned_T = aligner.A_to_B(spkT)
        spks_unit_in_bounds = np.where(np.logical_not(np.isnan(aligned_T)))[0]
        spkC_task = spkC[spks_unit_in_bounds]
        spkT_task = aligned_T[spks_unit_in_bounds].astype('int')
        np.save(os.path.join(target_dir,'spkC_task.npy'),spkC_task)
        np.save(os.path.join(target_dir,'spkT_task.npy'),spkT_task)

