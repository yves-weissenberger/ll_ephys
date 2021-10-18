import numpy as np
import os
import sys
import re
import copy as cp
import pandas as pd
from classes import all_sessions,single_session_analysis
package_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(package_dir)


from open_field_analysis import proc_grids


if __name__=='__main__':

    all_sess = all_sessions()
    ctr = 0
    for session_id,subject,date,session_path in all_sess:
        #try:
            #print(session)
            out = all_sess.load_of_session(session_path)
            spkT,spkC,single_units,events,lines,aligner,position = out

            #first get spatial firing map
            params = {'smooth':30,'n_bins':45,'min_dwell':5,'min_dwell_distance_pixels':20,'dt_position_ms':1}
            param_string = "_used_params_" + repr(params)
            spatial_rate_maps = []
            single_units = single_units
            n_units = len(single_units)
            for unit_loop_counter,unit_nr in enumerate(single_units):
                sys.stdout.write('\rprocessing cell{}/{}'.format(1+unit_loop_counter,n_units))
                sys.stdout.flush()
                srm = proc_grids.get_rate_map_nolan(unit_nr,aligner,spkC,spkT,position,**params)
                spatial_rate_maps.append(srm)
            #then process this data
            sess = single_session_analysis(subject=subject,date=date,session_id=session_id)
            print('\n')
            sess.add_data({'spatial_rate_maps'+param_string: cp.deepcopy(spatial_rate_maps)},single_units,
                         save_format={'spatial_rate_maps'+param_string: '.npy'})



            res = proc_grids.process_grid_data(spatial_rate_maps)
            print('\n')
            sess.add_data({'rate_map_autocorrelogram'+param_string:res['rate_map_autocorrelogram']},single_units,
                          save_format={'rate_map_autocorrelogram'+param_string: '.npy'})

            sess.add_data({'grid_spacing': res['grid_spacing']},single_units)
            sess.add_data({'grid_score': res['grid_score']},single_units)
            sess.add_data({'field_size': res['field_size']},single_units)

        #except Exception as e:
        #    print(e)

            #ctr += 1
            #if ctr>1:
            #    break
            #ctr+=1 