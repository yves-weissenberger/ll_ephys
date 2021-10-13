import numpy as np
import os
import sys
import re
import pandas as pd
from classes import all_sessions,single_session_analysis
package_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(package_dir)


from open_field_analysis import proc_grids


def get_total_spike_count(spkC,single_units):
    spike_counts = np.zeros(len(single_units))
    for ix,u in enumerate(single_units):
        spike_counts[ix] = np.sum(spkC==u)

    return spike_counts

def get_firing_rates(spkT,spkC,single_units):
    sample_rate = 30000
    spike_counts = get_total_spike_count(spkC,single_units)
    mxT = np.max(spkT)
    
    return sample_rate * spike_counts/mxT

def get_firing_rate_in_epochs(spkC,single_units):


    return None


if __name__=='__main__':

    all_sess = all_sessions()
    ctr = 0
    for session_id,subject,date,session_path in all_sess:
        #try:
        #print(session)
        out = all_sess.load_of_session(session_path)
        spkT,spkC,single_units,events,lines,aligner = out
        
        spike_counts = get_total_spike_count(spkC,single_units)
        spike_rates = get_firing_rates(spkT,spkC,single_units)


        sess = single_session_analysis(subject=subject,date=date,session_id=session_id)
        sess.add_data({'firing_rates': spike_rates},single_units)
        #sess.add_data({'num_spikes': spike_counts},single_units)

            #data_dict = {'spike_count': spike_counts,
            #             'unit_id': single_units,
            #             'subject': subject,
            #             'date': date}

            
            #df = pd.DataFrame.from_dict(data_dict,orient='columns')
            #print(df)
            
            
            
            #

            #for neuron in single_units:
            #get_total_spike



            #print(single_sess.df)


        #except Exception as e:
        #    print(e)


        if ctr>1:
            break
        ctr+=1 