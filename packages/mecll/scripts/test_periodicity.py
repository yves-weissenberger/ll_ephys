import numpy as np
import os
import sys
import re
import pandas as pd
from classes import all_sessions,single_session_analysis
package_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(package_dir)





if __name__=='__main__':

    all_sess = all_sessions()
    ctr = 0
    for session_id,subject,date,session_path in all_sess:
        print(session_path)
        #try:
        #print(session)
        out = all_sess.load_of_session(session_path)
        spkT,spkC,single_units,events,lines,aligner,position = out
        
        spike_counts = get_total_spike_count(spkC,single_units)
        spike_rates = get_firing_rates(spkT,spkC,single_units)


        sess = single_session_analysis(subject=subject,date=date,session_id=session_id)
        sess.add_data({'firing_rates': spike_rates},single_units)
        sess.add_data({'num_spikes': spike_counts},single_units)