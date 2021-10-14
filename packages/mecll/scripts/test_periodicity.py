import numpy as np
import os
import sys
import re
import pandas as pd
from classes import all_sessions
package_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(package_dir)



from analysis.test_periodicity import fit_cos_to_neuron


if __name__=='__main__':

    all_sess = all_sessions()
    for session in all_sess:
        try:
            print(session)
            out = all_sess.load_task_session(session)
            spkT,spkC,single_units,events,lines,aligner = out
            fit_cos_to_neurons
        except Exception as e:
            print(e)