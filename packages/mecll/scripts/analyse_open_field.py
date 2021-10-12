import numpy as np
import os
import sys
import re
import pandas as pd
from classes import all_sessions
package_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(package_dir)


from open_field_analysis import proc_grids


if __name__=='__main__':

    all_sess = all_sessions()
    for session in all_sess:
        try:
            print(session)
            out = all_sess.load_of_session(session)
        except Exception as e:
            print(e)