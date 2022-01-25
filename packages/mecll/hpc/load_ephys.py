# standard library
import os
import re
from typing import List, Tuple

# numpy 
import numpy as np

def load_ephys_data(root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,float]:
    """Load ephys data given root directory containing only 1 set of ephys files

    Args:
        ROOT (str): root

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,float]: [spike times etc]
    """
    paths = find_paths_ephys(root)
    spkT_u, spkC, rsync_times_spk, clust_qual,start = load_ephys_paths(paths)
    return spkT_u, spkC, rsync_times_spk, clust_qual, start

def find_paths_ephys(ROOT: str) -> List[str]:
    """Given root folder containing all relevant spike files,
       find paths of relevant files

    Args:
        ROOT (str): Root directory containing one set of spike files

    Returns:
        List[str]: list of full filepaths
    """
    paths = []
    for a,b,c in os.walk(ROOT):
        for c_ in c:
            pth_ = os.path.join(a,c_)
            if ('spike_times' in pth_ or
                'spike_clusters' in pth_ or
                'timestamps.npy' in pth_ or
                'cluster_KSLabel' in pth_ or
                'sync_messages' in pth_):
                paths.append(pth_)
        ##print(a,b,c)
    return paths



def load_ephys_paths(paths: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,float]:
    
    start = None
    for pth in paths:
        if 'spike_times' in pth:
            spkT_u = np.load(pth)
        elif 'spike_clusters' in pth:
            spkC = np.load(pth)
        elif 'timestamps.npy' in pth:
            print(pth)
            rsync_times_spk = np.load(pth)
        elif 'KSLabel' in pth:
            clust_qual = np.array(pd.read_table(pth).KSLabel.values)
        elif 'sync_messages' in pth:
            ls = open(pth,'r').readlines()
            for l_ in ls:
                if 'start time' in l_:
                    start = float(re.findall(r'start time: ([0-9]*)@30000Hz',l_)[0])
                    
    if start is None:
        print("WARNING! Did not find sync_messages.txt assuming start=0 ")
        start = 0
    return spkT_u, spkC, rsync_times_spk, clust_qual, start