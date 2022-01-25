from ..rsync import *

import numpy as np

def align_spikes_and_behaviour(pulse_times_behaviour: np.ndarray, pulse_times_ephys: np.ndarray,offset:int) -> Rsync_aligner:
    """Align pulse times from behaivoural and ephys recordings

    Args:
        pulse_times_behaviour (np.ndarray): 
        pulse_times_ephys (np.ndarray): 
        offset (int): if ephys recording started after window had been showing traces for a while this is required

    Returns:
        Rsync_aligner: an alignment object
    """

    return

