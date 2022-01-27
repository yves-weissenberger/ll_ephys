from typing import List, Union

from ..rsync import *

import numpy as np


def check_in_range(task_time_ranges: List[List[int]],t: Union[float,int]) -> bool:
    """Find out if one of the times is in the range of a given task

    Args:
        task_time_ranges (List[List[int]]): Each of the smaller lists contains the start
                                            and end times of a block.
        time (Union[float,int]): Timestamp that we are looking at the range for

    Returns:
        bool: [description]
    """
    in_range = False
    for r in task_time_ranges:
        if (r[0] <= t <= r[1]):
            in_range = True

    return in_range


def get_transitions_count_dict(ddp: dict, task_times: List[List[int]], task_nr: int = 0) -> dict:
    
    transitions = [{},{}]
    for i in ddp:
        
        t_ = i[2] * 1000
        if check_in_range(task_times[task_nr],t_):
            tmp = (i[0],i[1])
            if tmp in transitions[0].keys():
                transitions[0][tmp] += 1
            else:
                transitions[0][tmp] = 1
        else:
            tmp = (i[0],i[1])
            if tmp in transitions[1].keys():
                transitions[1][tmp] += 1
            else:
                transitions[1][tmp] = 1

    return transitions


def get_seq_from_transitions(transitions_port: dict) -> List:
    """Get sequence of ports (matching format of MEC data) that are traversed
       in a given task


    Args:
        transitions_port (dict): dictionary containing as keys tuples of the transitions and
                                 whose values are the number of occurrences of each transition


    Raises:
        Exception: [description]

    Returns:
        List: Sequence
    """ 
    transition_counts = list(transitions_port.values())
    diff = max(transition_counts) - min(transition_counts)
    if diff>8:
        #raise Exception('Transitions are not clean')
        print("WARNING THERE MAY BE FAULTY TRANSITIONS IN HERE, CHECK THE TRANSITIONS DICT")

    keys = list(transitions_port.keys())
    seq = [keys[0][0],keys[0][1]]
    for i in keys[1:]:
        if i[1] not in seq:
            seq.append(i[1])
    return seq

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

