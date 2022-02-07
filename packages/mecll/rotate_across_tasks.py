from typing import List, Tuple

import numpy as np



def preprocess_neural_activity(firing_rate_maps: np.ndarray, seq0: np.ndarray, seq1: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """Take directional firing rate maps, split them by task and remove spatial responses

    Args:
        firing_rate_maps (np.ndarray): array of mean responses at each port, size is (n_neurons x n_ports x n_tasks x n_directions)
        seq0 (np.ndarray): The sequence of ports traversed in task 1
        seq1 (np.ndarray): The sequence of ports traversed in task 2

    Returns:
        Tuple[np.ndarray,np.ndarray]: mean neural activity in each task with spatial component subtracted
    """

    # calculate 'spatial' responses
    spatial_map  = np.nanmean(firing_rate_maps,axis=(2,3))

    # this array is (cell,port,task,direction)
    mds_frm = firing_rate_maps.copy()


    # subtract spatial responses from task
    mds_frm = firing_rate_maps - spatial_map[:,:,np.newaxis,np.newaxis]

    task1 = mds_frm[:,:,0]
    
    #order task 1 by sequence 1
    task1 = task1[:,seq0]


    task2 = mds_frm[:,:,1]
    task2 = task2[:,seq1]
    
    return task1, task2

def nan_corrcoef(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Correlation coefficient of 2 arrays with nans in them

    Args:
        x (np.ndarray): Flat 1d array
        y (np.ndarray): Flat 1d array

    Returns:
        np.ndarray: [description]
    """
    cc = np.ma.corrcoef(np.ma.masked_where(np.isnan(x), x),
                        np.ma.masked_where(np.isnan(y), y) )
    return cc.data

def get_rotation_correlations(task1: np.ndarray, 
                              task2_ref: np.ndarray, 
                              by_neuron: bool=True) -> np.ndarray:

    """Get maximal correlation rotating acros the two lines while moving around
       in one direction.

    Args:
        task1 (np.ndarray): neural activity in task1, from one direction
        task2 (np.ndarray): neural activity in task 2, from one direction, this array will be rotated
        graph_type (str):   line | loop

    Returns:
        np.ndarray: Correlations at each rotation and with each possible flip
    """
    
    ccs = []

    # if there are lines we can just mask the invalid ones
    task1 = np.ma.masked_invalid(task1)
    task2_ref = np.ma.masked_invalid(task2_ref)
    
    for roll_amount in range(9):
        # stores correlation coefficients averaged over neurons
        # for one roll of the dice
        roll_cc_store = []
        for flip in [True,False]:
            task2 = task2_ref.copy()

            task2 = np.roll(task2,roll_amount,axis=1)
            if flip: task2 = np.flipud(task2)
            
            single_neuron_corr = []
            if by_neuron:
                for n_task1, n_task2 in zip(task1,task2):
                    single_neuron_corr.append(nan_corrcoef(n_task1,n_task2)[0,1])
                    cc_i = np.mean(single_neuron_corr)
            else:
                cc_i = nan_corrcoef(task1.flatten(),task2.flatten())[0,1]
            roll_cc_store.append(cc_i)
        ccs.append(roll_cc_store)
    
    return ccs