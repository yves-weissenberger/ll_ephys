import numpy as np


def nan_corrcoef(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Correlation coefficient of 2 arrays with nans in them

    Args:
        x (np.ndarray): Flat 1d array
        y (np.ndarray): Flat 1d array

    Returns:
        np.ndarray: [description]
    """
    cc = np.ma.corrcoef(np.ma.masked_where(np.isnan(x),x),
                np.ma.masked_where(np.isnan(y),y)
               )
    return cc.data

def get_rotation_correlations(task1: np.ndarray, task2_ref: np.ndarray) -> np.ndarray:

    """Get maximal correlation rotating acros the two lines while moving around
       in one direction.

    Args:
        task1 (np.ndarray): neural activity in task1, from one direction
        task2 (np.ndarray): neural activity in task 2, from one direction, this array will be rotated
        graph_type (str): line | loop

    Returns:
        np.ndarray: Correlations at each rotation and with each possible flip
    """
    
    ccs = []

    # if there are lines we can just mask the invalid ones
    task1 = np.ma.masked_invalid(task1)
    task2_ref = np.ma.masked_invalid(task2_ref)
    
    for roll_amount in range(9):

        for flip in [True,False]:
            task2 = task2_ref.copy()

            task2 = np.roll(task2,roll_amount,axis=1)
            if flip: task2 = np.flipud(task2)
            cc_i = nan_corrcoef(task1.flatten(),task2.flatten())
            ccs.append(cc_i)
    
    return np.array(ccs)