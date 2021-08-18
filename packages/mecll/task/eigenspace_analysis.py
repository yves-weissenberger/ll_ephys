import numpy as np


def get_square_evecs():
    spatial_transitions_square =  [[1,2],
                             [0,3,4],
                             [0,4,5],
                             [1,6],
                             [1,2,6,7],
                             [2,7],
                             [3,4,8],
                             [4,5,8],
                             [6,7]]

    A = np.zeros([9,9])
    for i,entries in enumerate(spatial_transitions_square):
        A[i,entries] = 1
        
    D = np.sum(A,axis=0)
    L = D - A
    evals,evecs = np.linalg.eig(L)
    evals = evals.real
    evecs = np.abs(evecs)
    return evals,evecs, (A,D,L)


def get_task_evecs(graph_type='line'):
    pass
