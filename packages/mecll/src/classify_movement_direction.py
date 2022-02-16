from typing import List

import numpy as np
from sklearn.model_selection import LeavePOut
from sklearn.svm import SVC


def get_svm_fit_within_task(frm_task: np.ndarray, valid_ix: List[int],
                            leave_out: int=1,permutation: bool=False) -> List[float]:
    """Fit a support vector machine that tries to distinguish which direction a subject is
    moving around 

    Args:
        frm_task (np.ndarray): array of neural activity. frm_task.shape = (n_neurons, n_ports, n_directions)
        valid_ix (List[int]): as lines have edges, their activity in 1 direction will be nan. This must be ignored
        leave_out (int, optional): Number of ports to leave out in the classification. Defaults to 1.
        permutation (bool, optional): Set to true to get a null distribution, shuffles the direction labels. Defaults to False.

    Returns:
        List[float]: [description]
    """
    
    lpo = LeavePOut(leave_out)
    
    svm_accuracy= []
    for train_,test_ in lpo.split(valid_ix):

        train_index = valid_ix[train_]
        test_index = valid_ix[test_]
        svm = SVC(kernel='linear')
        #
        X_train = np.hstack([frm_task[:,train_index,0],frm_task[:,train_index,1]])
        X_test =  np.hstack([frm_task[:,test_index,0],frm_task[:,test_index,1]])

        y_train, y_test = [0]*len(train_index) + [1]*len(train_index), [0]*len(test_index) + [1]*len(test_index)
        
        if permutation: y_train = np.random.permutation(y_train)

        svm.fit(X_train.T,y_train)
        svm_accuracy.append(np.mean(svm.predict(X_test.T)==y_test))
        #print(1)

    return svm_accuracy



def get_svm_fit_across_task(frm_task: np.ndarray, frm_task2: np.ndarray ,valid_ix: List[int], permutation: bool=False) -> float:
    

    svm = SVC(kernel='linear')
    #
    X_train = np.hstack([frm_task[:,valid_ix,0],frm_task[:,valid_ix,1]])
    X_test =  np.hstack([frm_task2[:,valid_ix,0],frm_task2[:,valid_ix,1]])

    y_train, y_test = [0]*len(valid_ix) + [1]*len(valid_ix), [0]*len(valid_ix) + [1]*len(valid_ix)
    if permutation: y_train = np.random.permutation(y_train)
    svm.fit(X_train.T,y_train)
    return np.mean(svm.predict(X_test.T)==y_test)


def get_unique_transitions(seq0: np.ndarray,seq1: np.ndarray):
    
    test_seq1 = []
    for ix,i in enumerate(seq1):
        #print(seq0[(np.where(seq0==i)[0]-1)%9],seq0[(np.where(seq0==i)[0])%9],seq1[(ix-1)%9],seq1[(ix)%9])
        if seq0[(np.where(seq0==i)[0]-1)%9]!=seq1[(ix-1)%9]:
            test_seq1.append(1)
        else:
            #print(seq1[ix])
            test_seq1.append(0)

    test_seq1 = np.array(test_seq1)
    return test_seq1

