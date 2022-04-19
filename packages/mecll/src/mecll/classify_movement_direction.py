from typing import List, Set, Tuple, Optional

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



def get_svm_fit_across_task(frm_task: np.ndarray, frm_task2: np.ndarray,
                            valid_ix: np.ndarray, overlapping_ports: List[int],
                            do_shuffle: bool=False,):

    
    svm = SVC(kernel='linear')
    #
    valid_ports_to_test_on = np.array([i for i in valid_ix if i not in overlapping_ports])
    #print(valid_ports_to_test_on)
    X_train = np.hstack([frm_task[:,valid_ix,0],frm_task[:,valid_ix,1]])
    y_train = [0]*len(valid_ix) + [1]*len(valid_ix)
    
    
    X_test =  np.hstack([frm_task2[:,valid_ports_to_test_on,0],frm_task2[:,valid_ports_to_test_on,1]])
    y_test = [0]*len(valid_ports_to_test_on) + [1]*len(valid_ports_to_test_on)
    
    if do_shuffle: y_train = np.random.permutation(y_train)
    svm.fit(X_train.T,y_train)
    # print("Train set accuracy={:.2f}".format(np.mean(svm.predict(X_train.T)==y_train)))
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

# -----------------------------------------------------------------------------------------------

def get_train_test_transitions_unbalanced(seq0: np.ndarray, seq1: np.ndarray,
                                          graph_type0: str, graph_type1: str
                                         ) -> Set[Tuple[int,int,str]]:
    """ The purpose of this"""
    
    
    
    seq0_transitions = get_transitions(seq0,
                                       graph_type=graph_type0,
                                       exclude=[])
    
    seq1_transitions = get_transitions(seq1,
                                       graph_type=graph_type1,
                                       exclude=seq0_transitions,
                                       add_direction=True
                                      )
        
    return seq0_transitions, seq1_transitions


def get_transitions(seq: np.ndarray, graph_type:str, exclude=[], add_direction=False) -> Set[Tuple[int,int,Optional[str]]]:
    
    transitions = set()
    for i in range(len(seq)):
        next_i = (i+1)
        prev_i = (i-1)
        
        if next_i<=8 or graph_type=='loop':
            transition = (seq[i], seq[next_i%9])
            if transition not in exclude:
                if add_direction:
                    transition = (seq[i], seq[next_i%9],'back')
                transitions.add(transition)
        
        if prev_i>=0 or graph_type=='loop':
            transition = (seq[i], seq[prev_i%9])
            if transition not in exclude:
                if add_direction:
                    transition = (seq[i], seq[prev_i%9],'fwd')

                transitions.add(transition)
    return transitions


def get_svm_fit_across_task_unbalanced(frm_task: np.ndarray, frm_task2: np.ndarray,
                            valid_ix: np.ndarray, overlapping_ports: List[int],
                            do_shuffle: bool=False,):
    
    svm = SVC(kernel='linear')

    valid_ports_to_test_on = np.array([i for i in valid_ix if i not in overlapping_ports])


    X_train = np.hstack([frm_task[:,valid_ix,0],frm_task[:,valid_ix,1]])
    y_train = [0]*len(valid_ix) + [1]*len(valid_ix)
    
    
    X_test =  np.hstack([frm_task2[:,valid_ports_to_test_on,0],frm_task2[:,valid_ports_to_test_on,1]])
    y_test = [0]*len(valid_ports_to_test_on) + [1]*len(valid_ports_to_test_on)
    
    if do_shuffle: y_train = np.random.permutation(y_train)
    svm.fit(X_train.T,y_train)
    print("Train set accuracy={:.2f}".format(np.mean(svm.predict(X_train.T)==y_train)))
    return np.mean(svm.predict(X_test.T)==y_test)


def get_features_and_targets(task_frm: np.ndarray,
                             seq: List[int],
                             include: Set=set()
                             ):
    
    rev_dir_map = {0:'fwd', 1: 'back'}
    X_train = []
    y_train = []
    for direction, task_data in enumerate(task_frm.swapaxes(0,2)):

        for port_num,port_dat in enumerate(task_data):

            if np.isfinite(np.sum(port_dat)):
                #print( (seq[port_num], seq[port_num+(2*direction)-1], rev_dir_map[direction]) )
                if (
                    (seq[port_num], seq[(port_num+(2*direction)-1)%9], rev_dir_map[direction]) in include
                    or len(include)==0
                   ):
                    #print( (seq[port_num], seq[(port_num+(2*direction)-1)%9], rev_dir_map[direction]) )
                    X_train.append(port_dat)
                    y_train.append(direction)

    return np.array(X_train), np.array(y_train)