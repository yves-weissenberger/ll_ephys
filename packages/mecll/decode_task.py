import sys
from typing import List, Tuple

import numpy as np
from sklearn import svm
from sklearn.svm import SVC

from .load import load_data

def get_valid_pokes_index_by_direction(frm_task: np.ndarray
                                       ) -> Tuple[np.ndarray,np.ndarray]:

    #
    task_dir1 = np.where(np.logical_not(np.isnan(np.sum(frm_task,axis=0)[:,0])))[0]
    task_dir2 = np.where(np.logical_not(np.isnan(np.sum(frm_task,axis=0)[:,1])))[0]

    
    return task_dir1, task_dir2


def create_task_train_test(task_X: np.ndarray,
                           train_fraction: float=0.8, rnd_seed=None
                           )-> Tuple[np.ndarray, np.ndarray]:
    
    
    n_ports1 = task_X.shape[1]
        
    np.random.seed(rnd_seed)
    
    perm_task1 = np.random.permutation(np.arange(n_ports1))
    
    n_train1 = int(np.floor(train_fraction*n_ports1))
    
    train_X, test_X = task_X[:,perm_task1[:n_train1]], task_X[:,perm_task1[n_train1:]]
    
    # check that order correctly determined
    assert(np.allclose(train_X[:,0],task_X[:,perm_task1[0]]))
    
    # and that no funky duplication is going on
    assert(np.sum(train_X,axis=0).shape[0]==len(np.unique(np.sum(train_X,axis=0))))
    
    return train_X, test_X
    

def get_task_X(frm_task: np.ndarray, task_dir1: np.ndarray,
               task_dir2: np.ndarray) -> np.ndarray:
    
    task_X = np.hstack([frm_task[:,task_dir1,0],
                        frm_task[:,task_dir2,1]  
    ])


    return task_X


def build_features_and_labels(X1: np.ndarray, X2: np.ndarray,
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """X1.shape=(n_features, n_samples)"""
    X = np.hstack([X1,X2])
    y = np.concatenate([np.ones(X1.shape[1]),
                              np.zeros([X2.shape[1]])
                             ])
    return X, y

def create_task_train_test(task_X: np.ndarray,
                           train_fraction: float=0.8, rnd_seed=None
                           )-> Tuple[np.ndarray, np.ndarray]:
    
    
    n_ports1 = task_X.shape[1]
        
    np.random.seed(rnd_seed)
    
    perm_task1 = np.random.permutation(np.arange(n_ports1))
    
    n_train1 = int(np.floor(train_fraction*n_ports1))
    
    train_X, test_X = task_X[:,perm_task1[:n_train1]], task_X[:,perm_task1[n_train1:]]
    
    return train_X, test_X
    


def fit_svm_classify_task(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    
    """

    svm = SVC(kernel='linear')
    svm.fit(X_train.T,y_train)
    return np.mean(svm.predict(X_test.T)==y_test)


def main(firing_rate_maps: np.ndarray,suppress_warnings: bool=True) -> List[float]:

    # can stuggle with division by 0 errors
    if suppress_warnings:
        import warnings
        warnings.filterwarnings('ignore')
    
    svm_fits = []
    for session_ix in range(8):
        sys.stdout.write("\rsession_index:{}".format(session_ix))
        sys.stdout.flush()
        tmp_mu = []
        for _ in range(50):

            firing_rate_maps,_,_ = load_data(session_ix)
            frm_task1 = firing_rate_maps[:,:,1]
            frm_task2 = firing_rate_maps[:,:,0]

            task1_dir1, task1_dir2 = get_valid_pokes_index_by_direction(frm_task1)
            task2_dir1, task2_dir2 = get_valid_pokes_index_by_direction(frm_task2)


            task1_X = get_task_X(frm_task1, task1_dir1, task1_dir2)
            task2_X = get_task_X(frm_task2, task2_dir1, task2_dir2)


            train1_X, test1_X = create_task_train_test(task1_X,rnd_seed=None)
            train2_X, test2_X = create_task_train_test(task2_X,rnd_seed=None)

            X_train, y_train = build_features_and_labels(train1_X,train2_X)
            X_test, y_test = build_features_and_labels(test1_X,test2_X)


            tmp_mu.append(fit_svm_classify_task(X_train,y_train,X_test,y_test))

        svm_fits.append(tmp_mu.copy())
    return svm_fits