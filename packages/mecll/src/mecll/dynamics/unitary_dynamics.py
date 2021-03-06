import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from typing import Union, List

Num = Union[int, float]
np_jnp_array = Union[np.ndarray,jnp.ndarray]


def get_unique_transitions(seq0: np.ndarray, seq1: np.ndarray) -> np.ndarray:
    """ Get transitions that are unique in physical space between the two
        tasks
    """
    test_seq1 = []
    for ix,i in enumerate(seq1):
        if seq0[(np.where(seq0==i)[0]-1)%9]!=seq1[(ix-1)%9]:
            test_seq1.append(1)
        else:
            test_seq1.append(0)

    test_seq1 = np.array(test_seq1)
    return np.where(test_seq1)[0]

def predict(T: jnp.ndarray,x: jnp.ndarray, y:jnp.ndarray,n:int,dim:int) -> float:
    """ Predict transitions """
    transition_matrix = jnp.reshape(T,(dim,dim))
    return jnp.sum((jnp.dot(jnp.linalg.matrix_power(transition_matrix,n),x)-y)**2)


def get_basis_tensor(dim: int) -> np.array:
    """ Returns a tensor that forms a basis for skew symmetric matrices. Look up Levi-Civita if curious.
        Use np.einsum('i...,i...',params,basis_tensor) to construct this matrix
    """
    basis_tensor = []
    for i in range(dim):
        for j in range(i+1,dim):
            bi =  np.zeros([dim,dim])
            bi[i,j] = -1
            bi = bi -bi.T
            basis_tensor.append(bi.T)
    basis_tensor = np.array(basis_tensor)
    return basis_tensor



def construct_M(skewM: np_jnp_array, dim: int) -> np_jnp_array:
    """ Perform Caley transform to construct unitary matrix from skew 
        symmetryic matrix
    """
    if type(skewM)==np.ndarray:
        return (np.eye(dim) - skewM)@np.linalg.inv(np.eye(dim)+skewM)
    else:
        return (jnp.eye(dim) - skewM)@jnp.linalg.inv(jnp.eye(dim)+skewM)

def caley_transform(skewM,dim):
    """ Perform Caley transform"""
    return (np.eye(dim) - skewM)@np.linalg.inv(np.eye(dim)+skewM)


def construct_skew_symmetric_matrix(params: np_jnp_array,basis_tensor: np_jnp_array) -> np_jnp_array:
    if type(params)==np.ndarray:
        return np.einsum('i...,i...',params,basis_tensor)
    else:
        return np.einsum('i...,i...',params,basis_tensor)


def predict_all(params: np_jnp_array,x: np_jnp_array, dim: int, basis_tensor: np_jnp_array) -> float:
    """loop over all states"""
    err = 0
    nT = len(x)
    k = 0
    skewM = construct_skew_symmetric_matrix(params,basis_tensor)
    M = construct_M(skewM,dim)
    for start_state in range(nT):
        for pred_state in range(start_state+1,nT-start_state):
            n_fwd = pred_state-start_state 
            err += predict(M,x[start_state],x[pred_state],n_fwd,dim)#*(1/n_fwd)
            k += 1
        #err += predict(M,x[start_state],x[start_state],9,dim)*2
        k += 1
    mse = err/k
    #print(mse)
    return mse


grad_predict_all = grad(predict_all)

def grad_wrapper(params,x,dim,basis_tensor):
    grad = grad_predict_all(jnp.array(params),x,dim,basis_tensor)
    grad = np.array(grad)
    #print(grad)
    return grad


##############


def evaluate_cc(activity: np.ndarray,pred: np.ndarray):
    """ Correlation coefficient between actual and predicted activity, looping over neurons"""
    n_neurons = activity.shape[1]
    true_cc = []

    for i in range(n_neurons):
        if (activity[:,i][1:].shape!=pred[:,i][:-1].shape):
            print(activity[:,i][1:].shape,pred[:,i][:-1].shape,i)
        true_cc.append(np.corrcoef(activity[:,i][1:],pred[:,i][:-1])[0,1])

    return true_cc, np.nanmean(true_cc)

def zscore_population_activity(dat: np.ndarray) -> np.ndarray:
    "This zscores the activity of each cell. Hence each row is activity in one state, each column is a neuron"
    out = (dat - np.mean(dat,axis=0))/np.std(dat,axis=0)
    out = out[:,~np.isnan(np.sum(out,axis=0))]
    return out

def fit_dynamics_model(pca_dim: int, pca_activity1: np.ndarray):
    basis_tensor_inf = get_basis_tensor(pca_dim)
    n_params = int(pca_dim*(pca_dim-1)/2)
    params = np.random.normal(size=(n_params))# + np.eye(dim)
    res = op.minimize(predict_all,
                   params,
                   (pca_activity1,pca_dim,basis_tensor_inf),
                   jac=grad_wrapper,
                   method='BFGS'
                   )
    return res

#def predict_neural_activity_from