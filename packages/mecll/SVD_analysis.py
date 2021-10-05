import numpy as np
from scipy.ndimage import gaussian_filter1d

def variance_explained_U(store_ref,store_cv,store_alt):
    """ 
    Arguments:
    =====================
    store_ref: np.array
               this is what you do SVD on (n_neurons x n_timepoints)
    store_cv:  np.array
               this is (n_neurons x n_timepoints) array from the same task
    store_alt: np.array
               this is (n_neurons x n_timepoints) array from the other task
    """
    U,S,V = np.linalg.svd(store_ref)
    
    #calculate explained variance explained by U
    ev_cv = np.sum(U.T.dot(store_cv)**2,axis=1)
    norm_ev_cv = evals_cv/np.sum(ev_cv)

    #calculate explained variance explained by U
    ev_alt = np.sum(U.T.dot(store_alt)**2,axis=1)
    norm_ev_alt = evals_alt/np.sum(ev_alt)
        
    return norm_ev_cv, norm_ev_alt


def variance_explained_both(store_ref,store_cv,store_alt):

    U,S,V = np.linalg.svd(store_ref)

    
    ev_cv = U.T.dot(store_cv).dot(V.T).diagonal()**2
    norm_ev_cv /= np.sum(ev_cv)
    
    
    ev_alt = U.T.dot(store_alt).dot(V.T).diagonal()**2
    norm_ev_alt /= np.sum(ev_alt)

    
    return norm_ev_cv, norm_ev_alt


def variance_explained_V(store_ref,store_cv,store_alt):
    """ Not 100% sure about this one but think its
        correct
    """
    U,S,V = np.linalg.svd(store_ref)
    
    #calculate explained variance explained by U
    ev_cv = np.sum(np.dot(store_cv,V.T)**2,axis=1)
    norm_ev_cv = evals_cv/np.sum(ev_cv)

    #calculate explained variance explained by U
    ev_alt = np.sum(np.dot(store_alt,V.T)**2,axis=1)
    norm_ev_alt = evals_alt/np.sum(ev_alt)
        
    return norm_ev_cv, norm_ev_alt


def get_mean_activity_matrix(single_trial_resps,order=None,half=0,downsample_factor=150,smoothing_factor=10,flip=False):
    
    """ This builds a matrix by concatenating mean population activity centered on each
        of the different pokes that describes how things evolve over time.
    """
    
    if order is None:
        order = np.arange(9)

    n_units = len(single_trial_resps)
    store_g1 = []
    
    downsample_axis = int(np.floor(len(single_trial_resps[4][3][0])/downsample_factor))
    #print(downsample_axis)
    for neuron_ix in range(n_units):
        tmp = [] 
        for poke_nr in order:
            activity = np.array(single_trial_resps[neuron_ix][poke_nr])
            #print(np.sum(activity))
            if half==0:
                mean_activity = np.mean(activity,axis=0)
            elif half==1:
                mean_activity = np.mean(activity[:int(len(activity)/2)],axis=0)
            elif half==2:
                mean_activity = np.mean(activity[int(len(activity)/2):],axis=0)
                
            try:
                mean_downsampled = mean_activity.reshape(-1,downsample_axis).mean(axis=1)
                #print(mean_downsampled.shape)
                mean_downsampled_smoothed = gaussian_filter1d(mean_downsampled,smoothing_factor)
            except ValueError:
                mean_downsampled_smoothed = np.zeros(downsample_factor) + np.nan
            
            if flip: mean_downsampled_smoothed = np.flipud(mean_downsampled_smoothed)
            tmp.append(mean_downsampled_smoothed)
            
        #print(np.array(tmp).shape)
        store_g1.append(np.hstack(tmp))
    return np.array(store_g1)