import numpy as np
from scipy.stats import norm
from .unitary_dynamics import caley_transform, get_basis_tensor


poke_pos = np.array([1,-1])*np.array([ [149,0],
                                 [68,19],[231,19],
                               [0,62],[149,62],[298,62],
                                 [68,105],[231,105],
                                      [149,124]])

def simulate_dynamics(dim=5,nT=9,frac_noise_dynamics=.7):
    """ Get latent states simulated as progressing with unitary dynamics
    
    Arguments:
    ================================

    dim:                    int
                            dimensionality of the states state
    
    nT:                     int
                            number of ports


    frac_noise_dynamics:    int
                            add gaussian noise to the dynamics
    
    """
    
    x0 = np.random.normal(size=(dim,1))
    a_params = np.random.normal(size=int(dim*(dim-1)/2))

    basis_tensor = get_basis_tensor(dim)
    skewM = np.einsum('i...,i...',a_params,basis_tensor)
    A = caley_transform(skewM,dim)


    x0_2 = np.random.normal(size=(dim,1))
    assert(np.logical_not(np.allclose(x0,x0_2)))


    x = np.array([np.linalg.matrix_power(A,i)@x0 for i in range(nT)])
    x_2 = np.array([np.linalg.matrix_power(A,i)@x0_2 for i in range(nT)])

    x = frac_noise_dynamics*x + (1-frac_noise_dynamics)*np.random.normal(size=x.shape)
    x_2 = frac_noise_dynamics*x_2 + (1-frac_noise_dynamics)*np.random.normal(size=x.shape)

    
    return x,x_2,A


def get_spatial_tuning(poke_pos,peak_loc,width):
    """"""
    #for i in range(9):
    distance_matrix = np.abs((poke_pos[peak_loc]-poke_pos)**2).sum(axis=1)
    fr = norm(scale=width).pdf(distance_matrix)
    return fr
        



def simulate_activity(n_neurons,x,x_2,frac_space=None,noise_scale=.1):
    """ Simulate neural activity"""

    dim = x.shape[1]
    act_matrix = []
    act_matrix2 = []
    space_order2 = np.random.permutation(np.arange(9))
    space_order2_inv = [list(space_order2).index(i) for i in range(9)]

    task_order2 = np.random.permutation(np.arange(9))
    task_order2_inv = [list(task_order2).index(i) for i in range(9)]
    for neuron_ix in range(n_neurons):
        
        if frac_space is None:
            frac_space = np.random.uniform(0,1)
        #else:
        #    frac_space = 
        peak_space = np.random.randint(0,9)
        std_space = np.random.randint(1000,12000)
        
        lamda1 = get_spatial_tuning(poke_pos,peak_space,std_space)
        lamda1 = lamda1/np.max(lamda1)
        
        
        neuron_state_couple = np.random.normal(size=(dim))
        
        lamda2 = np.squeeze(x)@neuron_state_couple
        lamda2_2 = np.squeeze(x_2)@neuron_state_couple

        
        activity = frac_space*lamda1 + (1-frac_space)*lamda2 + noise_scale*np.random.normal(scale=1.,size=9)
        act_matrix.append(activity.copy())
        
        activity2 = frac_space*lamda1+ (1-frac_space)*lamda2_2[task_order2]       + noise_scale*np.random.normal(scale=1.,size=9)
        act_matrix2.append(activity2.copy())
    act_matrix = np.array(act_matrix).T
    act_matrix2 = np.array(act_matrix2).T

    return act_matrix, act_matrix2, space_order2, space_order2_inv, task_order2, task_order2_inv