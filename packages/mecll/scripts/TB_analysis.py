import numpy as np
from 

""" 

In this analysis, we just
In SVD, the columns of U are orthonormal basis vectors. So each column contains the correlations. 

Thus 
"""



def collect_transitions():
    


if __name__=="__main__":

    n_states = 9
    for session in sessions:
        mean_activity = get_mean_activity()  #get the mean activity during each poke

        collect_single_trials = n_states = 9
        U,S,Vt = np.linalg.SVD(mean_activity)  

        #take the first n_states columns of U
        U_short = U[:,:n_states]

        trial_matrix = get_single_trial_matrix()  #T.shape = (n_cells x n_timepoints)
        
        #now get the expression of each of the 9 singular vectors in the population activity???
        #each row of y is a mixture of the rows of trial_matrix. This is just horizontally stacked trials.
        #each column of y is a mixture of the columns of U.T. What are the columns of U.T
        y = np.dot(U_short.T,trial_matrix)








