#from mecll import GLM
import numpy as np
import unittest
#from scipy.optimize import approx_fprime
#from scipy.special import factorial

class TestModule(unittest.TestCase):

    def _sim_data(self):

        """ Lets simulate a low rank matrix """


        self.n_neurons = 200
        self.n_states = 9
        self.n_factors = 3



        #A = np.random.normal(size=(self.n_neurons,self.n_states))
        
        Y_0 = np.random.normal(size=(n_neurons,))

