import unittest
import numpy as np
import scipy.optimize as op
from mecll import dynamics

def get_params(n:int) -> np.ndarray:
        "Helper function to get parameters"
        return np.random.normal(size=int(n*(n-1)/2))

class TestModule(unittest.TestCase):


    def test_unique_transitions(self):

        l1 = np.array([0,1,2,3,4,5,6,7,8])
        l2 = l1
        uniq = dynamics.get_unique_transitions(l1,l2)
        print(uniq)
        self.assertEqual(len(uniq),0)

        l2 = np.array([0,1,2,3,4,5,6,8,7])
        uniq = dynamics.get_unique_transitions(l1,l2)
        print(uniq)
        self.assertEqual(len(uniq),3)


    def test_get_basis_tensor(self):

        bT = dynamics.get_basis_tensor(2)
        for t in bT:
            self.assertEqual(np.sum(t),0)
            self.assertTrue(np.allclose(t,-t.T))

    def test_construct_skew(self):
        n = 2
        bT = dynamics.get_basis_tensor(n)
        params = get_params(n)
        S = dynamics.construct_skew_symmetric_matrix(params,bT)
        self.assertTrue(np.allclose(S,-S.T))
    

    def test_construct_M(self):
        n = 4
        bT = dynamics.get_basis_tensor(n)
        params = get_params(n)
        S = dynamics.construct_skew_symmetric_matrix(params,bT)
        O  = dynamics.construct_M(S,n)
        self.assertTrue(np.allclose(O@O.T,np.eye(n)))

