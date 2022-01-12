from mecll import GLM
import numpy as np
import unittest
from scipy.optimize import approx_fprime
from scipy.special import factorial

class TestModule(unittest.TestCase):

    def test_logL(self):

        l1 = np.log(GLM._pmf(1.,50))
        l2 = -GLM.logL(np.array([0]),np.array([1]),np.array([50])) - np.log(factorial(50))
        print(l1,l2)
        self.assertTrue(np.allclose(l1,l2))

    def test_grad(self):
        beta = np.random.normal(size=5)
        x = np.random.normal(size=(10,5))
        y = np.random.normal(size=(10))

        g1 = GLM.grad(beta,x,y)
        g2 = approx_fprime(beta,GLM.logL,1e-8,*(x,y))
        print(g1)
        print(g2)
        x = np.allclose(g1,g2,atol=1e-4)
        self.assertTrue(x)

