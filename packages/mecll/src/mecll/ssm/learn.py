import numpy as np

""" 
we are looking at an SSM of the form

y ~ N(C*x_t + d, R)
x_t ~ N(A*x_(t-1) + u,Q)


see https://cran.r-project.org/web/packages/MARSS/vignettes/EMDerivation.pdf 
for a detailed derivation


If want to get fancy, can do the second part of the optmization separately and do
a laplace approximation
"""


def get_A(x10,x11):
    
    
    """ 
    Arguments:

    x10:    np.array
            the posterior covariance between x_t
    """



    return None

def get_R():

    return None

def get_C():
    return None

def get_Q():
    return None