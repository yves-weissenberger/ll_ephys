import numpy as np
from scipy.special import factorial

def _pdf(rate,n):
    return (rate**n)*np.exp(-rate)/factorial(n)

def _logL(rate,n):
    return np.sum(n*np.log(rate) - rate)


def logL(beta,x,y):
    rate = np.exp(np.dot(beta,x))
    LL = np.sum(y*np.log(rate)-rate)
    return -LL

def grad(beta,x,y):
    
    g = np.exp(np.dot(beta,x))
    J = -x.dot(y-g)
    return J

def hess(beta,x,y,regP):

    g = np.exp(np.dot(beta,x))
    return -(np.dot(x*-g,x.T))
