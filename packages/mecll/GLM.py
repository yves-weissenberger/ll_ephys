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

def hess(beta,x,y):

    g = np.exp(np.dot(beta,x))
    return -(np.dot(x*-g,x.T))






def logL_prior(beta,x,y,p_mu,p_cov,p_covInv):
    rate = np.exp(np.dot(beta,x))
    LL = np.sum(y*np.log(rate)-rate)


    LL_prior = (-.5*((aa[1:]-p_mu).dot(p_covInv).dot(aa[1:]-p_mu)) 
                - .5*np.abs(np.linalg.slogdet(p_covInv)[1]) 
                - .5*(p_covInv.shape[0]-1)*np.log((2*np.pi))
                )
    return -(LL+LL_prior)

def grad_prior(beta,x,y,p_mu,p_cov,p_covInv):
    
    g = np.exp(np.dot(beta,x))
    J = -x.dot(y-g)
    J_prior = np.concatenate([[0],np.dot(p_covInv,beta[1:])])
    return (J + J_prior)

def hess_prior(beta,x,y,p_mu,p_cov,p_covInv):

    g = np.exp(np.dot(beta,x))
    H = -(np.dot(x*-g,x.T))

    H[1:,1:] -= p_covInv
    return -(np.dot(x*-g,x.T))
