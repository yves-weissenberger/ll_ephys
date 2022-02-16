import numpy as np

def get_glm_correlations():
    """ Fit a poisson GLM to binned spike trains to estimate correlations
        between cells, as described in Gardner et al (2019). The argument for doing this
        over direct corrcoef calculation is to factor out coupling to population rate
        https://www.nature.com/articles/s41593-019-0360-0
     """
    
    return None


def get_direct_correlations():
    """ Directly calculate the correlation coefficients between binned spike trains.
    """
    return None