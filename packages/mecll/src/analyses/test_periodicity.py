import numpy as np
import scipy.optimize as op
import numba



@numba.jit(nopython=True)
def np_pearson_cor(x, y):
    """ from 
        https://cancerdatascience.org/blog/posts/pearson-correlation/
    """
    xv = x - np.mean(x)
    yv = y - np.mean(y)
    xvss = np.sum(xv * xv)
    yvss = np.sum(yv * yv)
    result = np.dot(xv.T, yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return result[0][0]# np.maximum(np.minimum(result, 1.0), -1.0)


@numba.jit(nopython=True)
def fit_sin2(x,*params):
    y,t_ = params
    pred = np.cos(x[0]*t_ + x[1])
    cc = np_pearson_cor(pred,y)
    if np.isnan(cc): cc = -100
    return -cc



def _fit_sin2_LEGACY(x,*params):
    t_ = np.linspace(0,2*np.pi,9)
    y = np.array(params)
    y -= np.mean(y)
    y /=np.max(y)
    pred = np.cos(x[0]*t_ + x[1])
    cc = np.corrcoef(pred,y)[0,1]
    if np.isnan(cc): cc = -100
    return -cc





#Add as control not just random shuffles but also shuffles according to the graph of the 
#other task and also physical space
def fit_cos_to_neurons(resps_state_g1,resps_state_g2,NSHUFF=200, params_ranges=[slice(0,4,.5),slice(0,2*np.pi,np.pi/9)]):
    """
    This function uses a brute force approach to fit cosines to the activity of each cell
    """
    resp_sets = [resps_state_g1,resps_state_g2]

    t_ = np.linspace(0,2*np.pi,9)


    n_neurons = resps_state_g1.shape[0]
    print(n_neurons)

    res_sets = []
    ccs_sets = []
    cc_shuff_sets = []
    for neuron_ix in range(n_neurons):
        
        tmp_cc = []
        tmp_res = []
        tmp_cc_shuff = []
        for ix,resp_set in enumerate(resp_sets):    

            sys.stdout.write('\r running cell:{}'.format(neuron_ix))
            sys.stdout.flush()

            spks = resp_set[neuron_ix]
            res1 = op.brute(fit_sin2,params_ranges,args=(spks,t_),finish=None)

            
            cc1 = np.corrcoef(spks,np.cos(res1[0]*t_ + res1[1]))[0,1]

            cc1_shuff = []
            for _ in range(NSHUFF):
                spks_shuff = np.random.permutation(spks)
                res1_shuff = op.brute(fit_sin2,params_ranges,args=(spks_shuff,t_),finish=None)

                cc1_shuff.append(np.corrcoef(spks_shuff,np.cos(res1_shuff[0]*t_ + res1_shuff[1]))[0,1])

            tmp_cc.append(cc1)
            tmp_cc_shuff.append(cc1_shuff)
            tmp_res.append(res1)
        res_sets.append(tmp_res)
        ccs_sets.append(tmp_cc)
        cc_shuff_sets.append(tmp_cc_shuff)
    return np.array(res_sets), np.array(ccs_sets), np.array(cc_shuff_sets)