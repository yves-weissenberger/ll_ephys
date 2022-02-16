import numpy as np



def kf(y,x0,A,C,d,Q,R):
    """ this is the forward kalman filter"""
    n_timepoints = y.shape[1]
    n_dim = Q.shape[0]
    xs_t0 = np.zeros([n_dim,n_timepoints])  # x|y_{1:t-1}
    xs_t1 = np.zeros([n_dim,n_timepoints])  # x|y_{1:t}
    Vs_t0 = np.zeros([n_dim,n_dim,n_timepoints])
    Vs_t1 = np.zeros_like(Vs_t0)
    Vs_t1[:,:,0] = Vs_t0[:,:,0] = np.eye(n_dim)
    #K = np.zeros([n_dim,20])

    xs_t0[:,0] = x0
    xs_t1[:,0] = x0
    Im = np.eye(n_dim)
    for t in range(n_timepoints):
        
        if t>0:
            #prediction step
            xs_t0[:,t] = np.dot(A,xs_t1[:,t-1])  #this is the prior  
            Vs_t0[:,:,t] = np.dot(A,np.dot(Vs_t1[:,:,t-1],A.T)) + Q
        
        #calculate kalman gain
        tmp = np.linalg.inv(np.dot(C,Vs_t0[:,:,t]).dot(C.T) + R)  #1/observation covariance
        K = np.dot(Vs_t0[:,:,t],C.T).dot(tmp)

        #update step
        xs_t1[:,t] = xs_t0[:,t] + np.dot(K,y[:,t] - np.dot(C,xs_t0[:,t]))
        Vs_t1[:,:,t] = Vs_t0[:,:,t] - np.dot(K,C).dot(Vs_t0[:,:,t])
    return xs_t0, xs_t1, Vs_t0, Vs_t1


def ks(A,xs_t0, xs_t1, Vs_t0, Vs_t1):
    """ kalman smoother """
    n_dim,n_timepoints = xs_t1.shape
    
    x_T = np.zeros_like(xs_t1)
    x_T[:,-1] = xs_t1[:,-1]
    v_T = np.zeros_like(Vs_t1)
    v_T[:,:,-1] = Vs_t1[:,:,-1]
    for t_ in reversed(range(n_timepoints-1)):
        J = np.dot(Vs_t1[:,:,t_],A.T).dot(np.linalg.pinv(Vs_t0[:,:,t_+1]))
        #print(xs_t0[:,t_])
        x_T[:,t_] = xs_t1[:,t_] + J.dot(x_T[:,t_+1] - xs_t0[:,t_+1])
        v_T[:,:,t_] = Vs_t1[:,:,t_] + J.dot(v_T[:,:,t_+1] - Vs_t0[:,:,t_+1]).dot(J.T)
    
    return x_T, v_T
    