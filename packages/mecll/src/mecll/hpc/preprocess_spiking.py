import numpy as np

def get_binned_spikes(spks,bin_size=10):
    """ Takes in spike times of units and returns binned spike
        rates (no smoothing) at specified resolution. Bin size
        is in ms
    """

    #30 because sampling rate of the ephys is 
    maxT = (np.nanmax([np.nanmax(i) for i in spks])/30.)/bin_size

    spk_arr = np.zeros([len(spks),int(np.ceil(maxT))])
    for i,u in enumerate(spks):
        spk_arr[i,np.floor(u/30/bin_size).astype("int")[:,0]] = 1
    
    return spk_arr



def get_unit_spike_lists(spkT,spkC,unit_ids=None):
    """ Takes in essentially the kilosort/phy output and returns
        lists with the spike times per unit
        
        Arguments:
        ===================================
        
        spkT: spike times
        
        spkC: cluster membership of each spike
        
        unit_ids: units you want to sort
    """
    
    if unit_ids is None:
        unit_ids = np.arange(len(np.unique(spkC)))
    
    spks = []
    for uid in unit_ids:
        tmp = spkT[np.where(spkC==uid)[0]]
        spks.append(tmp[np.where(np.logical_not(np.isnan(tmp)))[0]])
    return spks