import numpy as np
import scipy.ndimage as ndi
import sys

def filter_gaussian(arr, sigma):
    """
        This just smoothes spatial firing rate maps but deals with bins
        that were not visited.
        https://stackoverflow.com/a/36307291/7128154
    """
    gauss = arr.copy()
    gauss[np.isnan(gauss)] = 0
    gauss = ndi.gaussian_filter(
            gauss, sigma=sigma, mode='constant', cval=0)

    norm = np.ones(shape=arr.shape)
    norm[np.isnan(arr)] = 0
    norm = ndi.gaussian_filter(
            norm, sigma=sigma, mode='constant', cval=0)

    # avoid RuntimeWarning: invalid value encountered in true_divide
    norm = np.where(norm==0, 1, norm)
    gauss = gauss/norm
    gauss[np.isnan(arr)] = np.nan
    return gauss


def get_occupancy_map(positions,ixs,bins):

    occupancy_map1,_,_ = np.histogram2d(position[ixs,0],
                                        position[ixs,1],bins=bins)
        
    return occupancy_map


def get_split_half_correlations(spkC,spkT,single_units,position,speed,SPEED_THRESH=2):
    

    n_timepoints = position.shape[0]
    half = int(np.floor(n_timepoints/2.))
    position1 = position[:half]
    position2 = position[half:]

    
    #filter indices by speed and ignore nans
    ixs = np.where(np.logical_and(np.all(np.isfinite(position1),axis=1),
                                  speed[:half]>SPEED_THRESH))[0]
    occupancy_map1,_,_ = np.histogram2d(position1[ixs,0],
                                        position1[ixs,1],
                                        bins=np.linspace(0,780,num=81))
    
    ixs = np.where(np.logical_and(np.all(np.isfinite(position2),axis=1),
                                  speed[half:]>SPEED_THRESH))[0]

    occupancy_map2,_,_ = np.histogram2d(position2[ixs,0],
                                        position2[ixs,1],
                                        bins=np.linspace(0,780,num=81))
    
    
    ccs = []
    for unit_nr in single_units:
        sys.stdout.write('\rprocessing unit:{}/{}'.format(unit_nr,single_units[-1]))
        spk_unit = spkT[np.where(spkC==unit_nr)[0]]



        used_spikes = np.array([i for i in spk_unit if speed[i]>SPEED_THRESH])

        used_spikes_half1 = used_spikes[used_spikes<half]
        #print(used_spikes_half1)
        #break
        used_spikes_half2 = used_spikes[used_spikes>=half]

        if len(used_spikes_half1)>0 and len(used_spikes_half2)>0:
            posS = position[used_spikes_half1] 
            spike_poss,_,_ = np.histogram2d(posS[:,0],posS[:,1],bins=np.linspace(0,780,num=81))

            norm_firing = spike_poss/occupancy_map1
            norm_smth_half1 = filter_gaussian(norm_firing,2)
            norm_smth_half1[np.logical_not(np.isfinite(norm_smth_half1))] = 0

            posS = position[used_spikes_half2] 
            spike_poss,_,_ = np.histogram2d(posS[:,0],posS[:,1],bins=np.linspace(0,780,num=81))
            norm_firing = spike_poss/occupancy_map2
            norm_smth_half2 = filter_gaussian(norm_firing,2)
            norm_smth_half2[np.logical_not(np.isfinite(norm_smth_half2))] = 0
            ixs = np.logical_and(np.isfinite(norm_smth_half1.flatten()),
                                 np.isfinite(norm_smth_half2.flatten()))
            ccs.append(np.corrcoef(norm_smth_half1.flatten()[ixs],norm_smth_half2.flatten()[ixs])[0,1])
        else:
            ccs.append(np.nan)
    return np.array(ccs)



def get_spatial_firing_rate_maps(position,speed,spkC,spkT,single_units,SMOOTHING_FACTOR=1.5,SPEED_THRESH=-np.inf):
    """ Basically what it says on the tin 
        
        Arguments:
        ==========================
        
        SMOOTHING_FACTOR:       float
                                standard deviation of gaussian kernel that will smooth the data 
                                approximately in units of cm
        
        SPEED_THRESH:           float
                                ignore frames where the animals speed was lower than this 
                                units are cm/s
                        
                        
    
    """
    
    
    firing_rate_maps = []

    #inidices of positions where 
    ixs = np.where(np.logical_and(np.all(np.isfinite(position),axis=1),
                                  speed>SPEED_THRESH))[0]
    
    #bin data to get occupancy map
    occupancy_map,_,_ = np.histogram2d(position[ixs,0],position[ixs,1],bins=np.linspace(0,780,num=81))

    n_units = len(single_units)
    for loop_counter,unit_nr in enumerate(single_units):
        sys.stdout.write('\r processing unit: {}/{}'.format(1+loop_counter,n_units))
        #select spikes belongong to some unit
        spk_unit = spkT[np.where(spkC==unit_nr)[0]]

        #filter out bins that are not used for the occupancy
        used_spikes = np.array([i for i in spk_unit if (speed[i]>SPEED_THRESH and
                                                        np.all(np.isfinite(position[i])))
                               ])
        
        #if any spikes are fired
        if len(used_spikes)>0:
            posS = position[used_spikes]  #these are the positions at which each spike occurred
            tot_spikes = len(used_spikes)
            
            #bin locations where spikes were fired
            spike_poss,_,_ = np.histogram2d(posS[:,0],posS[:,1],bins=np.linspace(0,780,num=81))

            
            norm_firing = spike_poss/occupancy_map  #calculate spikes_in_bin/time_in_bin
            
            norm_smth = filter_gaussian(norm_firing,SMOOTHING_FACTOR) #smooth
            norm_smth[np.logical_not(np.isfinite(norm_smth))] = 0 #deal with nans

            firing_rate_maps.append([unit_nr,norm_smth,tot_spikes,np.max(norm_smth)*30]) #bingo
        else:
            firing_rate_maps.append([unit_nr,np.zeros([50,50]),0,0])
    return firing_rate_maps