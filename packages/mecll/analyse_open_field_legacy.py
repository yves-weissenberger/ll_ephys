import numpy as np
import numba 

@numba.jit(nopython=True)
def gaussian_kernel(kernx):
    kerny = np.exp(np.power(kernx, 2)/2 * (-1))
    return kerny

@numba.jit(nopython=True)
def calculate_firing_rate_for_cluster_parallel(spike_positions,smooth, positions_x, positions_y, number_of_bins_x, number_of_bins_y, bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms):
    """ This basically loops over all bins, calculates the distances from """
    spike_positions_x,spike_positions_y = spike_positions
    firing_rate_map = np.zeros((number_of_bins_x, number_of_bins_y))
    for x in range(number_of_bins_x):
        #print(x,number_of_bins_x)
        for y in range(number_of_bins_y):
            #print(x)
            px = x * bin_size_pixels + (bin_size_pixels / 2)
            py = y * bin_size_pixels + (bin_size_pixels / 2)
            spike_distances = np.sqrt(np.power(px - spike_positions_x, 2) + np.power(py - spike_positions_y, 2))
            #print(spike_distances)
            spike_distances = spike_distances[~np.isnan(spike_distances)]
            occupancy_distances = np.sqrt(np.power((px - positions_x), 2) + np.power((py - positions_y), 2))
            occupancy_distances = occupancy_distances[~np.isnan(occupancy_distances)]
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_pixels)[0])

            
            #toget reliable firing rate estimate need to have been here at least min-dwell times
            if bin_occupancy >= min_dwell:
                firing_rate_map[x, y] = np.sum(gaussian_kernel(spike_distances/smooth)) / (np.sum(gaussian_kernel(occupancy_distances/smooth)) * (dt_position_ms/1000))

            else:
                firing_rate_map[x, y] = 0
    #firing_rate_map = np.rot90(firing_rate_map)
    return firing_rate_map



def get_gaussint_rate_maps(unit_nr,aligner,spkC,spkT,position,smooth=30,n_bins=45,min_dwell=5,min_dwell_distance_pixels=20,d_position_ms=1):

    """ Arguments:
    ==============================
    
    unit_nr
    """

    
    
    number_of_bins_x = n_bins = 45
    mx_pos = np.nanmax(position)
    bin_size_pixels=mx_pos/number_of_bins_x

    spk_unit = spkT[np.where(spkC==unit_nr)[0]]


    aligned_T = aligner.A_to_B(spk_unit)
    #get the spikes that are in bounds for position encoding
    spks_unit_in_bounds = np.where(np.logical_not(np.isnan(aligned_T)))[0]
    used_spikes = aligned_T[spks_unit_in_bounds].astype('int')
    used_spikes = np.array([i for i in used_spikes if speed[i]])#>SPEED_THRESH])

    posS = (position[used_spikes][:,0],position[used_spikes][:,1])
    
    
    rate_map = calculate_firing_rate_for_cluster_parallel(posS,smooth, positions_x, positions_y, 
                                                 number_of_bins_x, number_of_bins_y,
                                                 bin_size_pixels, min_dwell, 
                                                 min_dwell_distance_pixels, dt_position_ms)
    return rate_map



# https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def shift(array_to_shift, n):
    if n >= 0:
        return np.concatenate((np.full(n, np.nan), array_to_shift[:-n]))
    else:
        return np.concatenate((array_to_shift[-n:], np.full(-n, np.nan)))


'''
Shifts 2d array along given axis.
array_to_shift : 2d array that is to be shifted
n : array will be shifted by n places
axis : shift along this axis (should be 0 or 1)
'''

@jit(nopython=True)
def shift_2d(array_to_shift, n, axis):
    shifted_array = np.zeros_like(array_to_shift)
    if axis == 0:  # shift along x axis
        if n == 0:
            return array_to_shift
        if n > 0:
            shifted_array[:, :n] = 0
            shifted_array[:, n:] = array_to_shift[:, :-n]
        else:
            shifted_array[:, n:] = 0
            shifted_array[:, :n] = array_to_shift[:, -n:]

    if axis == 1:  # shift along y axis
        if n == 0:
            return array_to_shift
        elif n > 0:
            shifted_array[-n:, :] = 0
            shifted_array[:-n, :] = array_to_shift[n:, :]
        else:
            shifted_array[:-n, :] = 0
            shifted_array[-n:, :] = array_to_shift[:n, :]
    return shifted_array




# shifts array by x and y
def get_shifted_map(firing_rate_map, x, y):
    shifted_map = shift_2d(firing_rate_map, x, 0)
    shifted_map = shift_2d(shifted_map, y, 1)
    return shifted_map
