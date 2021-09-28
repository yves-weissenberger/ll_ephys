import numpy as np
from numba import jit
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


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
         # linear interpolation of NaNs
         nans, x= nan_helper(y)
         y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def remove_nans_from_both_arrays(array1, array2):
    not_nans_in_array1 = ~np.isnan(array1)
    not_nans_in_array2 = ~np.isnan(array2)
    array1 = array1[not_nans_in_array1 & not_nans_in_array2]
    array2 = array2[not_nans_in_array1 & not_nans_in_array2]
    return array1, array2


def remove_nans_and_inf_from_both_arrays(array1, array2):
    not_nans_in_array1 = ~np.isnan(array1)
    not_nans_in_array2 = ~np.isnan(array2)
    array1 = array1[not_nans_in_array1 & not_nans_in_array2]
    array2 = array2[not_nans_in_array1 & not_nans_in_array2]

    not_nans_in_array1 = ~np.isinf(array1)
    not_nans_in_array2 = ~np.isinf(array2)
    array1 = array1[not_nans_in_array1 & not_nans_in_array2]
    array2 = array2[not_nans_in_array1 & not_nans_in_array2]
    return array1, array2


# shifts array by x and y
def get_shifted_map(firing_rate_map, x, y):
    shifted_map = shift_2d(firing_rate_map, x, 0)
    shifted_map = shift_2d(shifted_map, y, 1)
    return shifted_map


# remove from both where either of them is 0
def remove_zeros(array1, array2):
    array2 = np.nan_to_num(array2).flatten()
    array1 = np.nan_to_num(array1).flatten()
    array2_tmp = np.take(array2, np.where(array1 != 0))
    array1_tmp = np.take(array1, np.where(array2 != 0))
    array2 = np.take(array2_tmp, np.where(array2_tmp[0] != 0))
    array1 = np.take(array1_tmp, np.where(array1_tmp[0] != 0))
    return array1.flatten(), array2.flatten()


# remove from both where either of them is not a number (nan) - I am not proud of this, but nothing worked with np.nan
def remove_nans(array1, array2):
    array2 = array2.flatten()
    array2[np.isnan(array2)] = 666
    array1 = array1.flatten()
    array1[np.isnan(array1)] = 666
    array2_tmp = np.take(array2, np.where(array1 != 666))
    array1_tmp = np.take(array1, np.where(array2 != 666))
    array2 = np.take(array2_tmp, np.where(array2_tmp[0] != 666))
    array1 = np.take(array1_tmp, np.where(array1_tmp[0] != 666))
    return array1.flatten(), array2.flatten()

