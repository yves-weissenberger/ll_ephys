import numpy as  np
from scipy.ndimage import gaussian_filter1d
def build_spike_array(single_units,spkT,spkC):

    nUnits = len(single_units)
    maxT = int(np.ceil(np.max(spkT/30.)))
    spike_array = np.zeros([nUnits,maxT])
    for ctr,unit in enumerate(single_units):
        spike_times = (np.floor(spkT[spkC==unit]/30.)).astype('int')
        spike_array[ctr,spike_times] = 1
    return spike_array
    

def smooth_spike_array(spke_array,sigma=10):
    spike_smooth = []
    for i in spike_array:
        spike_smooth.append(gaussian_filter1d(i,sigma))


    spike_smooth = np.array(spike_smooth)
    return spike_smooth


#I think this works as a downsampling thing. This is now downsampled to 100ms
def down_sample_spikes(spike_array,factor=50):
    #factor = 50
    n_units = spike_array.shape[0]
    n_timepoints = spike_array.shape[1]
    mx_ = int(np.floor(n_timepoints/factor)*factor)
    spike_array_downsample = np.reshape(spike_array[:,:mx_],[n_units,int(mx_/factor),factor]).sum(axis=2)
    return spike_array_downsample


def get_task_boudaries(lines,aligner):
    change_task_event = eval(lines[7][2:])['change_task']
    
    task_boundaries = []
    has_start = False
    for l in lines:
        
        if not has_start:
            if 'D'==l[0]:
                t_ = float(re.findall(r'D ([0-9]*)\s',l)[0])
                if np.isfinite(aligner.B_to_A(t_)):
                    print('1')
                    task_boundaries.append(aligner.B_to_A(t_))
                    has_start = True


        if 'change_task_start' in l:
            t_ = float(re.findall(r'P ([0-9]*)\s',l)[0])
            task_boundaries.append(aligner.B_to_A(t_))
            #print(l)
            #print(t_)
            
    for l in lines[::-1]:
        if l[0]=='D':
            t_  = float(re.findall(r'D ([0-9]*)\s',l)[0])
            if np.isfinite(aligner.B_to_A(t_)):
                print('2')
                task_boundaries.append(aligner.B_to_A(t_))
                break
    task_boundaries = (np.array(task_boundaries)/30.).astype('int')
    return task_boundaries