import numpy as np
import sys
import cv2
import time
#import tdqm

def extract_position_from_video(path,save_path=None,verbose=0):
    """ Function that extracts the position of the
        animal by detecting a dark blob
    """
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 1
    position = []
    st = time.time()
    while success:
        success,image = vidcap.read()
        if success:
            img2 = image[30:-230,360:-210,0]
            ret,thresh = cv2.threshold(img2,20,255,1)
            if np.any(thresh>0) and not np.sum(thresh>1000):
                pos = np.median(np.vstack(np.where(thresh>0)),axis=1)
            else:
                pos = [np.nan,np.nan]
            position.append(pos)
            sys.stdout.write("\rframeNR:{:.2f}  | iter speed:{:.4f}".format(count,(time.time()-st)/float(count)))
            sys.stdout.flush()
            #break
            #print('Read a new frame: ', success)
            count += 1
    return np.array(position)


def get_occupancy_map(position,sigma=25,dd_=100,tot_pix_size=[1000,1000]):
    """ make an array of the occupancy map"""
    occupancy_arr = np.zeros([1000,1000])
    g = gaussian_kernel(2*dd_,sigma)
    len_pos = len(position)
    for ctr,i in enumerate(position):
        if np.remainder(ctr,10)==0:
            sys.stdout.write('\rframe:{:.2f}/{:.2f}'.format(ctr,len_pos))
            sys.stdout.flush()
        try:
            xp, yp = i.astype(int)

            occupancy_arr[2*dd_+xp-dd_:2*dd_+xp+dd_,2*dd_+yp-dd_:2*dd_+yp+dd_] += g
        #occupancy_arr += g
        except:
            pass   
    return occupancy_arr

def split_occupancy_map(position,n_splits=8,sigma=25,dd_=100,tot_pix_size=[1000,1000]):
    """ split the data into n_splits and calculate occupancy maps for each part
        of the data.
    """
    len_position = position.shape[0]
    split_size= int(np.floor(len_position/n_splits))
    occupancy_maps = []
    for i in range(n_splits-1):
        tmp_ = get_occupancy_map(position[i*split_size:(i+1)*split_size],
                                 sigma=sigma,
                                 dd_=dd_,
                                 tot_pix_size=tot_pix_size)
        occupancy_maps.append(tmp_)
    tmp_ = get_occupancy_map(position[n_splits*split_size:])
    occupancy_maps.append(tmp_)
    return np.array(occupancy_maps)

    



def gaussian_kernel(win_size, sigma):
    t = np.arange(win_size)
    x, y = np.meshgrid(t, t)
    o = (win_size - 1) / 2
    r = np.sqrt((x - o)**2 + (y - o)**2)
    scale = 1 / (sigma**2 * 2 * np.pi)
    return scale * np.exp(-0.5 * (r / sigma)**2)