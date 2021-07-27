import numpy as np
import sys
import cv2


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
        img2 = image[30:-230,360:-210,0]
        ret,thresh = cv2.threshold(img2,20,255,1)
        pos = np.median(np.vstack(np.where(thresh>0)),axis=1)
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

def split_occupancy_map(position,sigma,dd_, tot_pix_size=[1000,1000])