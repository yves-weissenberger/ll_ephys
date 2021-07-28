import numpy as np
#import cv2
import sys
import os
import re
import sys
import os

#PACKAGE_PARENT = '..'
package_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(package_dir)

from open_field import extract_position_from_video, get_occupancy_map, split_occupancy_map


if __name__== "__main__":

    video_dir = sys.argv[1]
    res_folder = os.path.join(video_dir,'position_extracted_folder')
    if not os.path.isdir(res_folder):
        os.mkdir(res_folder)
    
    processed_data = os.listdir(res_folder)

    video_files = [i for i in os.listdir(video_dir) if '.mp4' in i]

    res_extensions = ['_positions.npy','occupany_maps.npy']

    unprocessed_videos = []
    for vf in video_files:
        print(vf)
        if 'OF' in vf:
            big_arena = 'OFB' in vf  #adjust sizes of everything. Also need to do a calibration
            
            for res_type in res_extensions:
                f_root = re.findall(r'(.*).mp4',vf)[0]

                if any([(f_root + res_type) not in processed_data for res_type in res_extensions]):
                    video_path = os.path.join(video_dir,vf)
                    position = extract_position_from_video(vf)
                    occ_map = split_occupancy_map(position)
                    np.save(os.path.join(res_folder,'_positions.npy'),position)
                    np.save(os.path.join(res_folder,'occupancy_maps.npy'),position)



                else:
                    pass
        else:
            pass                

