import os
from datetime import datetime
import shutil




if __name__=='__main__':

    subject = '39951'
    date = '2021-08-14'

    
    target_dir = r'D:\MEC_data\post_ks\2021-08-14_39951_ks25'
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    target_file = subject + '_' + date + '_all.dat'
    target_path = os.path.join(target_dir,target_file)
    path_1 = r'I:\mouse_dropbox\team_mouse Dropbox\data_storage\14-08-21\2021-08-14_16-01-05\Record Node 101\experiment1\recording1\continuous\Neuropix-PXI-100.2'
    path_2 = r'I:\mouse_dropbox\team_mouse Dropbox\data_storage\14-08-21\2021-08-14_17-20-04\Record Node 101\experiment1\recording1\continuous\Neuropix-PXI-100.0'
    dat_files = [
                os.path.join(path_1,'continuous.dat'),
                os.path.join(path_2,'continuous.dat'),
                ]

    with open(os.path.join(target_dir,'stitch_info_' + subject + '_' + date + '.txt'),'w') as sf:
        sf.write(repr(dat_files))
        sf.write('\n')
        for ix,ff_ in enumerate(dat_files):
            sf.write('nbytes_in_ix_{}: {}'.format(str(ix),str(os.stat(ff_).st_size)))
            sf.write('\n')
    
    with open(target_path, "ab") as wfd:
        for f in dat_files:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)
    print('\ndone!')