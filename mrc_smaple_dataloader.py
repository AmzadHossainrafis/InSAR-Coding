import torch
from mrc_insar_common.data import data_reader
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import logging
import glob
import tqdm 
import logging
from datetime import datetime
import re #for finding pattern /this module let you check if a particular string matches a given regular expression (or if a given regular expression matches a particular string, which comes down to the same thing).

log = logging.getLogger(__name__)


def get_delta_days(date_string): 
    '''this function find out the basic def of date'''
    date_format = "%Y%m%d"#This is the main formet the date represented in the reseource 
    tokens = re.split("_|\.", date_string) #for slpling the date from the given filde 
    #2017-01-12>>token[0] _ 2017-01-20>>token[1]

    date1 = datetime.strptime(tokens[0], date_format) #left side of the split is store in tokens[0]
    # example of datetime.stritime >>>date_string = "21 June, 2018" 
    #>>date_object = datetime.strptime(date_string, "%d %B,%Y")
    #>>date_object = 2018-06-21 in this the date formeting deff
    date2 = datetime.strptime(tokens[1], date_format) # right side of the split  
    delta_days = np.abs((date2 - date1).days) # just find out the absulute deff between the dates 
    return delta_days


class SpatialTemporalDataset(Dataset):#inharit the dataset class

    def __init__(self,
                 filt_dir,
                 filt_ext,
                 bperp_dir,
                 bperp_ext,
                 coh_dir,
                 coh_ext,
                 conv1,
                 conv2,
                 width,
                 height,
                 ref_mr_path,
                 ref_he_path,
                 patch_size=38,
                 stride=0.5):
        self.filt_paths = sorted(glob.glob('{}/*{}'.format(filt_dir, filt_ext)))#  provide a list of sorted data file path of filt_ext fometed file 
        self.bperp_paths = sorted(glob.glob('{}/*{}'.format(bperp_dir, bperp_ext)))# provide a list of sorted data file path of bperp_ext fometed file 
        self.coh_paths = sorted(glob.glob('{}/*{}'.format(coh_dir, coh_ext)))# rovide a list of sorted data file path of coh_ext fometed file 
        self.ref_mr_path = ref_mr_path
        self.ref_he_path = ref_he_path
        self.conv1 = conv1 #simple veriable 
        self.conv2 = conv2 
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.stride = stride

        self.stack_size = len(self.filt_paths) #find out the lenght of the sorted self.filt_path

        self.ddays = np.zeros(self.stack_size) #make a 1d list 0 of length of 
        self.bperps = np.zeros(self.stack_size)


        for idx in tqdm.tqdm(range(self.stack_size)): #tqdm >>it's just add a animation of loading 
            # read delta days
            bperp_path = self.bperp_paths[idx]# taking every element in the list one by one 
            date_string = bperp_path.split('/')[-1].replace(bperp_ext, "") 
            #split the the element in the list with respect to "/" [-1] represent the last element in the splited list
            #then replace the extention(bparp_ext) part with space 
            delta_day = get_delta_days(date_string)
            self.ddays[idx] = delta_day

            # read bperp
            self.bperps[idx] = data_reader.readBin(bperp_path, 1, 'float')[0][0]
        print('bp dday loaded')

        self.all_sample_coords = [(row_idx, col_idx)
                                  for row_idx in range(0, self.height - self.patch_size - 1, int(self.patch_size * stride)) 
                                  #range(start,end,skip)
                                  #0,1500-28-1, int(28*0.5) 
                                  #0,1471,14
                                  #to see the example please run this code
                                  #x =[(i,j) for i in range(0,1471,14) for j in range(0,1471,14)]
                                  #print(x)
                                  for col_idx in range(0, self.width - self.patch_size - 1, int(self.patch_size * stride))]

                                  #need to chk again

    def __len__(self):
        return len(self.all_sample_coords)

    def __getitem__(self, idx):
        coord = self.all_sample_coords[idx]

        mr_target = data_reader.readBin(self.ref_mr_path, self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))

    #why he mr outut 
    
    
#
        he_target = data_reader.readBin(self.ref_he_path, self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))

        filt_input = np.zeros([self.stack_size, self.patch_size, self.patch_size])    # [N, h ,w] for a single training sample, 
        coh_input = np.zeros([self.stack_size, self.patch_size, self.patch_size])    # [N, h ,w] for a single training sample

        for i in range(self.stack_size):
            # !! here is an example that only uses phase information 
            filt_input[i] = np.angle(data_reader.readBin(self.filt_paths[i], self.width, 'floatComplex', crop=(coord[0], coord[1], self.patch_size, self.patch_size)))
            #MRC InSAR Library - https://pypi.org/project/MRC-InSAR-Common/  follow this link  for datareader of readbin
 
            coh_input[i] = data_reader.readBin(self.coh_paths[i], self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))

            
   #pt with imshow all the value in the return 
   #try to find out the out
        return {
            'input': filt_input,    
            'coh': coh_input,
            'mr': np.expand_dims(mr_target, 0),#why  expand_dims()
            'he': np.expand_dims(he_target, 0),
            'ddays':
                self.
                ddays,    # ddays and bperps are shared for all training samples in a stack, it can be used in a more effecient way, here is just an example
            'bperps':
                self.bperps    # 
        }


if __name__ == "__main__":

    sample_filt_dir = '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/ifg_hr/'
    sample_filt_ext = '.diff.orb.statm_cor.natm.filt'

    sample_coh_dir = '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/ifg_hr/'
    sample_coh_ext = '.diff.orb.statm_cor.natm.filt.coh'

    sample_bperp_dir = '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/ifg_hr/'
    sample_bperp_ext = '.bperp'

    sample_width = 1500
    sample_height = 1500

    sample_conv1 = -0.0110745533168
    sample_conv2 = -0.00134047881374

    sample_patch_size = 28
    sample_stride = 0.5

    ref_mr_path = '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/fit_hr/def_fit_cmpy'
    ref_he_path = '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/fit_hr/hgt_fit_m'

    sample_db = SpatialTemporalDataset(sample_filt_dir, sample_filt_ext, sample_bperp_dir, sample_bperp_ext, sample_coh_dir, sample_coh_ext, sample_conv1, sample_conv2, sample_width, sample_height,ref_mr_path, ref_he_path, sample_patch_size, sample_stride)

    print('db length {}'.format(len(sample_db)))

    sample_dataloader = DataLoader(sample_db, 4, shuffle=True, num_workers=4)

    for batch_idx, batch in enumerate(sample_dataloader):
        print(batch_idx)
        #pt with imshow
        print(batch['input'].shape)
        print(batch['coh'].shape)
        print(batch['mr'].shape)
        print(batch['he'].shape)
        print(batch['ddays'].shape)
        print(batch['bperps'].shape)
        break

    # vsulize sample patchs in a batch
    fig, axs = plt.subplots(1,4, figsize=(8,2))
    input_shape = batch['input'][0].shape # first training example
    for i in range(input_shape[2]): # size of stack
        im = axs[i].imshow(batch['input'][0][i], cmap='jet', vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046) 
        if i == 3: 
            break
    fig.tight_layout()
    plt.show()
    
    
