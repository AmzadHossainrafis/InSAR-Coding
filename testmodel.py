from os import truncate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from mrc_smaple_dataloader import SpatialTemporalDataset
from pytorch_lightning import Trainer
#mr ifg 
class Lit(pl.LightningModule,SpatialTemporalDataset):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Lit, self).__init__()
        # Fully connected neural network with one hidden layer
        self.input_size =  784
        self.l1 = nn.Conv3d(input_size, 500, 5)
        self.maxpool=nn.MaxPool3d(3,3,2)
        self.l2 = nn.Conv3d(500, 10, 5)
        self.l3= nn.Linear()

    def forward(self, x):
        out = self.l1(x)
        out = self.(out)
        out = self.l2(out)
        
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        
        tensorboard_logs = {'train_loss': loss}
        # use key 'log'
        return {"loss": loss, 'log': tensorboard_logs}

    # define what happens for testing here

    def train_dataloader(self):

            
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
            train_dataset = SpatialTemporalDataset(sample_filt_dir, sample_filt_ext , sample_bperp_dir, sample_bperp_ext, sample_coh_dir, sample_coh_ext, sample_conv1, sample_conv2, sample_width, sample_height,ref_mr_path, ref_he_path, sample_patch_size, sample_stride)

            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=100, num_workers=4, shuffle=True
            )
            return train_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=.001)
    
  

if __name__ == '__main__':
    model = Lit(784, 500, 10)
    trainer = Trainer(max_epochs=1, fast_dev_run=True, gpus=1)
    trainer.fit(model)