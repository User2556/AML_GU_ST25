
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import pytorch_lightning as pl
import numpy as np
import random
















class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_superdir, batch_size, num_workers, data_split, split_seed, prec_dtype=np.float32):
        super().__init__()
        self.data_superdir = data_superdir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_split = data_split
        self.split_seed = split_seed

        self.prec_dtype = prec_dtype

    def setup(self, stage):
        dev_ds = CustomDataset(data_superdir=self.data_superdir, prec_dtype=self.prec_dtype)

        size_train_ds = int(self.data_split[0]*len(dev_ds))
        self.train_ds, self.val_ds = random_split(dev_ds, [size_train_ds, len(dev_ds)-size_train_ds],
                                                  generator=torch.Generator().manual_seed(self.split_seed)
                                                  )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    








class CustomDataset(Dataset):
    def __init__(self, data_superdir, prec_dtype, include_tc=False, dev=True):

        path_mid = 'train' if dev else 'test'
        self.data_dir = os.path.join(data_superdir, path_mid)
        
        self.prec_dtype = prec_dtype
        self.include_tc = include_tc

        self._source_data()
    
    def _source_data(self):
        ### ATT: SZ samples are assigned class indicator _0_, BP samples are assigned class indicator _1_ ###

        sz_data_dir = os.path.join(self.data_dir,'SZ')
        sz_subj_ind_list = os.listdir(sz_data_dir)
        sz_path_list = [(os.path.join(sz_data_dir, subj_ind,'fnc.npy'),
                        (os.path.join(sz_data_dir, subj_ind,'icn_tc.npy'))) for subj_ind in sz_subj_ind_list]
        
        sz_ds = []
        for subj_ind, path in zip(sz_subj_ind_list, sz_path_list):
            subj_features = [np.load(path[0]).flatten().astype(self.prec_dtype),
                             np.load(path[1]).astype(self.prec_dtype)]
            subj_features = subj_features if self.include_tc else subj_features[0]
            sz_ds.append((subj_features, np.array([1,0]).astype(self.prec_dtype), subj_ind))        
            
        
        bp_data_dir = os.path.join(self.data_dir,'BP')
        bp_subj_ind_list = os.listdir(bp_data_dir)
        bp_path_list = [(os.path.join(bp_data_dir, subj_ind,'fnc.npy'),
                        (os.path.join(bp_data_dir, subj_ind,'icn_tc.npy'))) for subj_ind in bp_subj_ind_list]
        bp_ds = []
        for subj_ind, path in zip(bp_subj_ind_list, bp_path_list):
            subj_features = [np.load(path[0]).flatten().astype(self.prec_dtype),
                             np.load(path[1]).astype(self.prec_dtype)]
            subj_features = subj_features if self.include_tc else subj_features[0]
            bp_ds.append((subj_features, np.array([0,1]).astype(self.prec_dtype), subj_ind))        
            
        
        combined_ds = sz_ds + bp_ds
        random.shuffle(combined_ds)
        self.combined_ds = combined_ds 
        
    def __len__(self):
        return len(self.combined_ds)
        
    def __getitem__(self, idx):
        return self.combined_ds[idx]