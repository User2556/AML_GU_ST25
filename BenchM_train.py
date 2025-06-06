
from BenchM_model import MLPClassifier
from data import CustomDataModule
import BenchM_config as config

from itertools import product
from tqdm import tqdm
import torch
import numpy as np
import pytorch_lightning as pl
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision("medium") # to make lightning happy






'''
Benchmark Classifier Training Pipeline:


** for every BenchM_config/REPR_DIMENSION (choose) **

1) run "PERFROM SPARSITY GRID SEARCH" >> examine model performances >> fill BenchM_config/FIN_SPARSITY 
2) run "MLP CV-TRAINING" >> examine model performances (further statistical analysis)
'''








### PERFROM SPARSITY GRID SEARCH ### 

model_name_fcn = lambda sparsity: f'BenchM_Model/{config.NUM_IN_FEAT}/SP_GridS/SP{sparsity}'



if __name__ == "__main__":
    pl.seed_everything(config.GLOBAL_SEED, workers=True)
    for sparsity in np.arange(*config.SPARSITY_GRIDS_INFO[0], config.SPARSITY_GRIDS_INFO[1]):
            
        print(f'###  TRAINING MLP (Grid Seach SPARSITY:{sparsity}  ###')
        
        
        logger = TensorBoardLogger("tb_logs", name=model_name_fcn(sparsity))
        
        checkpoint_callback = ModelCheckpoint(
            monitor="Loss/VALIDATION",
            mode="min",            
            save_top_k=3,          
            verbose=True,
        )
        
        model = MLPClassifier(
            width_list = config.MLP_WIDTH_LIST,
            num_in_feat = config.NUM_IN_FEAT,
            num_out_classes = config.NUM_OUT_CLASSES,
            learning_rate = config.LEARNING_RATE,
            
            sparsity = sparsity   ## GRID SEARCH ##
            )
        
        dm = CustomDataModule(
            data_superdir = config.DATA_SUPERDIR,
            batch_size = config.BATCH_SIZE,
            num_workers = config.NUM_WORKERS,
            data_split = config.DATA_SPLIT,
            split_seed = config.SPLIT_SEED
            )
        
        trainer = pl.Trainer(
            logger=logger,
            accelerator=config.ACCELERATOR,
            devices=config.DEVICES,
            min_epochs=config.MIN_EPOCHS,
            max_epochs=config.MAX_EPOCHS,
            precision=config.PRECISION,
            callbacks=[checkpoint_callback,
                       EarlyStopping(monitor="Loss/VALIDATION", patience=config.VAL_STOP_PATIENCE)
                      ],
        )
        
        trainer.fit(model, dm)







### MLP CV-TRAINING ### 

'''
model_name_fcn = lambda iter, sparsity: f'BenchM_Model/{config.NUM_IN_FEAT}/cvITER{iter}_SP{sparsity}'


if __name__ == "__main__":         
    pl.seed_everything(config.GLOBAL_SEED, workers=True)
    for cv_iter in range(config.CV_COUNT):

        print(f'###  CV-TRAINING MLP:  ITER:{cv_iter} @ SPARSITY:{config.FIN_SPARSITY}  ###')
        

        logger = TensorBoardLogger("tb_logs", name=model_name_fcn(cv_iter, config.FIN_SPARSITY))
        checkpoint_callback = ModelCheckpoint(
            monitor="Loss/VALIDATION",    
            mode="min",            
            save_top_k=3,          
            verbose=True,
        )
        model = MLPClassifier(
            width_list = config.MLP_WIDTH_LIST,
            num_in_feat = config.NUM_IN_FEAT,
            num_out_classes = config.NUM_OUT_CLASSES,
            learning_rate = config.LEARNING_RATE,

            sparsity = config.FIN_SPARSITY   ## OPTIMAL SPARSITY LEVELS ###
            )
        
        dm = CustomDataModule(
            data_superdir = config.DATA_SUPERDIR,
            batch_size = config.BATCH_SIZE,
            num_workers = config.NUM_WORKERS,
            data_split = config.DATA_SPLIT,

            split_seed = np.random.randint(0)   ## CV-TRAINING ##
            )
        
        trainer = pl.Trainer(
            logger=logger,
            accelerator=config.ACCELERATOR,
            devices=config.DEVICES,
            min_epochs=config.MIN_EPOCHS,
            max_epochs=config.MAX_EPOCHS,
            precision=config.PRECISION,
            callbacks=[checkpoint_callback,
                       EarlyStopping(monitor="Loss/VALIDATION", patience=config.VAL_STOP_PATIENCE)
                      ],
        )
        trainer.fit(model, dm)
'''