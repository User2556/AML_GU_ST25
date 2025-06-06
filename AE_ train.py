
from AE_model import AE, AE_MLPClassifier
from data import CustomDataModule
import AE_config as config

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
AE-Based Classification Training Pipeline:


** for every AE_config/REPR_DIMENSION (choose) **


1) run "PRETRAINING AE FEATURE REPRESENTATIONS" >> fill AE_config/AE_CKPT_PATH
2) run "PERFROM SPARSITY GRID SEARCH" >> examine model performances >> fill AE_config/FIN_SPARSITY
3) run "MLP CV-TRAINING" >> examine model performances (further statistical analysis)
'''












### PRETRAINING AE FEATURE REPRESENTATIONS ###

model_name_fcn = lambda repr_dimension: f'AE_Model/{repr_dimension}'


if __name__ == "__main__":         
    pl.seed_everything(config.GLOBAL_SEED, workers=True)

    print(f'###  TRAINING AE:  REPR_DIM:{config.REPR_DIMENSION}  ###')
        

    logger = TensorBoardLogger("tb_logs", name=model_name_fcn(config.REPR_DIMENSION))

    checkpoint_callback = ModelCheckpoint(
        monitor="Loss/VALIDATION",    
        mode="min",            
        save_top_k=3,          
        verbose=True,
    )

    model = AE(
         pre_width_list = config.AE_PRE_WIDTH_LIST,
         repr_dimension = config.REPR_DIMENSION,
         num_in_feat = config.NUM_IN_FEAT,
         learning_rate = config.LEARNING_RATE,
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


















### PERFROM SPARSITY GRID SEARCH ### 
'''
model_name_fcn = lambda repr_dimension: f'AE_Model/{repr_dimension}/SP_GridS/SP{sparsity}'


if __name__ == "__main__":
    pl.seed_everything(config.GLOBAL_SEED, workers=True)
    for sparsity in np.arange(*config.SPARSITY_GRIDS_INFO[0], config.SPARSITY_GRIDS_INFO[1]):
            
            print(f'###  TRAINING AE_MLP (Grid Seach SPARSITY:{sparsity}  ###')


            logger = TensorBoardLogger("tb_logs", name=model_name_fcn(sparsity))

            checkpoint_callback = ModelCheckpoint(
                monitor="Loss/VALIDATION",
                mode="min",            
                save_top_k=3,          
                verbose=True,
            )

            encoder_info = [config.AE_CKPT_PATH, config.AE_PRE_WIDTH_LIST + [config.REPR_DIMENSION]]
            model = AE_MLPClassifier(
                 encoder_info=encoder_info,
                
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
'''







### MLP CV-TRAINING ### 
'''
model_name_fcn = lambda repr_dimension, iter, sparsity: f'AE_Model/{repr_dimension}/cvITER{iter}_SP{sparsity}'


if __name__ == "__main__":         
    pl.seed_everything(config.GLOBAL_SEED, workers=True)
    for cv_iter in range(config.CV_COUNT):

        print(f'###  CV-TRAINING AE_MLP:  ITER:{cv_iter} @ REPR-DIM:{config.REPR_DIMENSION} & SPARSITY:{config.FIN_SPARSITY}  ###')
        

        logger = TensorBoardLogger("tb_logs", name=model_name_fcn(config.REPR_DIMENSION, cv_iter, config.FIN_SPARSITY))
        
        checkpoint_callback = ModelCheckpoint(
            monitor="Loss/VALIDATION",    
            mode="min",            
            save_top_k=3,          
            verbose=True,
        )
        
        encoder_info = [config.AE_CKPT_PATH, config.AE_PRE_WIDTH_LIST + [config.REPR_DIMENSION]]
        model = AE_MLPClassifier(
             encoder_info=encoder_info,
            
             width_list = config.MLP_WIDTH_LIST,
             num_in_feat = config.NUM_IN_FEAT,
             num_out_classes = config.NUM_OUT_CLASSES,
             learning_rate = config.LEARNING_RATE,
            
             sparsity = config.FIN_SPARSITY   ## OPTIMAL SPARSITY ##
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