
from MIFS_model import MIFS_MLPClassifier
from data import CustomDataModule
import MIFS_config as config

from sklearn.feature_selection import mutual_info_classif
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
MIFS-Based Classification Training Pipeline:


0) run "FIND maxMI FEATURE RANKING"


** for every MIFS_config/REPR_DIMENSION (choose) **

1) run "PERFROM SPARSITY GRID SEARCH" >> examine model performances >> fill AE_config/FIN_SPARSITY
3) run "MLP CV-TRAINING" >> examine model performances (further statistical analysis)
'''























def get_Bin_MIs(stack_x_iter, y_dens):

    def calc_MI(feat_marg_dens, feat_y0_dens, feat_y1_dens, y_dens, dx):
        eps = 1e-12 #introduce a small offset to avoid div-by-0 errors
        ratio_y0 = np.log((feat_y0_dens + eps) / (feat_marg_dens + eps))
        ratio_y1 = np.log((feat_y1_dens + eps) / (feat_marg_dens + eps))

        mi = y_dens[0]*np.sum(feat_y0_dens*ratio_y0)*dx + y_dens[1]*np.sum(feat_y1_dens*ratio_y1)*dx
        return mi

    freedman_h = lambda iqr, n: 2*(iqr/n**(1/3))


    num_feat = stack_x_iter[0].shape[0]
    mi_log = np.zeros(num_feat)

    for feat_idx, (feat_slice_y0, feat_slice_y1) in tqdm(enumerate(zip(stack_x_iter[0], stack_x_iter[1])), total=num_feat):
        
        #Marginal Feature Density 
        feat_slice_full = np.concatenate([feat_slice_y0, feat_slice_y1], axis=0)

        # (BINNING-Approach: Freedman's Rule)        
        q1 = np.percentile(feat_slice_full, 25)
        q3 = np.percentile(feat_slice_full, 75)
        iqr = q3 - q1
        bin_h = freedman_h(iqr, feat_slice_full.size)
        bin_k = int(np.ceil((feat_slice_full.max() - feat_slice_full.min()) / bin_h))

        bin_edges = np.histogram_bin_edges(feat_slice_full, bins=bin_k)
        bin_width = np.diff(bin_edges)[0] 

        feat_marg_dens = np.histogram(feat_slice_full, bins=bin_edges, density=True)[0]
        
        
        feat_y0_dens = np.histogram(feat_slice_y0, bins=bin_edges, density=True)[0]
        feat_y1_dens = np.histogram(feat_slice_y1, bins=bin_edges, density=True)[0]

        mi = calc_MI(feat_marg_dens, feat_y0_dens, feat_y1_dens, y_dens, bin_width)
        mi_log[feat_idx] = mi
    
    return mi_log


def get_feat_rank(mi_log):
    feat_rank = np.argsort(mi_log)[::-1] #feature ranking from largest to lowest MI
    return feat_rank




















### FIND maxMI FEATURE RANKING ###


if __name__ == "__main__":         
    pl.seed_everything(config.GLOBAL_SEED, workers=True)

    print(f'###  FINDING MIFS RANKING:  REPR_DIM:{config.REPR_DIMENSION}  ###')
        

    dm = CustomDataModule(
        data_superdir = config.DATA_SUPERDIR,
        batch_size = config.BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
        data_split = config.DATA_SPLIT,
        split_seed = config.SPLIT_SEED
        )
    train_ds = dm.train_ds
    
    

    #preparing data matrices
    list_x_sz0 = []
    for sample in dm.train_ds:
        if sample[1][0] == 1:
            list_x_sz0.append(sample[0])
    stack_x_sz0 = np.stack(list_x_sz0, axis=1)

    list_x_bp1 = []
    for sample in dm.train_ds:
        if sample[1][1] == 1:
            list_x_bp1.append(sample[0])
    stack_x_bp1 = np.stack(list_x_bp1, axis=1)
    stack_x_iter = [stack_x_sz0, stack_x_bp1]

    stack_y_full = np.concatenate([np.full(stack_x_sz0.shape[1], 0), np.full(stack_x_bp1.shape[1], 1)], axis=0)
    
    #calculating y_marginal density
    py1 = np.count_nonzero(stack_y_full) / stack_y_full.size
    y_dens = np.array([1-py1, py1])



    #determining feature MI scores
    mi_log_Bin = get_Bin_MIs(stack_x_iter, y_dens)

    #determining maxMI index ranking and logging
    feat_rank_Bin = get_feat_rank(mi_log_Bin)
    mi_arch_Bin = np.stack(feat_rank_Bin, axis=0)
    np.save('tb_logs/MIFS_Model/mi_arch.npy', mi_arch_Bin)














### PERFROM SPARSITY GRID SEARCH ### 

model_name_fcn = lambda repr_dimension: f'MIFS_Model/{repr_dimension}/SP_GridS/SP{sparsity}'


if __name__ == "__main__":
    pl.seed_everything(config.GLOBAL_SEED, workers=True)
    for sparsity in np.arange(*config.SPARSITY_GRIDS_INFO[0], config.SPARSITY_GRIDS_INFO[1]):
            
            print(f'###  TRAINING MIFS_MLP (Grid Seach SPARSITY:{sparsity}  ###')


            logger = TensorBoardLogger("tb_logs", name=model_name_fcn(sparsity))

            checkpoint_callback = ModelCheckpoint(
                monitor="Loss/VALIDATION",
                mode="min",            
                save_top_k=3,          
                verbose=True,
            )

            mifs_info = [config.MI_ARCH_PATH, config.REPR_DIMENSION]
            model = MIFS_MLPClassifier(
                 encoder_info=mifs_info,
                
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

model_name_fcn = lambda repr_dimension, iter, sparsity: f'MIFS_Model/{repr_dimension}//cvITER{iter}_SP{sparsity}'


if __name__ == "__main__":         
    pl.seed_everything(config.GLOBAL_SEED, workers=True)
    for cv_iter in range(config.CV_COUNT):

        print(f'###  CV-TRAINING MIFS_MLP:  ITER:{cv_iter} @ REPR-DIM:{config.REPR_DIMENSION} & SPARSITY:{config.FIN_SPARSITY}  ###')
        

        logger = TensorBoardLogger("tb_logs", name=model_name_fcn(config.REPR_DIMENSION, cv_iter, config.FIN_SPARSITY))
        
        checkpoint_callback = ModelCheckpoint(
            monitor="Loss/VALIDATION",    
            mode="min",            
            save_top_k=3,          
            verbose=True,
        )
        
        mifs_info = [config.MI_ARCH_PATH, config.REPR_DIMENSION]
        model = MIFS_MLPClassifier(
             encoder_info=mifs_info,
            
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
