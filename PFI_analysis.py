
import AE_config as config
from data import CustomDataModule
from AE_model import AE_MLPClassifier

import os

import pytorch_lightning as pl
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from mne.viz.circle import _plot_connectivity_circle
import numpy as np
from matplotlib.colors import to_rgba


def get_fullFNC(flat_FNC, n_components=105):
    triu_indices = np.triu_indices(n_components, k=1)
    
    fnc_matrix = np.zeros((n_components, n_components))
    flat_FNC = flat_FNC.flatten()
    fnc_matrix[triu_indices] = flat_FNC
    fnc_matrix += fnc_matrix.T
    
    np.fill_diagonal(fnc_matrix, 1)

    return fnc_matrix




permutation_iter = 100








'''
PFI-Analysis of AE-Based Classifier:

1) carry out the instructions in ++ AE_train.py ++ for all REPR_DIMENSION
2) identify most performant CV-iteration for the optimal REPR_DIMENSION and notate below, together with FIN_SPARSITY used
3) run below
'''




opt_repr_dimension = np.nan ##FILL##
opt_cviter = np.nan ##FILL##
opt_sparsity = np.nan ##FILL##

























if __name__ == "__main__":

    mlp_width_list = config.get_down_powers(opt_repr_dimension)
    ae_ckpt_path = f'./tb_logs/AE_Model/{opt_repr_dimension}'
    mlp_ckpt_path = f'./tb_logs/AE_Model/{opt_repr_dimension}/cvITER{opt_cviter}_SP{opt_sparsity}'



    #setting up AE_MLP model
    encoder_info = [ae_ckpt_path, config.AE_PRE_WIDTH_LIST + [opt_repr_dimension]]
    model = AE_MLPClassifier.from_checkpoint(
          mlp_ckpt_path,

          encoder_info=encoder_info,
          width_list = mlp_width_list,
          num_in_feat = config.NUM_IN_FEAT,
          num_out_classes = config.NUM_OUT_CLASSES,
          learning_rate = config.LEARNING_RATE,

          sparsity = opt_sparsity   ## GRID SEARCH ##
          )

    device = f'{config.ACCELERATOR}:{config.DEVICES[0]}'
    model.to(device)
    model.eval()



    #initializing data module
    dm = CustomDataModule(
        data_superdir = config.DATA_SUPERDIR,
        batch_size = config.BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
        data_split = config.DATA_SPLIT,
        split_seed = config.SPLIT_SEED
        )                

    dm.setup(stage=None)
    val_ds = dm.val_ds



    #calculating baseline loss
    bt_x = torch.stack([torch.from_numpy(sample[0]) for sample in val_ds])
    bt_x = bt_x.to(device)

    bt_y = torch.stack([torch.from_numpy(sample[1]) for sample in val_ds])
    bt_y = bt_y.to(device)

    loss, pred = model._common_step([bt_x, bt_y, None])
    baseline_loss = torch.mean(loss).item()



    #calculating PFI scores
    B, F = bt_x.shape

    loss_log = torch.zeros((F,permutation_iter))
    for f in tqdm(range(F)):
        for t in range(permutation_iter):
            perm = torch.randperm(B)

            bt_x_shuffled = bt_x.clone()
            bt_x_shuffled[:, f] = bt_x_shuffled[perm, f]

            loss, pred = model._common_step([bt_x_shuffled, bt_y, None])
            loss_log[f,t] = torch.mean(loss).item()

    loss_log = torch.mean(loss_log, dim=1)
    pfi_scores = (loss_log - baseline_loss) / baseline_loss



    #plot configurations
    N = 105

    vi_list = [str(num) for num in np.arange(1,13)]
    cb_list = [str(num) for num in np.arange(13,26)]
    tb_list = [str(num) for num in np.arange(26,39)]
    sc_list = [str(num) for num in np.arange(39,62)]
    sm_list = [str(num) for num in np.arange(62,75)]
    hc_list = [str(num) for num in np.arange(75,106)]

    node_names = vi_list + cb_list + tb_list + sc_list + sm_list + hc_list

    ncolor_base = ['#AEC6CF', '#FFB347', '#77DD77', '#CBAACB', '#FDFD96', '#FF6961']
    node_colors = len(vi_list)*[ncolor_base[0]] + len(cb_list)*[ncolor_base[1]] + len(tb_list)*[ncolor_base[2]] + len(sc_list)*[ncolor_base[3]] + len(sm_list)*[ncolor_base[4]] + len(hc_list)*[ncolor_base[5]]



    #plotting PFI scores in as chord diagram
    con = get_fullFNC(pfi_scores)

    fig, axes = _plot_connectivity_circle(con, node_names=node_names, node_colors=node_colors, facecolor='white', textcolor='black', colormap='Greys', colorbar_size=0.6, linewidth=3, vmin=0)
    fig.savefig("pfi_chord.png", dpi=300, bbox_inches='tight', transparent=True)





