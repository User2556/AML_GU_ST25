import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchmetrics.classification import BinaryAUROC
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import torchvision




###  AE-MODEL:  FEATURE EXTRACTION  ###


class BaseFExtr(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
    
        self.lr = learning_rate

        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        loss, pred = self._common_step(batch)
        self.log("Loss/TRAINING", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred = self._common_step(batch)
        self.log("Loss/VALIDATION", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def _common_step(self, batch):
        x, y, _ = batch

        pred = self.forward(x)
        loss = self.loss_fn(pred, x)
        return loss, pred

    def predict_step(self, batch, batch_idx=None):
        loss, pred = self._common_step(batch)
        class_pred = torch.argmax(pred)
        return loss, class_pred

    def configure_optimizers(self):
        return {
            "optimizer": torch.optim.Adam(self.parameters(), lr=self.lr),
        }




class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.use_projection = in_dim != out_dim
        self.proj = nn.Linear(in_dim, out_dim) if self.use_projection else nn.Identity()

        layers = []
        
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.LazyBatchNorm1d())
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(out_dim, out_dim))
        layers.append(nn.LazyBatchNorm1d())
        
        self.sub_resblock = nn.Sequential(*layers)
        

    def forward(self, x):
        residual = self.proj(x)

        out = self.sub_resblock(x)
        out = F.relu(out + residual)
        return out



class AE(BaseFExtr):
    def __init__(self, pre_width_list, repr_dimension, num_in_feat, **kwargs):
        super().__init__(**kwargs)

        self.pre_width_list = pre_width_list
        self.repr_dimension = repr_dimension

        self.num_feat = num_in_feat

        self._build_model()


    def _stack_segment(self, width_list):
        layers = []
        for i in range(len(width_list) - 1): #iterating over layer widths
            in_dim, out_dim = width_list[i], width_list[i + 1]
            
            layers.append(ResidualBlock(in_dim, out_dim))
            
        return nn.Sequential(*layers)


    def _build_model(self):
        width_list = self.pre_width_list + [self.repr_dimension]

        self.encoder = self._stack_segment(width_list)
        self.decoder = self._stack_segment(width_list[::-1])


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




































###  MLP-MODEL:  FEATURE CLASSIFICATION  ###






class BaseClassifier(pl.LightningModule):
    def __init__(self, learning_rate, sparsity):
        super().__init__()
        self.lr = learning_rate
        self.sparsity = sparsity

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.acc_fn = BinaryAUROC()

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        loss, pred = self._common_step(batch)
        acc = self.acc_fn(pred, y)
        
        #implement L1 regularization
        l1_loss = sum(param.abs().sum() for param in self.parameters())
        loss += self.sparsity * l1_loss
        
        self.log("Loss/TRAINING", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Accuracy/TRAINING", acc, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        loss, pred = self._common_step(batch)
        acc = self.acc_fn(pred, y)
        self.log("Loss/VALIDATION", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Accuracy/VALIDATION", acc, on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
    def _common_step(self, batch):
        x, y, _ = batch

        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        return loss, pred

    def predict_step(self, batch, batch_idx=None):
        loss, pred = self._common_step(batch)
        class_pred = torch.argmax(pred)
        return loss, class_pred

    def configure_optimizers(self):
        return {
            "optimizer": torch.optim.Adam(self.parameters(), lr=self.lr),
        }










class AE_MLPClassifier(BaseClassifier):
    def __init__(self, encoder_info, width_list, num_in_feat, num_out_classes, **kwargs):
        super().__init__(**kwargs)

        self.encoder_info = encoder_info #[ckpt_path, width_list]

        self.width_list = width_list

        self.num_in_feat = num_in_feat
        self.num_out_classes = num_out_classes

        self._build_model()

        
    
    def _get_encoder(self):
        ae_model = AE.load_from_checkpoint(
            checkpoint_path=self.encoder_info[0],
            pre_width_list=self.encoder_info[1][:-1],
            repr_dimension=self.encoder_info[1][-1],
            num_in_feat = self.num_in_feat,
            learning_rate = self.lr
            )
        
        encoder = ae_model.encoder

        for param in encoder.parameters():
            param.requires_grad = False
        
        encoder.eval()
        return encoder


    def _stack_segment(self, width_list):

        layers = []
        for i in range(len(width_list) - 1): #iterating over layer widths
            in_dim, out_dim = width_list[i], width_list[i + 1]

            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(width_list) - 2:
                layers.append(nn.ReLU())
                    
        return nn.Sequential(*layers)



    def _build_model(self):
        self.fEngin_model = self._get_encoder()
        self.classif_model = self._stack_segment(self.width_list)

        
    def forward(self, x):
        x = self.fEngin_model(x)
        x = self.classif_model(x)
        return x

