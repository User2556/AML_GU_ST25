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






class MLPClassifier(BaseClassifier):
    def __init__(self, width_list, num_in_feat, num_out_classes, **kwargs):
        super().__init__(**kwargs)

        self.width_list = width_list

        self.num_in_feat = num_in_feat
        self.num_out_classes = num_out_classes

        self._build_model()


    def _stack_segment(self, width_list):
        
        layers = []
        for i in range(len(width_list) - 1): #iterating over layer widths
            in_dim, out_dim = width_list[i], width_list[i + 1]

            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(width_list) - 2:
                layers.append(nn.ReLU())
                    
        return nn.Sequential(*layers)


    def _build_model(self):
        self.classif_model = self._stack_segment(self.width_list)

        
    def forward(self, x):
        x = self.classif_model(x)
        return x





