import logging
import os

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import functional as F

from nevermore.data import NYUv2DataModule
from nevermore.model.network import MultiSegNet
from nevermore.metric import Abs_CosineSimilarity


__all__ = ['MultiSegnetNYUv2Model']

logger = logging.getLogger(__name__)

class MultiSegnetNYUv2Model(pl.LightningModule):

    def __init__(
        self,
        learning_rate,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.segnet = MultiSegNet(
            input_channels=3,
            seg_output_channels=NUM_CLASSES,
            dep_output_channels=1,
            nor_output_channels=3
        )

        self.miou = torchmetrics.IoU(num_classes=NUM_CLASSES, ignore_index=0)
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.cos = Abs_CosineSimilarity(reduction='abs')

    def forward(self, x):
        return self.segnet.forward(x)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y_seg_hat, y_dep_hat, y_nor_hat, y_seg_softmax = self(x)

        y_seg = batch['mask']
        y_dep = batch['depth']
        y_nor = batch['normal'].flatten(start_dim=1)

        loss_seg = F.cross_entropy(y_seg_softmax, y_seg)
        y_dep_hat = y_dep_hat.squeeze()
        loss_dep = F.mse_loss(y_dep_hat, y_dep)
        y_nor_hat = y_nor_hat.flatten(start_dim=1)
        loss_nor = torch.mean(F.cosine_similarity(y_nor_hat, y_nor))

        loss = loss_seg + loss_dep + loss_nor

        self.log('train_loss', loss)
        self.log('train_loss_seg', loss_seg, prog_bar=True)
        self.log('train_loss_dep', loss_dep, prog_bar=True)
        self.log('train_loss_nor', loss_nor, prog_bar=True)

        return loss

    def training_epoch_end(self, training_step_outputs):
        print(self.trainer.lr_schedulers[0]['scheduler'].get_lr())
        for out in training_step_outputs:
            pass

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y_seg_hat, y_dep_hat, y_nor_hat, y_seg_softmax = self(x)

        y_seg = batch['mask']
        y_dep = batch['depth']
        y_nor = batch['normal'].flatten(start_dim=1)

        loss_seg = F.cross_entropy(y_seg_softmax, y_seg)
        y_dep_hat = y_dep_hat.squeeze()
        loss_dep = F.mse_loss(y_dep_hat, y_dep)
        y_nor_hat = y_nor_hat.flatten(start_dim=1)
        loss_nor = torch.mean(F.cosine_similarity(y_nor_hat, y_nor))

        self.log('val_seg_loss', loss_seg)
        self.log('val_seg_iou_step', self.miou(y_seg_softmax, y_seg))
        self.log('val_dep_loss', loss_dep)
        self.log('val_dep_rmse_step', self.rmse(y_dep_hat, y_dep))
        self.log('val_nor_loss', loss_nor)
        self.log('val_dep_cos_step', self.cos(y_nor_hat, y_nor))

    def validation_epoch_end(self, validation_step_outputs):

        val_miou = self.miou.compute()
        self.log('val_seg_iou', val_miou)
        logger.info("val_seg_iou:" + str(val_miou.item()))
        self.miou.reset()

        val_rmse = self.rmse.compute()
        self.log('val_dep_mse', val_rmse)
        logger.info("val_dep_mse:" + str(val_rmse.item()))
        self.rmse.reset()

        val_cos = self.cos.compute()
        self.log('val_nor_cos', val_cos)
        logger.info("val_nor_cos:" + str(val_cos.item()))
        self.cos.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.segnet.parameters(), lr=self.hparams.learning_rate
        )
        lr_schedule = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=462, gamma=0.2
        )
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': lr_schedule}
        return optim_dict
