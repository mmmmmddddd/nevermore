import logging

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import functional as F

from nevermore.data import NYUv2Dataset
from nevermore.metric import Abs_CosineSimilarity
from nevermore.model.network import SingleSegNet

__all__ = ['SingleSegnetNYUv2Model']

logger = logging.getLogger(__name__)


class SingleSegnetNYUv2Model(pl.LightningModule):

    def __init__(self, learning_rate, task):
        super().__init__()
        self.save_hyperparameters()

        allowed_task = ("segmentation", "depth", "normal")
        if task not in allowed_task:
            raise ValueError(
                f"Expected argument `tsak` to be one of "
                f"{allowed_task} but got {task}"
            )
        task_output_channels = dict(
            segmentation=len(NYUv2Dataset.CLASSES),
            depth=1,
            normal=3,
        )

        self.task = task
        self.segnet = SingleSegNet(
            input_channels=3,
            output_channels=task_output_channels[task],
        )

        self.miou = torchmetrics.IoU(
            num_classes=len(NYUv2Dataset.CLASSES), ignore_index=0
        )
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.cos = Abs_CosineSimilarity(reduction='abs')

    def forward(self, x):
        return self.segnet.forward(x)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y_hat, y_hat_softmax = self(x)

        if self.task == 'segmentation':
            y_seg = batch['mask']
            loss_seg = F.cross_entropy(y_hat_softmax, y_seg)
            loss = loss_seg
            self.log('train_loss_seg', loss, prog_bar=True)

        if self.task == 'depth':
            y_dep = batch['depth']
            y_hat = y_hat.squeeze()
            loss_dep = F.mse_loss(y_hat, y_dep)
            loss = loss_dep
            self.log('train_loss_dep', loss, prog_bar=True)

        if self.task == 'normal':
            y_nor = batch['normal'].flatten(start_dim=1)
            y_hat = y_hat.flatten(start_dim=1)
            loss_nor = torch.mean(F.cosine_similarity(y_hat, y_nor))
            loss = loss_nor
            self.log('train_loss_nor', loss, prog_bar=True)

        return loss

    def training_epoch_end(self, training_step_outputs):
        print(self.trainer.lr_schedulers[0]['scheduler'].get_lr())
        for out in training_step_outputs:
            pass

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y_hat, y_hat_softmax = self(x)

        if self.task == 'segmentation':
            y_seg = batch['mask']
            loss_seg = F.cross_entropy(y_hat_softmax, y_seg)
            self.log('val_seg_loss', loss_seg)
            self.log('val_seg_iou_step', self.miou(y_hat_softmax, y_seg))

        if self.task == 'depth':
            y_dep = batch['depth']
            y_hat = y_hat.squeeze()
            loss_dep = F.mse_loss(y_hat, y_dep)
            self.log('val_dep_loss', loss_dep)
            self.log('val_dep_rmse_step', self.rmse(y_hat, y_dep))

        if self.task == 'normal':
            y_nor = batch['normal'].flatten(start_dim=1)
            y_hat = y_hat.flatten(start_dim=1)
            loss_nor = torch.mean(F.cosine_similarity(y_hat, y_nor))
            self.log('val_nor_loss', loss_nor)
            self.log('val_dep_cos_step', self.cos(y_hat, y_nor))

    def validation_epoch_end(self, validation_step_outputs):

        if self.task == 'segmentation':
            val_miou = self.miou.compute()
            self.log('val_seg_iou', val_miou)
            logger.info("val_seg_iou:" + str(val_miou.item()))
            self.miou.reset()

        if self.task == 'depth':
            val_rmse = self.rmse.compute()
            self.log('val_dep_mse', val_rmse)
            logger.info("val_dep_mse:" + str(val_rmse.item()))
            self.rmse.reset()

        if self.task == 'normal':
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
