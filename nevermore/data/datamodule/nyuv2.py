import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from nevermore.data.dataset import NYUv2Dataset


__all__ = ['NYUv2DataModule']


class NYUv2DataModule(pl.LightningDataModule):

    """
    DataModule for NYUV2.
    """

    def __init__(
        self,
        data_root,
        input_size,
        output_size,
        batch_size,
        num_workers,
    ):
        super().__init__()

        self.data_root = data_root
        self.train_list_file = os.path.join(data_root, "train.txt")
        self.val_list_file = os.path.join(data_root, "val.txt")
        self.img_dir = os.path.join(data_root, "images")
        self.mask_dir = os.path.join(data_root, "segmentation")
        self.depth_dir = os.path.join(data_root, "depths")
        self.normal_dir = os.path.join(data_root, "normals")
        self.input_size = input_size
        self.output_size = output_size

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = NYUv2Dataset(
                list_file=self.train_list_file,
                img_dir=os.path.join(self.img_dir, "train"),
                mask_dir=os.path.join(self.mask_dir, "train"),
                depth_dir=os.path.join(self.depth_dir, "train"),
                normal_dir=os.path.join(self.normal_dir, "train"),
                input_size=self.input_size,
                output_size=self.output_size
            )
            self.val_dataset = NYUv2Dataset(
                list_file=self.val_list_file,
                img_dir=os.path.join(self.img_dir, "test"),
                mask_dir=os.path.join(self.mask_dir, "test"),
                depth_dir=os.path.join(self.depth_dir, "test"),
                normal_dir=os.path.join(self.normal_dir, "test"),
                input_size=self.input_size,
                output_size=self.output_size
            )
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = NYUv2Dataset(
                list_file=self.val_list_file,
                img_dir=os.path.join(self.img_dir, "test"),
                mask_dir=os.path.join(self.mask_dir, "test"),
                depth_dir=os.path.join(self.depth_dir, "test"),
                normal_dir=os.path.join(self.normal_dir, "test"),
                input_size=self.input_size,
                output_size=self.output_size
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
