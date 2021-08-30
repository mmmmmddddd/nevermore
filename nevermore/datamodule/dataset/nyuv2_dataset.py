"""NYUv2 Dateset Segmentation Dataloader"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

NYUv2_CLASSES = (
    'background',  # always index 0
    'bed',
    'books',
    'ceiling',
    'chair',
    'floor',
    'furniture',
    'objects',
    'painting',
    'sofa',
    'table',
    'tv',
    'wall',
    'window'
)
NUM_CLASSES = len(NYUv2_CLASSES)


class NYUv2Dateset(Dataset):
    """NYUv2Dateset Dataset
    from https://github.com/say4n/pytorch-segnet/blob/master/src/dataset.py
    """

    def __init__(
        self,
        list_file,
        img_dir,
        mask_dir,
        depth_dir,
        normal_dir,
        input_size,
        output_size,
        transform=None
    ):
        self.images = open(list_file, "rt").read().split("\n")[:-1]
        self.transform = transform

        self.img_extension = ".jpg"
        self.mask_extension = ".png"
        self.depth_extension = ".png"
        self.normal_extension = ".png"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir
        self.depth_root_dir = depth_dir
        self.normal_root_dir = normal_dir

        self.input_size = input_size
        self.output_size = output_size

        self.counts = self.__compute_class_probability()

        self.transform = transforms.Compose([
            # transforms.Resize(input_resolution,interpolation=Image.NEAREST),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(
            self.image_root_dir, name + self.img_extension
        )
        mask_path = os.path.join(
            self.mask_root_dir, name + self.mask_extension
        )
        depth_path = os.path.join(
            self.depth_root_dir, name + self.depth_extension
        )
        normal_path = os.path.join(
            self.normal_root_dir, name + self.normal_extension
        )

        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)
        gt_depth = self.load_depth(path=depth_path)
        gt_normal = self.load_normal(path=normal_path)


        if self.transform:
            image = self.transform(image)
        # image  C * H * W
        data = {
            'image': image,
            'mask': torch.LongTensor(gt_mask),
            'depth': torch.FloatTensor(gt_depth),
            'normal': torch.FloatTensor(gt_normal),
            'image_name': name
        }

        return data

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))

        for name in self.images:
            mask_path = os.path.join(
                self.mask_root_dir, name + self.mask_extension
            )

            raw_image = Image.open(mask_path).resize(self.output_size, Image.NEAREST)
            imx_t = np.array(raw_image).reshape(self.output_size[0] * self.output_size[1])
            imx_t[imx_t == 255] = len(NYUv2_CLASSES)

            for i in range(NUM_CLASSES):
                counts[i] += np.sum(imx_t == i)

        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / np.sum(values)

        return torch.Tensor(p_values)

    def load_image(self, path=None):
        raw_image = Image.open(path)
        # raw_image = np.transpose(raw_image.resize(RESOLUTION, Image.NEAREST), (2, 0, 1))
        # imx_t = np.array(raw_image, dtype=np.float32) / 255.0
        imx_t = raw_image.resize(self.input_size, Image.NEAREST)
        
        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize(self.output_size, Image.NEAREST)
        imx_t = np.array(raw_image)
        # border
        imx_t[imx_t == 255] = len(NYUv2_CLASSES)
        # return H * W
        return imx_t

    def load_depth(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize(self.output_size, Image.NEAREST)
        imx_t = np.array(raw_image)

        return imx_t

    def load_normal(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize(self.output_size, Image.NEAREST)
        imx_t = np.array(raw_image)
        imx_t = np.transpose(imx_t, (2,0,1))
        #return C * H * W
        return imx_t


if __name__ == "__main__":
    data_root = '/data/dixiao.wei/NYU'
    train_list_file = os.path.join(data_root, "train.txt")
    img_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "segmentation")
    depth_dir = os.path.join(data_root, "depths")
    normal_dir = os.path.join(data_root, "normals")

    objects_dataset = NYUv2Dateset(list_file=train_list_file,
                                     img_dir=os.path.join(img_dir,"train"),
                                     mask_dir=os.path.join(mask_dir,"train"),
                                     depth_dir=os.path.join(depth_dir,"train"),
                                     normal_dir=os.path.join(normal_dir,"train"),
                                     )

    print(objects_dataset.get_class_probability())

    sample = objects_dataset[0]
    image, mask = sample['image'], sample['mask']

    image.transpose_(0, 2)

    fig = plt.figure()

    a = fig.add_subplot(1, 2, 1)
    plt.imshow(image)

    a = fig.add_subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()
