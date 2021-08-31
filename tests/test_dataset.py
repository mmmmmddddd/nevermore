import os

from tests import NYUv2_EXISTS, NYUv2_PATH

import matplotlib.pyplot as plt
import pytest

from nevermore.data.dataset import NYUv2Dataset


@pytest.mark.skipif(not NYUv2_EXISTS, reason="depends on NYUv2")
def test_NYUv2Dataset():
    data_root = NYUv2_PATH
    train_list_file = os.path.join(data_root, "train.txt")
    img_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "segmentation")
    depth_dir = os.path.join(data_root, "depths")
    normal_dir = os.path.join(data_root, "normals")

    assert len(NYUv2Dataset.CLASSES) == 14

    dataset = NYUv2Dataset(
        list_file=train_list_file,
        img_dir=os.path.join(img_dir, "train"),
        mask_dir=os.path.join(mask_dir, "train"),
        depth_dir=os.path.join(depth_dir, "train"),
        normal_dir=os.path.join(normal_dir, "train"),
        input_size=(320, 320),
        output_size=(320, 320),
    )

    print(dataset.get_class_probability())

    sample = dataset[0]
    image, mask = sample['image'], sample['mask']

    image.transpose_(0, 2)

    fig = plt.figure()

    a = fig.add_subplot(1, 2, 1)
    plt.imshow(image)

    a = fig.add_subplot(1, 2, 2)
    plt.imshow(mask)
    # uncomment below if you want take a look
    # plt.show()
