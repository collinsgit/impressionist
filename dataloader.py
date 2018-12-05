import os
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms


# TODO: utilize opts, replacing batch size, patch size, image size, (num workers?)
BATCH_SIZE = 10
NUM_BATCHES = 100
PATCH_SIZE = 8
IMAGE_SIZE = 256


def get_image_loader(data_set, opts):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    data_path = os.path.join('./data/', data_set)

    dataset = datasets.ImageFolder(data_path, transform)

    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    return data_loader


def get_data(data_set, opts):
    for batch, (images, _) in enumerate(get_image_loader(data_set, opts)):
        if batch >= NUM_BATCHES:
            return

        # images = images.view(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)
        dims = (randint(0, IMAGE_SIZE - 2 * (PATCH_SIZE // 2) - 1),
                randint(0, IMAGE_SIZE - 2 * (PATCH_SIZE // 2) - 1))
        patches = images[:, :, dims[0]: dims[0] + PATCH_SIZE, dims[1]: dims[1] + PATCH_SIZE]
        patches = torch.normal(patches, torch.full_like(patches, 0.05))
        labels = torch.Tensor().new_full((BATCH_SIZE,),
                                         (IMAGE_SIZE - 2 * (PATCH_SIZE // 2)) * dims[0] + dims[1],
                                         dtype=torch.long)

        yield images, patches, labels


if __name__ == '__main__':
    pass
