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
PATCH_SIZE = 5
IMAGE_SIZE = 128


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
    for images, _ in get_image_loader(data_set, opts):
        # images = images.view(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)
        dims = (randint(0, IMAGE_SIZE - PATCH_SIZE), randint(0, IMAGE_SIZE - PATCH_SIZE))
        patches = images[:, :, dims[0]: dims[0] + PATCH_SIZE, dims[1]: dims[1] + PATCH_SIZE]
        labels = torch.Tensor().new_full((BATCH_SIZE,), (IMAGE_SIZE - PATCH_SIZE) * dims[0] + dims[1], dtype=torch.long)

        yield images, patches, labels


if __name__ == '__main__':
    pass
