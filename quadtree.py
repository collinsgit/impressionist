import numpy as np
from PIL import Image
from scipy import ndimage
from torchvision import transforms


class QuadTree:
    def __init__(self, energy, dims, size, components):
        self.energy = energy
        self.dims = dims
        self.size = size
        self.components = components

    def get_patches(self, min_energy=15000):
        if self.energy < min_energy or self.components is None:
            yield self.dims, self.size, self.energy / self.size**2
        else:
            for component in self.components:
                yield from component.get_patches(min_energy)


def get_energy(image, blurred=False):
    img = np.array(image)
    img = sum(img[:, :, i] / img.shape[2] for i in range(img.shape[2]))

    sx = ndimage.sobel(img, axis=0)
    sy = ndimage.sobel(img, axis=1)
    energy = np.hypot(sx, sy)

    if blurred:
        energy = ndimage.filters.gaussian_filter(energy, 0.5)

    for i in range(energy.shape[0]):
        for j in range(energy.shape[1]):
            if energy[i, j] > 50:
                energy[i, j] = 255
            else:
                energy[i, j] = 0

    return energy


def build_quad(energy_map, dims=(0, 0), min_size=8):
    if min(energy_map.shape) == min_size:
        return QuadTree(energy_map.sum(), dims, min_size, None)

    i_2, j_2 = map(lambda x: x // 2, energy_map.shape)

    components = (build_quad(energy_map[:i_2, :j_2], (dims[0], dims[1]), min_size),
                  build_quad(energy_map[:i_2, j_2:], (dims[0], dims[1] + j_2), min_size),
                  build_quad(energy_map[i_2:, :j_2], (dims[0] + i_2, dims[1]), min_size),
                  build_quad(energy_map[i_2:, j_2:], (dims[0] + i_2, dims[1] + j_2), min_size))

    return QuadTree(sum(comp.energy for comp in components),
                    dims,
                    2 * components[0].size,
                    components)
