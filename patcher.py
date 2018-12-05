from random import randint
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy.stats import multivariate_normal as normal

import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from vgg import VGG
from quadtree import get_energy, build_quad
from color_transfer import swap_colors

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


FEATURES = 64
IMAGE_SIZE = 1024

suggested = 0


def indices(shape, n):
    # selections = np.random.multinomial(n, energy)
    #
    # for i, x in enumerate(selections):
    #     if x:
    #         yield i // shape[-1], i % shape[-1]

    for _ in range(n):
        yield randint(0, shape[-2] - 1), randint(0, shape[-1] - 1)


def pick_patches(patch_feature, style, shape, suggestions, energy, n=500, p=0.9, depth=0):
    sim = nn.CosineSimilarity()

    idx = [(i, j) for i, j in indices(shape, n)]

    styles = style[[i for i, _ in idx], [j for _, j in idx], :]

    dist = sim(styles, patch_feature.view(1, FEATURES))

    this_best = dist.max().item()

    if suggestions:
        sugg_dist = sim(style[[i for i, _ in suggestions], [j for _, j in suggestions], :], patch_feature.view(1, FEATURES))
        best = sugg_dist.max().item()
        if best >= min(0.75, this_best) or energy < 50:
            global suggested
            suggested += 1
            return suggestions[sugg_dist.max(0)[1].item()], best

    if this_best < p and depth < 10:
        other, best = pick_patches(patch_feature, style, shape, None, energy, n, p, depth+1)
        if best > this_best:
            return other, best

    return idx[dist.max(0)[1].item()], this_best


def get_features(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    model = VGG().to(device)
    image = transform(image).to(device)
    image = image.view((1, *image.size()))
    return model(image)


def get_target_patches(quad_tree, target, min_energy):
    dims = {}
    energies = {}
    for dim, size, energy in quad_tree.get_patches(min_energy):
        if size in dims:
            dims[size].append(dim)
            energies[size].append(energy)
        else:
            dims[size] = [dim]
            energies[size] = [energy]

    features = get_features(target)
    factor = int(target.size[-1] / features.size()[-1])
    # avg = nn.AdaptiveMaxPool2d(1)

    patches = {}
    for size in dims:
        avg = nn.LPPool2d(2, size // factor)

        patches[size] = avg(torch.cat([features[:,
                                   :,
                                   dim[0] // factor: (dim[0]+size) // factor,
                                   dim[1] // factor: (dim[1]+size) // factor]
                                   for dim in dims[size]]))

    return dims, patches, energies


def make_suggestions(dim, cache, step=8, n=(-8, 8)):
    # keys = sorted(list(cache.keys()), key=lambda x: abs(x[0] - dim[0]) + abs(x[1] - dim[1]))
    keys = [(dim[0] + step*i, dim[1] + step*j) for i in range(-n[0], n[1]+1) for j in range(-n[0], n[1]+1)
            if (dim[0] + step*i, dim[1] + step*j) in cache]

    suggestions = [(cache[key][0] + dim[0] - key[0], cache[key][1] + dim[1] - key[1]) for key in keys]

    return suggestions


def make_quilt(target, style, patch_dims, target_patches, patch_energies, save_results=True):
    style_features = get_features(style)
    style = transforms.ToTensor()(style)
    style_features = F.interpolate(style_features, style.size()[-2:])

    patches = sum(len(patch_dims[key]) for key in patch_dims)

    final_image = torch.zeros((3, *target.size))

    i = 0
    output_period = 100

    cache = {}

    for size in sorted(target_patches, reverse=True):
        print('Filling in patches of size', size)
        radius = size // 2
        temp_features = style_features[:, :, radius:-radius, radius:-radius]
        features_shape = temp_features.size()
        temp_features = temp_features.permute(2, 3, 1, 0).contiguous().view((*features_shape[2:], FEATURES))

        for dim, patch, energy in zip(patch_dims[size], target_patches[size], patch_energies[size]):
            suggestions = make_suggestions(dim, cache, n=(size//2, 3 * size//2))
            suggestions = list(filter(lambda x: 0 <= x[0] < features_shape[-2] and 0 <= x[1] < features_shape[-1],
                                      suggestions))
            style_dim, _ = pick_patches(patch, temp_features, features_shape, suggestions, energy)
            cache[dim] = style_dim

            final_image[:, dim[0]:dim[0]+size, dim[1]:dim[1]+size] = style[:, style_dim[0]:style_dim[0]+size, style_dim[1]:style_dim[1]+size]

            i += 1
            if i % output_period == 0:
                print('Patches Completed: [{0:}/{1:}]'.format(i, patches))
                # global suggested
                # print('Accepted Suggestions:', suggested)

    if save_results:
        transforms.ToPILImage()(final_image).save('data/personal/stage1.jpg')

    return final_image


def mean_balance(final_image, target, patch_dims, save_results=True):
    for size in sorted(patch_dims, reverse=True):
        for dim in patch_dims[size]:
            patch = final_image[:, dim[0]: dim[0] + size, dim[1]: dim[1] + size]
            patch_mean = patch.mean()
            patch_std = ((patch - patch_mean)**2).mean()**0.5

            original_patch = target[:, dim[0]: dim[0] + size, dim[1]: dim[1] + size]
            original_mean = original_patch.mean()
            original_std = ((original_patch - original_mean)**2).mean()**0.5

            patch -= patch_mean
            patch *= ((0.4 * original_std + 0.6 * patch_std) / patch_std)
            patch += original_mean

    final_image = torch.clamp(final_image, 0., 1.)
    if save_results:
        transforms.ToPILImage()(final_image).save('data/personal/stage2.jpg')

    return final_image


def blend_patches(final_image, patch_dims, save_results=True):
    masks = {}
    for size in sorted(patch_dims, reverse=True):
        for dim in patch_dims[size]:
            radius = size // 8 if size > 16 else size // 4
            patch = final_image[:, dim[0]: dim[0] + size, dim[1]: dim[1] + size]

            super_patch = final_image[:, max(0, dim[0] - radius): min(final_image.size(1) - 1, dim[0] + size + radius),
                          max(0, dim[1] - radius): min(final_image.size(2) - 1, dim[1] + size + radius)]

            super_patch = F.interpolate(super_patch.view((1, *super_patch.size())), (size, size))[0]

            if size not in masks:
                mask = normal.pdf([[i, j] for i in range(size) for j in range(size)],
                                  [(size - 1.) / 2., (size - 1.) / 2.],
                                  size ** 2 * np.eye(2) / 4.)
                mask /= max(mask[:])
                mask = mask.reshape((size, size))
                masks[size] = torch.Tensor(mask)
            mask = masks[size]

            patch = mask * patch + (1 - mask) * super_patch

            final_image[:, dim[0]: dim[0] + size, dim[1]: dim[1] + size] = patch

    final_image = torch.clamp(final_image, 0., 1.)
    if save_results:
        transforms.ToPILImage()(final_image).save('data/personal/stage3.jpg')

    return final_image


def build_texture(target, style, load_from=None):
    energy_map = get_energy(target, blurred=True)
    quad_tree = build_quad(energy_map, min_size=16)

    min_energy = 30 * energy_map.shape[-1]

    print('Organizing Target Patches')
    patch_dims, target_patches, patch_energies = get_target_patches(quad_tree, target, min_energy)

    print('Finding Patch Matches')
    if load_from is None:
        final_image = make_quilt(target, style, patch_dims, target_patches, patch_energies)
    else:
        final_image = transforms.ToTensor()(Image.open(load_from))

    global suggested
    print('Utilized {} Suggested Patches'.format(suggested))

    print('Balancing Patch Means')
    target = transforms.ToTensor()(target)
    final_image = mean_balance(final_image, target, patch_dims)

    print('Blending Patches Together')
    final_image = blend_patches(final_image, patch_dims)

    return final_image


def recolor(target, image):
    t = np.array(target)
    img = np.array(image)
    v = sum(img[:, :, layer] / img.shape[2] for layer in range(img.shape[2])).round()

    hsv = colors.rgb_to_hsv(t)

    hsv[:, :, 2] = v

    rgb = np.array(colors.hsv_to_rgb(hsv)).astype(dtype='uint8')

    return Image.fromarray(rgb)


def style_transfer(target_file, style_file, load_from=None):
    style = Image.open(style_file)
    target = Image.open(target_file)

    greyscale = transforms.Grayscale(3)
    resize = lambda x, c: transforms.Resize((c, c))(transforms.CenterCrop((min(x.size), min(x.size)))(x))
    target = resize(target, IMAGE_SIZE)

    img = build_texture(greyscale(target), greyscale(style), load_from)

    img = transforms.ToPILImage()(img)
    img = recolor(resize(target, IMAGE_SIZE), img)
    img = Image.fromarray(swap_colors(np.array(img), np.array(style)))

    return img


if __name__ == '__main__':
    testing_file = 'buildings'
    target_file = './data/personal/' + testing_file + '.jpg'
    style_file = './data/val/van_gogh/vangogh1.jpg'

    load_from = None
    load_from = './data/personal/stage1.jpg'
    img = style_transfer(target_file, style_file, load_from=load_from)

    img.save('data/personal/' + testing_file + '_output2.jpg')
    plt.imshow(img)
    plt.show()
