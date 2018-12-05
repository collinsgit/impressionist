import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

magic1 = np.array([[0.3811, 0.5783, 0.0402],
                   [0.1967, 0.7244, 0.0782],
                   [0.0241, 0.1288, 0.8444]])
magic2 = np.array([[(1./np.sqrt(3)), 0., 0.],
                   [0., (1./np.sqrt(6)), 0.],
                   [0., 0., (1/np.sqrt(2))]]).dot(([[1., 1., 1.],
                                                    [1., 1., -2.],
                                                    [1., -1., 0.]]))


eps = 1e-5


def rgb_to_lab(img):
    img = img.dot(magic1.T)
    img = np.log(img + eps)
    img = img.dot(magic2.T)
    return img


def lab_to_rgb(img):
    img = img.dot(np.linalg.inv(magic2).T)
    img = np.e ** img
    img = img.dot(np.linalg.inv(magic1).T)
    return img


def get_stats(img):
    means = []
    stds = []

    for layer in range(img.shape[-1]):
        mean = img[:, :, layer].mean()
        means.append(mean)

        var = ((img[:, :, layer] - mean)**2).mean()
        stds.append(np.sqrt(var))

    return means, stds


def swap_colors(target, style, std_factor=1):
    target = rgb_to_lab(target)
    style = rgb_to_lab(style)

    t_means, t_stds = get_stats(target)
    s_means, s_stds = get_stats(style)

    for layer in range(target.shape[-1]):
        target[:, :, layer] -= t_means[layer]
        target[:, :, layer] *= (std_factor * s_stds[layer] / t_stds[layer])
        target[:, :, layer] += s_means[layer]

    result = lab_to_rgb(target)
    result = np.vectorize(lambda x: min(255, max(0, x)))(result)
    return result.round().astype('uint8')


if __name__ == '__main__':
    style = Image.open('./data/val/pollock/pollock13.jpg')
    target = Image.open('./data/val/picasso/picasso3.jpg')

    target = swap_colors(np.array(target), np.array(style))

    img = Image.fromarray(target)

    plt.imshow(img)
    plt.show()
