# Tutors: Stephanie Käs and Ilya Fradlin
# Task: use normalized central differences (-0.5,0,0.5), positive odd windows and return the numerical flow field.
import cv2
import matplotlib.pyplot as plt
import math
import torch
from glob import glob
from functools import lru_cache
from tqdm import tqdm
import argparse
import flow_vis
from matplotlib import pyplot
import torch.nn


def get_differential_filter():
    """
    get differential filter in both x and y direction
    """

    # === STUDENT SOLUTION GOES HERE ===
    # Central derivative: use [-0.5, 0, 0.5].
    filter_x = None
    filter_y = None
    # ==================================
    diff_kernel = torch.stack((filter_x, filter_y))

    diff_filter = torch.nn.Conv2d(in_channels=1, out_channels=2, padding=1,
                                  kernel_size=3, groups=1, bias=False)
    diff_filter.weight.data = diff_kernel[:, None, :, :]
    diff_filter.weight.requires_grad = False

    return diff_filter


def get_gaussian_filter():
    # https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
    # Set these to whatever you want for your gaussian filter
    kernel_size = 7
    sigma = 1

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=1, out_channels=1, padding=3,
                                kernel_size=kernel_size, groups=1, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def optical_flow(image1, image2, window_size):
    """
    image1 and image2 is a pair of intensity images the optical flow is computed on
    window_size: size of the optical flow window (spatial coherence constraint)
    """
    gaussian_filter = get_gaussian_filter()
    diff_filter = get_differential_filter()

    # look up which shape is expected for the 2D convolutional operator of PyTorch
    # what will be the batch size and channel dimension?

    flow_output = torch.zeros((image1.shape[0], image1.shape[1], 2))
    # === STUDENT SOLUTION GOES HERE ===
    # TODO don't forget about batch and channel dimensions you need to provide for the convolutional operator in PyTorch
    # take a look in the documentation
    # ===========================

    flow_color = flow_vis.flow_to_color(torch.clamp(flow_output, -1, 1).numpy(), convert_to_bgr=False)
    pyplot.imshow(flow_color)
    pyplot.show()
    return flow_output


if __name__ == '__main__':
    image1 = torch.from_numpy(cv2.imread(f"videos/postcard/thumb00263.png", 0)).float()
    image2 = torch.from_numpy(cv2.imread(f"videos/postcard/thumb00264.png", 0)).float()

    optical_flow(image1, image2, 5)
