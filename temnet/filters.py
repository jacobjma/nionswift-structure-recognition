from numbers import Number
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2) ** 2 / float(2 * sigma ** 2)

    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()


def get_gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    #if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
    #    raise TypeError("ksize must be an odd positive integer. Got {}"
    #                    .format(ksize))
    window_1d: torch.Tensor = gaussian(ksize, sigma)
    return window_1d


def get_gaussian_kernel2d(ksize: Tuple[int, int], sigma: Tuple[float, float]):
    #if not isinstance(ksize, tuple) or len(ksize) != 2:
    #    raise TypeError("ksize must be a tuple of length two. Got {}".format(ksize))
    #if not isinstance(sigma, tuple) or len(sigma) != 2:
    #    raise TypeError("sigma must be a tuple of length two. Got {}".format(sigma))

    ksize_x, ksize_y = ksize
    sigma_x, sigma_y = sigma
    kernel_x = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y = get_gaussian_kernel(ksize_y, sigma_y)

    kernel_x = kernel_x.unsqueeze(-1)
    kernel_y = kernel_y.unsqueeze(-1).t()
    return kernel_x, kernel_y


class GaussianFilter(nn.Module):
    def __init__(self, sigma) -> None:
        super(GaussianFilter, self).__init__()

        if isinstance(sigma, Number):
            sigma = (sigma,) * 2

        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=False)

    @staticmethod
    def compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
        """Computes zero padding tuple."""
        computed = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]

    def forward(self, x: torch.Tensor):
        kernel_size = ((2 * torch.ceil(self.sigma[0] * 2.) + 1).type(torch.long),
                       (2 * torch.ceil(self.sigma[1] * 2.) + 1).type(torch.long))

        if not torch.is_tensor(x):
            raise TypeError("Input x type is not a torch.Tensor. Got {}"
                            .format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(x.shape))
        # prepare kernel
        b, c, h, w = x.shape
        kernel_x, kernel_y = get_gaussian_kernel2d(kernel_size, self.sigma)
        kernel_x = kernel_x.to(x.device).to(x.dtype)
        kernel_y = kernel_y.to(x.device).to(x.dtype)
        kernel_x = kernel_x.repeat(c, 1, 1, 1)
        kernel_y = kernel_y.repeat(c, 1, 1, 1)

        padding = self.compute_zero_padding(kernel_size)

        x = F.pad(x, pad=(padding[1],) * 2 + (padding[0],) * 2, mode='constant', value=x.mean())
        x = F.conv2d(F.conv2d(x, kernel_x, stride=1, groups=c), kernel_y, stride=1, groups=c)
        return x


def gaussian_filter(src, sigma, kernel_size=None):
    return GaussianFilter(sigma, kernel_size)(src)


class DoubleGaussianFilter(nn.Module):

    def __init__(self, sigma1, sigma2):
        super().__init__()

        self.gaussian_filter1 = GaussianFilter(sigma1)
        self.gaussian_filter2 = GaussianFilter(sigma2)

    def forward(self, x):
        return self.gaussian_filter1(x) - self.gaussian_filter2(x)


class PeakEnhancementFilter(nn.Module):

    def __init__(self, alpha, sigma, iterations):
        super().__init__()
        self.gaussian_filter = GaussianFilter(sigma)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.iterations = nn.Parameter(torch.tensor(iterations), requires_grad=False)

    def forward(self, tensor):
        temp = tensor.clone()
        temp2 = self.gaussian_filter(temp)

        for i in range(self.iterations):
            temp = temp ** self.alpha
            temp *= temp2 / (self.gaussian_filter(temp) + 1e-7)

        return temp
