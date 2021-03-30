from numbers import Number

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm


def lowpass_noise(shape, cutoff):
    if isinstance(cutoff, Number):
        cutoff = (cutoff,) * 2

    kx = np.fft.fftfreq(shape[0]) / cutoff[0] / .5
    ky = np.fft.fftfreq(shape[1]) / cutoff[1] / .5
    k = np.sqrt(kx[None] ** 2 + ky[:, None] ** 2)
    r = np.random.rand(*k.shape).astype(np.float32)
    mask = (k < 1).astype(np.float32)
    noise = np.fft.fftn(mask * np.exp(-1.j * r * 2 * np.pi), axes=tuple(range(len(k.shape))))
    noise = (noise.real + noise.imag) / 2
    return (noise / (np.std(noise) + 1e-6))


def grid_sample(array, positions):
    k = torch.tensor(((array.shape[1] - 1) / 2, (array.shape[0] - 1) / 2), device=array.device)
    positions = (positions - k) / k
    return F.grid_sample(array[None, None], positions[None], align_corners=True, padding_mode='border')[0, 0]


def index_grid(shape, dtype=torch.long, device=None):
    y = torch.arange(shape[0], dtype=dtype, device=device)
    x = torch.arange(shape[1], dtype=dtype, device=device)
    return torch.stack(torch.meshgrid(y, x)[::-1], -1)


def disc_index_meshgrid(radius, dtype=torch.long, device=None):
    cols = torch.zeros((2 * radius + 1, 2 * radius + 1), dtype=dtype, device=device)
    cols[:] = torch.linspace(0, 2 * radius, 2 * radius + 1, dtype=dtype) - radius
    rows = cols.T
    inside = (rows ** 2 + cols ** 2) <= radius ** 2
    return torch.stack((cols[inside], rows[inside]), -1)


class ScanDistortion(nn.Module):

    def __init__(self, displacements_x=None, displacements_y=None):
        super().__init__()

        if displacements_x is not None:
            if not torch.is_tensor(displacements_x):
                displacements_x = torch.from_numpy(displacements_x)

            displacements_x = displacements_x.type(torch.float32)
            displacements_x = nn.Parameter(data=displacements_x, requires_grad=True)

        if (displacements_y is not None):
            if not torch.is_tensor(displacements_y):
                displacements_y = torch.from_numpy(displacements_y)

            displacements_y = displacements_y.type(torch.float32)

        self.displacements_x = nn.Parameter(data=displacements_x, requires_grad=True)
        self.displacements_y = nn.Parameter(data=displacements_y, requires_grad=True)

    def apply(self, image):
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)
            input_is_numpy = True
        else:
            input_is_numpy = False

        image = image.to(self.device)
        image = image.type(torch.float32)
        displacements = index_grid(image.shape, dtype=torch.float32)
        with torch.no_grad():
            if len(self.displacements_x) > 0:
                displacements[..., 0] += self.displacements_x
            if len(self.displacements_y) > 0:
                displacements[..., 1] += self.displacements_y

        distorted = grid_sample(image, displacements)
        if input_is_numpy:
            return distorted.detach().cpu().numpy()
        else:
            return distorted

    @property
    def device(self):
        return self.displacements_x.device

    def unapply(self, image, rbf_sigma=.1):
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)

        image = image.to(self.device)

        query_grid = self._rbf_interpolation_queries(image.shape, radius=int(np.ceil(rbf_sigma)), device=self.device)
        return self._unapply(image, query_grid, rbf_sigma).detach().cpu().numpy()

    def _unapply(self, image, query_grid, rbf_sigma):
        optimized_probe_positions = index_grid(image.shape, dtype=torch.float32, device=self.device)
        optimized_probe_positions[..., 0] = optimized_probe_positions[..., 0] - self.displacements_x[:, None]
        optimized_values = grid_sample(image, optimized_probe_positions)

        pixel_positions = optimized_probe_positions.clone()

        optimized_probe_positions = optimized_probe_positions[query_grid[..., 1], query_grid[..., 0]]
        optimized_values = optimized_values[query_grid[..., 1], query_grid[..., 0]]

        kernels = torch.exp(
            -.5 * ((pixel_positions[..., None, :] - optimized_probe_positions) ** 2).sum(-1) / rbf_sigma ** 2)
        kernels = kernels / kernels.sum(-1, keepdims=True)
        return (kernels * optimized_values).sum(-1)

    def _rbf_interpolation_queries(self, shape, radius, device):
        query_grid = index_grid(shape, device=device)[:, :, None] + \
                     disc_index_meshgrid(radius, device=device)[None, None]
        query_grid[..., 0] = torch.clip(query_grid[..., 0], 0, shape[1] - 1)
        query_grid[..., 1] = torch.clip(query_grid[..., 1], 0, shape[0] - 1)
        return query_grid

    def fit(self, image, num_iterations, horizontal_smoothing=5, l2_regularization=None, rbf_sigma=.1):
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)

        image = image.to(self.device)

        if len(self.displacements_x) == 0:
            self.displacements_x.data = torch.zeros(image.shape[0], device=self.device)

        f = DirectionalGaussianFilter2d(horizontal_smoothing).to(self.device)
        image = f(image[None, None])[0, 0]

        query_grid = self._rbf_interpolation_queries(image.shape, radius=int(np.ceil(rbf_sigma)), device=self.device)
        optimizer = torch.optim.Adam([self.displacements_x], lr=200e-2)

        pbar = tqdm(total=num_iterations)
        for i in range(num_iterations):
            rbf_interp = self._unapply(image, query_grid, rbf_sigma)

            optimizer.zero_grad()

            loss = ((rbf_interp[1:] - rbf_interp[:-1]) ** 2).sum()
            if l2_regularization:
                loss += l2_regularization * (self.displacements_x ** 2).sum()

            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix({'loss': loss.detach().item()})
