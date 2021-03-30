import numpy as np
import torch
import torch.fft
from torch import nn
from torchinterp1d import Interp1d
from tqdm.auto import tqdm
from numbers import Number


class GaussianModel(nn.Module):

    def __init__(self, sigma, height, fourier_space=False):
        super().__init__()

        if isinstance(sigma, Number):
            sigma = [sigma]

        if isinstance(height, Number):
            height = [height]

        self.sigma = torch.nn.Parameter(data=torch.tensor(sigma, dtype=torch.float32), requires_grad=True)
        self.height = torch.nn.Parameter(data=torch.tensor(height, dtype=torch.float32), requires_grad=True)

        assert len(self.sigma.shape) == 1
        assert len(self.height.shape) == 1
        assert len(self.sigma) == len(self.height)

        self.fourier_space = fourier_space

    def eval_realspace(self, x):
        xdim = len(x.shape)
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.sigma.device, dtype=torch.float32)

        height = self.height / self.height.sum()

        return (torch.exp(-x[..., None] ** 2 / (2 * self.sigma[(None,) * xdim] ** 2)) * height).sum(-1)

    def eval_fourierspace(self, x):
        xdim = len(x.shape)
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.sigma.device, dtype=torch.float32)

        height = self.height / self.height.sum()

        return (torch.exp(-x[..., None] ** 2 * self.sigma[(None,) * xdim] ** 2 * 2 * np.pi ** 2) *
                height[(None,) * xdim] * self.sigma[(None,) * xdim] ** 2 * 2 * np.pi).sum(-1)

    def __call__(self, x):
        if self.fourier_space:
            return self.eval_fourierspace(x)
        else:
            return self.eval_realspace(x)


class InterpolatedModel(nn.Module):

    def __init__(self, nodes, values):
        super().__init__()
        if not torch.is_tensor(nodes):
            nodes = torch.tensor(nodes, dtype=torch.float32)

        self.nodes = nodes

        if not torch.is_tensor(values):
            values = torch.tensor(values, dtype=torch.float32)

        self.values = torch.nn.Parameter(data=values, requires_grad=True)

    def __call__(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.values.device, dtype=torch.float32)
        return Interp1d()(self.nodes, self.values, x)


class ProbeSuperposition(nn.Module):

    def __init__(self, shape, probe_model, positions, intensities=None):
        super().__init__()

        if intensities is None:
            intensities = np.ones(len(positions))

        assert len(intensities) == len(positions)
        self.probe_model = probe_model
        self.positions = torch.nn.Parameter(data=torch.tensor(positions, dtype=torch.float32), requires_grad=True)
        self.intensities = torch.nn.Parameter(data=torch.tensor(intensities, dtype=torch.float32), requires_grad=True)
        self.shape = shape

        kx = np.fft.fftfreq(shape[1])
        ky = np.fft.fftfreq(shape[0])
        self.k = torch.tensor(np.sqrt(kx[None] ** 2 + ky[:, None] ** 2), dtype=torch.float32)

    def forward(self):
        array = torch.zeros(self.shape, device=self.positions.device)
        self.k = self.k.to(self.positions.device)
        self.intensities = self.intensities.to(self.positions.device)

        rounded = torch.floor(self.positions).type(torch.long)
        rows, cols = rounded[:, 1], rounded[:, 0]

        rows = torch.clip(rows, 0, array.shape[0] - 1)
        cols = torch.clip(cols, 0, array.shape[1] - 1)

        array[rows, cols] += (1 - (self.positions[:, 1] - rows)) * (
                1 - (self.positions[:, 0] - cols)) * self.intensities
        array[(rows + 1) % self.shape[0], cols] += (self.positions[:, 1] - rows) * (
                1 - (self.positions[:, 0] - cols)) * self.intensities
        array[rows, (cols + 1) % self.shape[1]] += (1 - (self.positions[:, 1] - rows)) * (
                self.positions[:, 0] - cols) * self.intensities
        array[(rows + 1) % self.shape[0], (cols + 1) % self.shape[1]] += (rows - self.positions[:, 1]) * (
                cols - self.positions[:, 0]) * self.intensities

        array = torch.fft.ifftn(torch.fft.fftn(array, dim=(-2, -1)) * self.probe_model(self.k), dim=(-2, -1))
        return array.real

    def get_loss(self, target, weights=None):
        prediction = self()
        losses = ((prediction - target) ** 2)
        if weights is not None:
            losses *= weights
        return losses.sum()

    def optimize(self, target, optimizers, num_iter, weights=None):

        if weights is not None:
            assert target.shape == weights.shape
            weights = torch.tensor(weights, device=self.positions.device)

        target = torch.tensor(target, device=self.positions.device)

        pbar = tqdm(total=num_iter)
        for i in range(num_iter):
            loss = self.get_loss(target, weights)

            for optimizer in optimizers:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in optimizers:
                optimizer.step()

            pbar.update(1)
            pbar.set_postfix({'loss': loss.detach().item()})

        pbar.close()
