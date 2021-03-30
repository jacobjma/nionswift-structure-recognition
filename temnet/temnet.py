import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import bisect

import numpy as np
import pytorch_lightning as pl
from scipy import ndimage
from skimage.transform import rescale
from tqdm.auto import tqdm

from temnet.filters import PeakEnhancementFilter, GaussianFilter
from temnet.metrics import precision_and_recall
from temnet.r2unet import ConvHead, R2UNet
from temnet.utils import as_rgb_image, draw_points, pad_to_size, unpad, generate_batches, calculate_padding
from skimage.measure import find_contours, approximate_polygon
from skimage.morphology import binary_closing, disk, remove_small_objects
from torchgeometry.losses import FocalLoss
import json
import os
from pathlib import Path


def insort_bottom_k(sorted_list, x, k):
    bisect.insort(sorted_list, x)
    if len(sorted_list) > k:
        del sorted_list[-1]


class TEMNetMetrics:

    def __init__(self, tolerance, max_examples):
        self.tolerance = tolerance
        self.max_examples = max_examples
        self.metrics = []
        self.examples = {'data': [], 'scores': []}

    def collect(self, temnet, batch, density_logits, segmentation_logits=None):
        densities = nn.Sigmoid()(density_logits)

        if segmentation_logits is not None:
            segmentations = nn.Softmax(dim=1)(segmentation_logits)
        else:
            segmentations = None

        densities = densities * segmentations[:, 0, None]
        centers, _ = temnet.calculate_density_centers(densities)

        precision = 0.
        recall = 0.
        f_score = 0.
        accuracy = 0.
        for i in range(len(batch['images'])):

            try:
                if len(centers[i]) > 0:
                    rows, cols = np.round(centers[i]).astype(np.int).T
                    mask = batch['weights'][i].cpu().numpy()[0, cols, rows]
                else:
                    mask = None
            except KeyError:
                mask = None

            try:
                mask_true = batch['centers_mask'][i]
            except KeyError:
                mask_true = None

            metrics, indices = precision_and_recall(centers[i],
                                                    batch['centers'][i],
                                                    mask_true=mask_true,
                                                    mask=mask,
                                                    cutoff=self.tolerance,
                                                    return_indices=True)

            new_precision, new_recall, new_f_score = metrics
            true_positives, false_positives, false_negatives = indices

            precision += new_precision
            recall += new_recall
            f_score += new_f_score
            accuracy += new_f_score == 1.

            example_index = bisect.bisect(self.examples['scores'], new_f_score)
            if example_index < self.max_examples:
                example_data = {'image': batch['images'][i][0].detach().cpu().numpy(),
                                'density': densities[i][0].detach().cpu().numpy(),
                                'false_positives': centers[i][false_positives],
                                'false_negatives': batch['centers'][i][false_negatives]}

                if segmentations is not None:
                    example_data['segmentation'] = np.argmax(segmentations[i].detach().cpu().numpy(), axis=0) / 2

                self.examples['scores'].insert(example_index, new_f_score)
                self.examples['data'].insert(example_index, example_data)

                self.examples['scores'] = self.examples['scores'][:self.max_examples]
                self.examples['data'] = self.examples['data'][:self.max_examples]

        precision /= len(batch['images'])
        recall /= len(batch['images'])
        f_score /= len(batch['images'])
        accuracy /= len(batch['images'])
        self.metrics += [{'precision': precision, 'recall': recall, 'f_score': f_score, 'accuracy': accuracy}]

    def log(self, phase, idx, logger):
        avg_precision = torch.tensor([x['precision'] for x in self.metrics]).mean()
        avg_recall = torch.tensor([x['recall'] for x in self.metrics]).mean()
        avg_f_score = torch.tensor([x['f_score'] for x in self.metrics]).mean()
        avg_accuracy = torch.tensor([x['accuracy'] for x in self.metrics]).mean()

        logger.experiment.add_scalar(f'precision/{phase}', avg_precision, idx)
        logger.experiment.add_scalar(f'recall/{phase}', avg_recall, idx)
        logger.experiment.add_scalar(f'f_score/{phase}', avg_f_score, idx)
        logger.experiment.add_scalar(f'accuracy/{phase}', avg_accuracy, idx)

        for i, example in enumerate(self.examples['data']):
            image = as_rgb_image(example['image'])
            logger.experiment.add_image(f'image_{phase}/{i}', image, idx, dataformats='HWC')
            image = draw_points(image, example['false_negatives'], 3, [255, 0, 0])
            image = draw_points(image, example['false_positives'], 3, [0, 255, 0])
            logger.experiment.add_image(f'overlay_{phase}/{i}', image, idx, dataformats='HWC')

            density = as_rgb_image(example['density'] * 255, normalize=False)
            logger.experiment.add_image(f'density_{phase}/{i}', density, idx, dataformats='HWC')

            if example['segmentation'] is not None:
                segmentation = as_rgb_image(example['segmentation'] * 255, normalize=False)
                logger.experiment.add_image(f'segmentation_{phase}/{i}', segmentation, idx, dataformats='HWC')

    def reset(self):
        self.metrics = []
        self.examples = {'data': [], 'scores': []}


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=3, gamma=2, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        # if self.alpha is None:
        #     self.alpha = torch.ones(2)
        # elif isinstance(self.alpha, (list, np.ndarray)):
        #     self.alpha = np.asarray(self.alpha)
        #     self.alpha = np.reshape(self.alpha, (2))
        #     assert self.alpha.shape[0] == 2, \
        #         'the `alpha` shape is not match the number of class'
        # elif isinstance(self.alpha, (float, int)):
        #     self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        # else:
        #     raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -torch.sum(pos_weight * torch.log(prob)) / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * torch.sum(neg_weight * F.logsigmoid(-output)) / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss

        return loss


class TEMNet(pl.LightningModule):

    def __init__(self,
                 density_net=None,
                 segmentation_net=None,
                 peak_enhancement=None,
                 prefilter=None,
                 gaussian_width=2.5,
                 detection_threshold=.5,
                 train_sampling=.1,
                 margin=0.,
                 recurrent_normalization=True,
                 min_segment_area=0.):

        super().__init__()

        self.density_net = density_net
        self.segmentation_net = segmentation_net

        self.density_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.segmentation_loss = FocalLoss(alpha=.9, reduction='none')
        # nn.CrossEntropyLoss(reduction='none')

        self.peak_enhancement = peak_enhancement
        self.prefilter = prefilter

        self.gaussian_width = gaussian_width
        self.detection_threshold = detection_threshold
        self.train_sampling = train_sampling
        self.margin = margin
        self.min_segment_area = min_segment_area
        self.recurrent_normalization = recurrent_normalization

    def calculate_density_centers(self, density):
        with torch.no_grad():
            peaks = self.peak_enhancement(density)
            peaks = peaks[:, 0].detach().cpu()

        adjusted_threshold = (self.gaussian_width * np.sqrt(2 * np.pi)) ** 2 * self.detection_threshold

        centers = []
        sums = []
        for i in range(len(peaks)):
            labels, n = ndimage.label(peaks[i] > 1e-6)
            new_sums = ndimage.sum(peaks[i], labels, np.arange(1, n))
            new_centers = np.array(ndimage.center_of_mass(peaks[i], labels, np.arange(1, n)))

            if len(new_centers) > 0:
                new_centers = new_centers[:, ::-1]
                is_detection = new_sums > adjusted_threshold
                new_centers = new_centers[is_detection]
                new_sums = new_sums[is_detection]

            sums.append(new_sums)
            centers.append(new_centers)
        return centers, sums

    def calculate_segments(self, segmentations, padding, scale_factor):
        segmentations = segmentations[:].detach().cpu()
        # segmentations = (segmentations > .5).cpu().numpy()

        new_contours = []
        contour_classes = []
        for segmentation in segmentations:
            new_contours.append([])
            contour_classes.append([])
            for i in range(1, self.segmentation_net.out.out_channels):
                class_segmentation = segmentation == i
                class_segmentation = binary_closing(class_segmentation, disk(2))
                class_segmentation = np.pad(class_segmentation, [(1, 1), (1, 1)])
                class_segmentation = remove_small_objects(class_segmentation,
                                                          self.min_segment_area / self.train_sampling ** 2)
                contours = find_contours(class_segmentation, .5)
                contours = [approximate_polygon(contour, .5) for contour in contours]
                contours = [(contour[:, ::-1] - (padding[0], padding[2])) / scale_factor for contour in contours]
                new_contours[-1] += contours
                contour_classes[-1] += [i] * len(contours)

        return new_contours, contour_classes

    def _calculate_padding_and_scale_factor(self, old_shape, old_sampling):
        scale_factor = old_sampling / self.train_sampling
        target_shape = (int(np.round(old_shape[-2] * scale_factor)),
                        int(np.round(old_shape[-1] * scale_factor)))
        padding = calculate_padding(target_shape, (target_shape[0] + self.margin, target_shape[1] + self.margin), 16)
        return padding, scale_factor

    def unpad_images(self, images, old_shape, old_sampling):
        padding, scale_factor = self._calculate_padding_and_scale_factor(old_shape, old_sampling)
        images = unpad(images, padding)

        return rescale(images, 1 / scale_factor)

    def unpad_points(self, points, old_shape, old_sampling):
        padding, scale_factor = self._calculate_padding_and_scale_factor(old_shape, old_sampling)
        return (points - (padding[0], padding[2])) / scale_factor

    def prepare_images(self, images):
        images = torch.tensor(images, device=self.device)

        if len(images.shape) == 2:
            images = images[None, None]

        if len(images.shape) == 3:
            images = images[:, None]

        images = images.to(self._device)

        return images
        # if self.prefilter is not None:
        #    images = self.prefilter(images)

    def forward(self, images, sampling, return_aligned_maps=False):

        images = self.prepare_images(images)

        if sampling is not None:
            scale_factor = sampling / self.train_sampling
            antialising_sigma = (1 / scale_factor - 1) / 2
            if antialising_sigma > 0:
                images = GaussianFilter(sigma=antialising_sigma)(images)
            images = F.interpolate(images, scale_factor=scale_factor, mode='area', recompute_scale_factor=True)
        else:
            scale_factor = None

        margin = self.margin / self.train_sampling
        target_shape = (images.shape[2] + margin, images.shape[3] + margin)

        if self.prefilter is not None:
            images = self.prefilter(images)


        images = (images - images.mean()) / images.std()
        #images = self.masked_normalization(images)

        images, padding = pad_to_size(images, target_shape)

        if self.segmentation_net is not None:
            segmentations = nn.Softmax(1)(self.segmentation_net(images))

        else:
            segmentations = None

        images = images * segmentations[:, 0, None]
        densities = nn.Sigmoid()(self.density_net(images))
        densities = densities * segmentations[:, 0, None]

        segmentations = torch.argmax(segmentations, dim=1)
        centers, sums = self.calculate_density_centers(densities)

        centers = [(x - (padding[0], padding[2])) / scale_factor for x in centers]

        output = {}
        output['centers'] = centers
        output['sums'] = sums
        output['labels'] = np.zeros(len(centers), dtype=np.int)

        if return_aligned_maps:
            aligned_densities = unpad(densities, padding)
            aligned_densities = F.interpolate(aligned_densities, scale_factor=1 / scale_factor, mode='area',
                                              recompute_scale_factor=True)
            output['aligned_densities'] = torch.squeeze(aligned_densities).detach().cpu().numpy()

        output['densities'] = torch.squeeze(densities).detach().cpu().numpy()

        if segmentations is not None:
            contours, contour_classes = self.calculate_segments(segmentations, padding, scale_factor)

            output['contours'] = contours
            output['contour_labels'] = contour_classes

            if return_aligned_maps:
                aligned_segmentations = unpad(segmentations, padding).type(torch.float32)[:, None]
                aligned_segmentations = F.interpolate(aligned_segmentations, scale_factor=1 / scale_factor, mode='area',
                                                      recompute_scale_factor=True)
                output['aligned_segmentations'] = np.round(
                    torch.squeeze(aligned_segmentations).detach().cpu().numpy()).astype(np.int)

            output['segmentations'] = torch.squeeze(segmentations).detach().cpu().numpy()

        return output

    def process_series(self, images, sampling, max_batch, pbar=True):
        data = []

        pbar = tqdm(total=len(images), disable=not pbar)
        for start, stop in generate_batches(len(images), max_batch=max_batch):
            batch = images[start:stop]

            points, sums, contours = self(batch, sampling)

            for i in range(stop - start):
                data.append({})
                data[-1]['points'] = points[i].tolist()
                data[-1]['segments'] = [contour.tolist() for contour in contours[i]]

            pbar.update(stop - start)
        pbar.close()
        return data

    def on_fit_start(self):
        self.metrics = {'train': TEMNetMetrics(2 * self.gaussian_width, 5),
                        'val': TEMNetMetrics(2 * self.gaussian_width, 5)}

    def masked_normalization(self, images):
        segmentation_logits = self.segmentation_net(images)
        weights = nn.Softmax(1)(segmentation_logits)[:, 0, None]
        masked_images = images * weights

        weighted_sum = weights.sum(dim=(-2, -1), keepdims=True)
        weighted_mean = masked_images.sum(dim=(-2, -1), keepdims=True) / weighted_sum
        weighted_std = torch.sqrt(
            ((images - weighted_mean) ** 2 * weights).sum(dim=(-2, -1), keepdims=True) / weighted_sum)
        return (images - weighted_mean) / weighted_std

    def step(self, batch, batch_idx, phase):
        images = batch['images']

        if self.recurrent_normalization:
            images = self.masked_normalization(images)

        images[:, :, :20] = 0
        images[:, :, -20:] = 0
        images[:, :, :, :20] = 0
        images[:, :, :, -20:] = 0

        if self.segmentation_net:
            segmentation_logits = self.segmentation_net(images)
            segmentation_loss = self.segmentation_loss(segmentation_logits, batch['segments']).mean()

            self.log('segmentation_loss', segmentation_loss, prog_bar=True, on_step=True)
        else:
            segmentation_loss = None
            segmentation_logits = None

        density_logits = self.density_net(images)

        if 'weights' in batch.keys():
            density_loss = (self.density_loss(density_logits, batch['labels']) * batch['weights']).mean()
        else:
            density_loss = self.density_loss(density_logits, batch['labels']).mean()

        self.log('density_loss', density_loss, prog_bar=True, on_step=True)

        if segmentation_loss is not None:
            loss = segmentation_loss + density_loss
        else:
            loss = density_loss

        self.metrics[phase].collect(self, batch, density_logits, segmentation_logits)
        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def epoch_end(self, outputs, phase):
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar(f'loss/{phase}', avg_loss, self.current_epoch)
        self.metrics[phase].log(phase, self.current_epoch, self.logger)
        self.metrics[phase].reset()

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, 'val')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def load_preset_model(preset):
    path = os.path.join(Path(__file__).parent.parent.absolute(), 'models')

    if preset == 'hbn':
        path = os.path.join(path, 'hbn.json')

    return load_temnet(path)


def load_temnet(path):
    with open(path, 'r') as fp:
        state = json.load(fp)


    path = os.path.join(Path(path).parent, state['parameters'])

    density_net = ConvHead(R2UNet(1, 8), 1)
    segmentation_net = ConvHead(R2UNet(1, 8), 3)
    peak_enhancement = PeakEnhancementFilter(2, 5, 8)
    prefilter = GaussianFilter(sigma=5)

    temnet = TEMNet.load_from_checkpoint(path,
                                         strict=False,
                                         density_net=density_net,
                                         segmentation_net=segmentation_net,
                                         peak_enhancement=peak_enhancement,
                                         prefilter=prefilter,
                                         min_segment_area=1.4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    temnet = temnet.to(device)

    return temnet
