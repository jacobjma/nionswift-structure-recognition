import cv2
import numpy as np
from scipy import ndimage

from .utils import ind2sub, StructureRecognitionModule
from .widgets import Section, line_edit_template, combo_box_template

NBINS_ANGULAR = 64
GAUSS_WIDTH = .8


def polar_bins(shape, inner, outer, nbins_angular=32, nbins_radial=None):
    if nbins_radial is None:
        nbins_radial = outer - inner

    sy, sx = shape
    Y, X = np.ogrid[0:sy, 0:sx]

    r = np.hypot(Y - sy / 2, X - sx / 2)
    radial_bins = -np.ones(shape, dtype=int)
    valid = (r > inner) & (r < outer)
    radial_bins[valid] = nbins_radial * (r[valid] - inner) / (outer - inner)

    angles = np.arctan2(Y - sy // 2, X - sx // 2) % (2 * np.pi)

    angular_bins = np.floor(nbins_angular * (angles / (2 * np.pi)))
    angular_bins = np.clip(angular_bins, 0, nbins_angular - 1).astype(np.int)

    bins = -np.ones(shape, dtype=int)
    bins[valid] = angular_bins[valid] * nbins_radial + radial_bins[valid]

    return bins


def rollout_image(f, inner, outer, nbins_angular=32, nbins_radial=None):
    if nbins_radial is None:
        nbins_radial = outer - inner

    bins = polar_bins(f.shape, inner, outer, nbins_angular=nbins_angular, nbins_radial=nbins_radial)

    with np.errstate(divide='ignore', invalid='ignore'):
        unrolled = ndimage.mean(f, bins, range(0, bins.max() + 1))

    unrolled = unrolled.reshape((nbins_angular, nbins_radial))

    for i in range(unrolled.shape[1]):
        y = unrolled[:, i]
        nans = np.isnan(y)
        y[nans] = np.interp(nans.nonzero()[0], (~nans).nonzero()[0], y[~nans], period=len(y))
        unrolled[:, i] = y

    return unrolled


def create_template(peaks, intensities, nbins_angular, gauss_width, margin=1):
    peaks = np.array(peaks) * nbins_angular + margin
    intensities = np.array(intensities)
    y, x = np.mgrid[:nbins_angular + 2 * margin, -margin:margin + 1]
    r2 = x ** 2 + (y[None] - peaks[:, None, None]) ** 2
    template = np.exp(-r2 / (2 * gauss_width ** 2)).sum(axis=0)

    template[margin:2 * margin] += template[-margin:]
    template[-2 * margin:-margin] += template[:margin]
    template = template[margin:-margin]
    return template


def get_spots(power_spec, peak_positions, peak_intensities):
    inner = 1
    outer = min(power_spec.shape) // 2
    margin = 1

    template = create_template(peak_positions, peak_intensities, NBINS_ANGULAR, GAUSS_WIDTH, margin)

    unrolled = rollout_image(power_spec, inner, outer, nbins_angular=NBINS_ANGULAR)
    unrolled /= unrolled.max()

    unrolled = np.pad(unrolled, [(unrolled.shape[0] // 2, unrolled.shape[0] // 2), (0, 0)], mode='wrap').astype(
        np.float32)
    h = cv2.matchTemplate(unrolled, template.astype(np.float32), method=2)

    rows, cols = ind2sub(h.shape, h.argmax())

    angle = (rows + .5) / NBINS_ANGULAR * 2 * np.pi
    angles = peak_positions * 2 * np.pi + angle
    radial = cols + inner + margin + .5

    return radial, angles


presets = {'graphene':
               {'crystal_structure': 'graphene',
                'spot_radius': .246,
                }
           }

template = {'graphene':
                {'prefactor': np.sqrt(3.) / 2.,
                 'peak_positions': np.linspace(0, 1, 6, endpoint=False),
                 'peak_intensities': np.ones(6)
                 }
            }


class ScaleDetectionModule(StructureRecognitionModule):

    def __init__(self, ui, document_controller):
        super().__init__(ui, document_controller)

        self.structure = None
        self.spot_radius = None

    def create_widgets(self, column):
        section = Section(self.ui, 'Scale detection')
        column.add(section)

        spot_radius_row, self.spot_radius_line_edit = line_edit_template(self.ui, 'Spot radius [nm]')
        structure_row, self.structure_combo_box = combo_box_template(self.ui, 'Crystal structure', ['Graphene'])
        section.column.add(structure_row)
        section.column.add(spot_radius_row)

    def set_preset(self, name):
        self.structure_combo_box.current_item = presets[name]['crystal_structure']
        self.spot_radius_line_edit.text = presets[name]['spot_radius']

    def fetch_parameters(self):
        self.structure = self.structure_combo_box._widget.current_item.lower()
        self.spot_radius = float(self.spot_radius_line_edit._widget.text)

    def detect_scale(self, data):
        # source_data_item = self.document_controller.target_data_item
        # data = source_data_item.xdata.data
        prefactor = template[self.structure]['prefactor']
        peak_positions = template[self.structure]['peak_positions']
        peak_intensities = template[self.structure]['peak_intensities']

        assert len(data.shape) == 2

        fft = np.fft.fftshift(np.fft.fft2(data))
        power_spec = np.abs(fft) ** 2

        radial, angles = get_spots(power_spec, peak_positions, peak_intensities)

        scale = radial * self.spot_radius / float(min(power_spec.shape)) * prefactor

        return scale

        #print(scale)

        # logging.info('Detected scale: {:.5f} pixels / nm'.format(scale))

        # main_column.add(self.spot_radius_line_edit._row)
