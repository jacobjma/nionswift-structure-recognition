from .model import build_model_from_dict, presets
from .utils import StructureRecognitionModule
from .widgets import Section, line_edit_template


class DeepLearningModule(StructureRecognitionModule):

    def __init__(self, ui, document_controller):
        super().__init__(ui, document_controller)

        self.training_sampling = None
        self.mask_model = None
        self.density_model = None
        self.nms_distance = None

    def create_widgets(self, column):
        section = Section(self.ui, 'Deep learning')
        column.add(section)

        # model_row, self.model_line_edit = line_edit_template(self.ui, 'Model')
        mask_weights_row, self.mask_weights_line_edit = line_edit_template(self.ui, 'Mask weights')
        density_weights_row, self.density_weights_line_edit = line_edit_template(self.ui, 'Density weights')
        training_scale_row, self.training_sampling_line_edit = line_edit_template(self.ui, 'Training sampling [A]')
        margin_row, self.margin_line_edit = line_edit_template(self.ui, 'Margin [A]')
        nms_distance_row, self.nms_distance_line_edit = line_edit_template(self.ui, 'NMS distance [A]')
        nms_threshold_row, self.nms_threshold_line_edit = line_edit_template(self.ui, 'NMS threshold')

        # section.column.add(model_row)
        section.column.add(mask_weights_row)
        section.column.add(density_weights_row)
        section.column.add(training_scale_row)
        section.column.add(margin_row)
        section.column.add(nms_distance_row)
        section.column.add(nms_threshold_row)

    def set_preset(self, name):
        # self.model_line_edit.text = presets[name]['model_file']
        self.mask_weights_line_edit.text = presets[name]['mask_model']['weights']
        self.density_weights_line_edit.text = presets[name]['density_model']['weights']
        self.training_sampling_line_edit.text = presets[name]['training_sampling']
        self.margin_line_edit.text = presets[name]['margin']
        self.nms_distance_line_edit.text = presets[name]['nms']['distance']
        self.nms_threshold_line_edit.text = presets[name]['nms']['threshold']

    def forward_pass(self, preprocessed_image):
        density, classes = self.model(preprocessed_image)
        return density, classes

    def fetch_parameters(self):
        self.training_sampling = float(self.training_sampling_line_edit.text)
        self.margin = float(self.margin_line_edit.text)
        self.nms_distance = float(self.nms_distance_line_edit.text)
        self.nms_threshold = float(self.nms_threshold_line_edit.text)

        # parameters = presets

        self.model = build_model_from_dict(presets['graphene'])

        # self.load_model()

        # models_dir = os.path.join(os.path.dirname(__file__), 'models')
        #
        # # self.model_file = os.path.join(models_dir, self.model_line_edit.text)
        # self.parameters_file = os.path.join(models_dir, self.parameters_line_edit.text)
        #
        # self.model = UNet([DensityMap(), ClassificationMap(4)], init_features=32, in_channels=1, p=0.)
        # self.model.load_state_dict(torch.load(self.parameters_file, map_location=torch.device('cpu')))

        # json_file = open(self.model_file, 'r')
        # self.model = keras.models.model_from_json(json_file.read())
        # self.model.load_weights(self.parameters_file)
