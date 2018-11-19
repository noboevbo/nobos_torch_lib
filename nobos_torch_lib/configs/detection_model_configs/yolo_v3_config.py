from nobos_torch_lib.configs.general_model_config import GeneralModelConfig


class YoloV3Config(GeneralModelConfig):
    __slots__ = ['resolution', 'confidence', 'nms_thresh', 'network_config_file']

    def __init__(self):
        super().__init__()
        self.resolution = 320
        self.confidence = 0.8
        self.nms_thresh = 0.4
