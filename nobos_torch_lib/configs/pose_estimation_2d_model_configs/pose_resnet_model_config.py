from nobos_torch_lib.configs.general_model_config import GeneralModelConfig


class PoseResNetModelConfig(GeneralModelConfig):
    def __init__(self):
        super().__init__()
        self.input_width = 256
        self.input_height = 192
        self.num_joints = 17
        self.final_conv_kernel = 1
        self.deconv_with_bias = False
        self.num_deconv_layers = 3
        self.num_deconv_filters = [256, 256, 256]
        self.num_deconv_kernels = [4, 4, 4]
        self.num_layers = 50
        self.post_process = True
