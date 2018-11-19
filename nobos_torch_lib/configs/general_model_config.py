class GeneralModelConfig(object):
    __slots__ = ['model_state_file', 'use_gpu', 'gpu_number']

    def __init__(self):
        self.use_gpu = True
        self.gpu_number = 0
        self.model_state_file = ''
