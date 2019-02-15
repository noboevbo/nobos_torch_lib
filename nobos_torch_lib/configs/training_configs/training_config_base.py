import os

from nobos_commons.utils.file_helper import get_create_path

from nobos_torch_lib.learning_rate_schedulers.learning_rate_scheduler_base import LearningRateSchedulerBase
from nobos_torch_lib.learning_rate_schedulers.learning_rate_scheduler_dummy import LearningRateSchedulerDummy


class TrainingConfigBase(object):
    def __init__(self, model_name: str, model_dir: str):
        self.model_name: str = model_name
        self.model_dir: str = get_create_path(model_dir)

        self.num_epochs = 150
        self.checkpoint_epoch: int = 50

        # Optimizer
        self.learning_rate: float = 0.01
        self.momentum: float = 0.9
        self.weight_decay: float = 5e-4

        # LR Scheduler
        self.learning_rate_scheduler: LearningRateSchedulerBase = LearningRateSchedulerDummy()

    def get_output_path(self, epoch: int):
        return os.path.join(self.model_dir, "{}_cp{}.pth".format(self.model_name, str(epoch).zfill(4)))
