from torch.optim import Optimizer

from nobos_torch_lib.learning_rate_schedulers.learning_rate_scheduler_base import LearningRateSchedulerBase


class LearningRateSchedulerDummy(LearningRateSchedulerBase):
    def __call__(self, optimizer: Optimizer, epoch: int):
        return optimizer
