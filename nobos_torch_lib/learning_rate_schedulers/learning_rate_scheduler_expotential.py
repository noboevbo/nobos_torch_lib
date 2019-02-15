from torch.optim import Optimizer

from nobos_torch_lib.learning_rate_schedulers.learning_rate_scheduler_base import LearningRateSchedulerBase


class LearningRateSchedulerExpotential(LearningRateSchedulerBase):
    def __init__(self, optimizer, lr_decay: float = 0.1, lr_decay_epoch: int = 30):
        super().__init__(optimizer)
        self.__lr_decay = lr_decay
        self.__lr_decay_epoch = lr_decay_epoch

    def __call__(self, optimizer: Optimizer, epoch: int):
        if epoch == 0 or epoch % self.__lr_decay_epoch:
            return optimizer
        print("Adjust learning rate")
        for param_group in optimizer.param_groups:
            param_group['lr'] *= self.__lr_decay
        return optimizer
