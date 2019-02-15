from torch.optim import Optimizer


class LearningRateSchedulerBase(object):
    def __call__(self, optimizer: Optimizer, epoch: int):
        raise NotImplementedError
