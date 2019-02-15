def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=30):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer
    print("Adjust learning rate")
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer
