class LRSchedulerSGD:
    def __init__(self, mu_0=0.01, alpha=10, beta=0.75, blocks_with_smaller_lr=0):
        self.mu_0 = mu_0
        self.alpha = alpha
        self.beta = beta
        self.blocks_with_smaller_lr = blocks_with_smaller_lr

    def step(self, opt, current_epoch, total_epoch):
        p = (current_epoch + 1.) / total_epoch
        mu_p = self.mu_0 / ((1 + self.alpha * p) ** self.beta)
        for param_group in opt.param_groups[:self.blocks_with_smaller_lr]:
            param_group['lr'] = 0.1 * mu_p
        for param_group in opt.param_groups[self.blocks_with_smaller_lr:]:
            param_group['lr'] = mu_p

class DANNCASchedulerSGD:
    def __init__(self, gamma=0.1, power=0.5, base_lr=0.01):
        self.gamma = gamma
        self.power = power
        self.base_lr = base_lr

    def step(self, opt, current_epoch, total_epoch):
        for param_group in opt.param_groups:
            param_group['lr'] = self.base_lr * (1 + self.gamma * current_epoch) ** (-self.power)
