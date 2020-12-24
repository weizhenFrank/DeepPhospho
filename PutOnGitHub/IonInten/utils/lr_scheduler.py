from bisect import bisect_right

import torch
from torch.optim import Optimizer

import IonInten.configs.config_main as cfg


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            warmup_iters,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_method="linear",
            last_epoch=-1,
            # PlateauLRSCHEDULER
            mode='min', factor=0.1, patience=1,  # 3-->1
            verbose=True, threshold=1e-4, threshold_mode='abs',
            cooldown=1, min_lr=0, eps=1e-8
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + mode + ' is unknown!')
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            self.is_better = lambda a, best: a < best * rel_epsilon
            self.mode_worse = float('Inf')
        elif mode == 'min' and threshold_mode == 'abs':
            self.is_better = lambda a, best: a < best - threshold
            self.mode_worse = float('Inf')
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            self.is_better = lambda a, best: a > best * rel_epsilon
            self.mode_worse = -float('Inf')
        else:  # mode == 'max' and epsilon_mode == 'abs':
            self.is_better = lambda a, best: a > best + threshold
            self.mode_worse = -float('Inf')

    def epoch_step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

            return [
                base_lr
                * warmup_factor
                * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * warmup_factor
                * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first `warmup_steps` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    # Parameters
    model_size : `int`, required.
        The hidden size parameter which dominates the number of parameters in your model.
    warmup_steps : `int`, required.
        The number of steps to linearly increase the learning rate.
    factor : `float`, optional (default = 1.0).
        The overall scale factor for the learning rate decay.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            model_size: int,
            warmup_steps: int,
            factor: float = 1.0,
            last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.model_size = model_size
        self.last_epoch = 0
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def step(self, metric: float = None, epoch: int = None) -> None:
        self.step_batch(epoch)

    def step_batch(self, batch_num_total: int = None) -> None:
        if batch_num_total is None:
            self.last_epoch += 1  # type: ignore
        else:
            self.last_epoch = batch_num_total
        for param_group, learning_rate in zip(self.optimizer.param_groups, self.get_values()):
            param_group["lr"] = learning_rate

    def get_values(self):
        step = max(self.last_epoch, 1)
        scale = self.factor * (
                self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
        return [scale for _ in range(len(self.base_lrs))]

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError


def make_lr_scheduler(optimizer, steps: tuple, warmup_iters):
    return WarmupMultiStepLR(
        optimizer,
        steps,
        warmup_iters,
        # gamma=0.1,
        warmup_factor=1. / 3,
        warmup_method="linear",
    )
