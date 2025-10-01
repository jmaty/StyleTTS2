# coding:utf-8
from functools import reduce

import torch
from torch.optim import AdamW

from logger import get_logger

# Setup logger
logger = get_logger(__name__)


class MultiOptimizer:
    def __init__(self, optimizers=None, schedulers=None):
        if optimizers is None:
            optimizers = {}
        if schedulers is None:
            schedulers = {}
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.keys = list(optimizers.keys())
        self.param_groups = reduce(
            lambda x, y: x + y, [v.param_groups for v in self.optimizers.values()]
        )

    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict()) for key in self.keys]
        return state_dicts

    def load_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.optimizers[key].load_state_dict(val)
            except Exception:
                logger.warning("%s not loaded", key)

    def step(self, key=None, scaler=None, step_scheduler: bool = True):
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler, step_scheduler) for key in keys]

    def _step(self, key, scaler=None, step_scheduler=True):
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

        # Optionally advance the LR scheduler exactly once per logical step.
        if step_scheduler:
            # Some schedulers (e.g., OneCycleLR) have a fixed number of total_steps
            # and raise if stepped beyond that. Guard against over-stepping so that
            # auxiliary optimizer updates in the training loop do not crash training.
            self.schedulers[key].step()

    def zero_grad(self, key=None):
        if key is not None:
            self.optimizers[key].zero_grad()
        else:
            _ = [self.optimizers[key].zero_grad() for key in self.keys]

    def scheduler(self, *args, key=None):
        if key is not None:
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step(*args) for key in self.keys]

    def __setitem__(self, key, params_dict):
        """
        Set parameters for a specific optimizer.
        Args:
            key (str): The key identifying the optimizer to update.
            params_dict (dict): Dictionary containing parameter names and their new values
                               to be applied to all parameter groups of the optimizer.
        Raises:
            KeyError: If the specified optimizer key is not found in the optimizers collection.
        Example:
            optimizer_manager['adam'] = {'lr': 0.001, 'weight_decay': 1e-5}
        """

        if key not in self.optimizers:
            raise KeyError(f"Optimizer key '{key}' not found")

        for param_name, value in params_dict.items():
            for g in self.optimizers[key].param_groups:
                g[param_name] = value

    def __getitem__(self, key):
        """
        Retrieve the first parameter group of an optimizer by key.
        Args:
            key: The string key identifying the optimizer to retrieve.
        Returns:
            dict: The first parameter group of the specified optimizer, containing
                  parameters like 'lr', 'momentum', 'weight_decay', etc.
        Raises:
            KeyError: If the specified optimizer key is not found in the collection.
        Note:
            This method only returns the first parameter group. If multiple parameter
            groups are needed, access the optimizer directly via self.optimizers[key].
        """

        if key not in self.optimizers:
            raise KeyError(f"Optimizer key '{key}' not found")

        # Vrátí první param_group (nebo všechny, pokud je potřebujete)
        return self.optimizers[key].param_groups[0]


def define_scheduler(optimizer, optimizer_params, epochs, steps_per_epoch):
    if optimizer_params.scheduler == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer_params.scheduler_params.max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=optimizer_params.scheduler_params.pct_start,
            div_factor=optimizer_params.scheduler_params.div_factor,
            final_div_factor=optimizer_params.scheduler_params.final_div_factor,
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {optimizer_params.scheduler}")
    return scheduler


def build_optimizer(param_dict, optimizer_params, epochs, steps_per_epoch):
    """
    Builds a multi-optimizer with corresponding learning rate schedulers.
    Args:
        parameters_dict (dict): Dictionary mapping optimizer keys to model parameters.
            Each key's parameters will be optimized separately.
        scheduler_params_dict (dict): Dictionary mapping optimizer keys to scheduler parameters.
            Should contain configuration for each optimizer's scheduler.
        lr (float): Base learning rate for all optimizers.
    Returns:
        MultiOptimizer: A wrapper containing all optimizers and their schedulers.
            Each optimizer is an AdamW instance with weight_decay=1e-4, betas=(0.0, 0.99), eps=1e-9.
    """
    # raw_param_groups = {k: list(self._model[k].parameters()) for k in self._model}

    # # Leave only parameters with requires_grad=True (default),
    # parameters_filtered = {
    #     k: [p for p in v if p.requires_grad] for k, v in raw_param_groups.items()
    # }

    # not_trainable_modules = [k for k, v in parameters_filtered.items() if len(v) == 0]
    # parameters_dict = {k: v for k, v in param_groups.items() if v}

    # Create optimizers
    if optimizer_params.optimizer == "AdamW":
        optim = {
            k: AdamW(
                params,
                lr=optimizer_params.lr,
                weight_decay=optimizer_params.weight_decay,
                betas=optimizer_params.betas,
                eps=optimizer_params.eps,
            )
            for k, params in param_dict.items()
        }
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_params.optimizer}")

    # Create schedulers
    schedulers = {
        k: define_scheduler(opt, optimizer_params, epochs, steps_per_epoch)
        for k, opt in optim.items()
    }

    # Combine into MultiOptimizer
    multi_optim = MultiOptimizer(optim, schedulers)

    return multi_optim
