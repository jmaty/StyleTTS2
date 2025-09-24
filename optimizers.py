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

    def step(self, key=None, scaler=None):
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler) for key in keys]

    def _step(self, key, scaler=None):
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

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


def define_scheduler(optimizer, params):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params.get("max_lr", 2e-4),
        epochs=params.get("epochs", 200),
        steps_per_epoch=params.get("steps_per_epoch", 1000),
        pct_start=params.get("pct_start", 0.0),
        div_factor=params.get("div_factor", 1),
        final_div_factor=params.get("final_div_factor", 1),
    )
    return scheduler


def build_optimizer(parameters_dict, scheduler_params_dict, lr):
    """
    Build per-module optimizers.

    Supports component-specific learning rates in two ways:
    - If `scheduler_params_dict[key]["lr"]` is present, it takes precedence.
    - Otherwise falls back to the global `lr` argument.

    This preserves backward compatibility while allowing dedicated LRs to be
    defined at the time parameter groups are prepared.
    """

    def _hyper_for(key):
        # Pull optimizer hyperparams from scheduler_params_dict[key]
        cfg = scheduler_params_dict.get(key, {})
        lr_key = cfg.get("lr", None)
        h = {
            "lr": lr_key if lr_key is not None else lr,
            "betas": cfg.get("betas", (0.0, 0.99)),
            "eps": cfg.get("eps", 1e-9),
            "weight_decay": cfg.get("weight_decay", 1e-4),
        }
        # Extras: any keys not used by OneCycleLR or above hyperparams
        scheduler_keys = {
            "max_lr",
            "epochs",
            "steps_per_epoch",
            "pct_start",
            "lr",
            "betas",
            "eps",
            "weight_decay",
        }
        extras = {k: v for k, v in cfg.items() if k not in scheduler_keys}
        h["extras"] = extras
        return h

    optim = {}
    for key, params in parameters_dict.items():
        h = _hyper_for(key)
        opt = AdamW(
            params,
            lr=h["lr"],
            weight_decay=h["weight_decay"],
            betas=h["betas"],
            eps=h["eps"],
        )
        # Apply extra param-group fields if specified (e.g., initial_lr, min_lr)
        if h["extras"]:
            for g in opt.param_groups:
                g.update(h["extras"])
        optim[key] = opt

    schedulers = dict(
        [(key, define_scheduler(opt, scheduler_params_dict[key])) for key, opt in optim.items()]
    )

    multi_optim = MultiOptimizer(optim, schedulers)
    return multi_optim
