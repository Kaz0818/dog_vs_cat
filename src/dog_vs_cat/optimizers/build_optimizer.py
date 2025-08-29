from typing import Literal
import torch
import torch.optim as optim
from torch.nn import Module

OptimizerName = Literal["adamw", "adam", "sgd"]

def build_optimizer(
    model: Module,
    name: OptimizerName = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,   # SGD用
):
    name = name.lower()
    params = model.parameters()

    lr = float(lr)
    weight_decay = float(weight_decay)
    momentum = float(momentum)
    
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer: {name}")