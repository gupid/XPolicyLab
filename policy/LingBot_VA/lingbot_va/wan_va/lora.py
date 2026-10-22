"""LoRA injection, explicit trainable heads and merged Diffusers export."""
from __future__ import annotations

import re
from typing import Iterable

import torch
from torch import nn


def apply_lora_checkpointing(model):
    """Replay identical dropout masks during activation recomputation."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    for index, block in enumerate(model.blocks):
        model.blocks[index] = checkpoint_wrapper(block, preserve_rng_state=True)


class LoRALinear(nn.Module):
    """A frozen Linear layer with a trainable low-rank residual."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        factory = dict(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False, **factory)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False, **factory)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    @property
    def in_features(self):
        return self.base_layer.in_features

    @property
    def out_features(self):
        return self.base_layer.out_features

    def forward(self, hidden_states):
        result = self.base_layer(hidden_states)
        update = self.lora_B(self.lora_A(self.lora_dropout(hidden_states)))
        return result + update * self.scaling

    def delta_weight(self):
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling


def _target_matches(module_name: str, target_modules: Iterable[str]) -> bool:
    """Match DiffSynth-style short targets against LingBot module names."""
    leaf = module_name.rsplit(".", 1)[-1]
    if leaf == "to_out":
        return False
    for target in target_modules:
        target = target.strip()
        if module_name == target or module_name.endswith('.' + target):
            return True
        if target == "o" and module_name.endswith("to_out.0"):
            return True
        if target in {"q", "k", "v"} and leaf == f"to_{target}":
            return True
        if target == "ffn.0" and module_name.endswith("ffn.net.0.proj"):
            return True
        if target == "ffn.2" and module_name.endswith("ffn.net.2"):
            return True
        if target == "action_embedder" and module_name.endswith("action_embedder"):
            return True
        if target == "action_proj_out" and module_name.endswith("action_proj_out"):
            return True
        if target == "action_time" and (
            module_name.endswith("condition_embedder_action.time_proj")
            or module_name.endswith("condition_embedder_action.time_embedder.linear_1")
            or module_name.endswith("condition_embedder_action.time_embedder.linear_2")
        ):
            return True
    return False


def apply_lora(model: nn.Module, rank: int = 32, alpha: float | None = None,
               target_modules: str | Iterable[str] = "q,k,v,o,ffn.0,ffn.2",
               dropout: float = 0.0,
               train_modules: str | Iterable[str] = ()) -> list[str]:
    """Freeze ``model`` and replace matching Linear modules with LoRALinear."""
    if isinstance(target_modules, str):
        target_modules = [part for part in target_modules.split(",") if part]
    target_modules = tuple(target_modules)
    if isinstance(train_modules, str):
        train_modules = [part for part in train_modules.split(',') if part]
    alpha = float(rank if alpha is None else alpha)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    replacements: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _target_matches(module_name, target_modules):
            parent_name, _, child_name = module_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            replacements.append((module_name, parent, child_name, module))
    for module_name, parent, child_name, module in replacements:
        setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
    if not replacements:
        raise ValueError(f"No Linear modules matched LoRA targets: {list(target_modules)}")
    for name in train_modules:
        model.get_submodule(name).requires_grad_(True)
    return [name for name, _, _, _ in replacements]


def lora_parameters(model: nn.Module):
    return [parameter for name, parameter in model.named_parameters()
            if name.endswith("lora_A.weight") or name.endswith("lora_B.weight")]


def merged_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a base-model-compatible state dict with LoRA deltas merged."""
    state = {name: value.detach() for name, value in model.state_dict().items()}
    modules = {name.replace("._checkpoint_wrapped_module", ""): module
               for name, module in model.named_modules()}
    merged: dict[str, torch.Tensor] = {}
    handled: set[str] = set()
    for name, value in state.items():
        match = re.match(r"^(.*)\.base_layer\.(weight|bias)$", name)
        if match:
            prefix, kind = match.groups()
            module = modules[prefix]
            if kind == "weight":
                value = value + module.delta_weight().detach().to(value.device, value.dtype)
            merged[f"{prefix}.{kind}"] = value
            handled.update({name, f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"})
        elif name not in handled and ".lora_A." not in name and ".lora_B." not in name:
            merged[name] = value
    return merged


def merge_lora_state_dict(state: dict[str, torch.Tensor], rank: int, alpha: float) -> dict[str, torch.Tensor]:
    """Merge a possibly FSDP-normalized LoRA state dict without module access."""
    scaling = float(alpha) / int(rank)
    merged: dict[str, torch.Tensor] = {}
    handled: set[str] = set()
    for name, value in state.items():
        match = re.match(r"^(.*)\.base_layer\.(weight|bias)$", name)
        if match:
            prefix, kind = match.groups()
            if kind == "weight":
                a = state.get(f"{prefix}.lora_A.weight")
                b = state.get(f"{prefix}.lora_B.weight")
                if a is not None and b is not None:
                    value = value + (b @ a).to(value.device, value.dtype) * scaling
            merged[f"{prefix}.{kind}"] = value
            handled.update({name, f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"})
        elif name not in handled and ".lora_A." not in name and ".lora_B." not in name:
            merged[name] = value
    return merged


def lora_summary(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"parameters": total, "trainable_parameters": trainable}
