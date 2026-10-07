# Copyright 2025 juno community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""Opt-in LoRA support for juno training.

Adapters are injected *in place* via ``peft.inject_adapter_in_model`` rather than
by wrapping the framework in a ``PeftModel``. Several framework internals resolve
submodules by attribute path at run time -- ``QWen3.forward`` looks up
``self.model.model.language_model`` to attach a hidden-state hook, and
``juno-policy`` monkey-patches the ``forward`` of every Qwen decoder layer
-- so an extra wrapper level in the module tree would break them. In-place
injection only swaps leaf ``nn.Linear`` modules and leaves every path intact.

Checkpoints are written with the adapters folded back into the base weights, so a
LoRA run emits the same ``state_dict`` layout as a full fine-tune and the eval
side needs no changes.
"""

import os
import re
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
DEFAULT_TARGET_SCOPES = "qwen_vl_interface,mot_reasoning_layers"
LORA_RESUME_MODES = ("adapter", "restart")

_LORA_PARAM_RE = re.compile(r"(?:^|\.)lora_")


def _as_list(value, default: str) -> List[str]:
    if value is None:
        value = default
    if not isinstance(value, str):
        value = ",".join(str(item) for item in value)
    return [item.strip() for item in value.split(",") if item.strip()]


def _lora_cfg(cfg):
    trainer_cfg = getattr(cfg, "trainer", None)
    if trainer_cfg is None:
        return None
    return trainer_cfg.get("lora", None)


def _resolve(model: nn.Module, path: str) -> Optional[nn.Module]:
    module = model
    for attr in path.split("."):
        if not hasattr(module, attr):
            return None
        module = getattr(module, attr)
    return module


def _named_lora_layers(model: nn.Module) -> Dict[str, nn.Module]:
    try:
        from peft.tuners.lora import LoraLayer
    except ImportError:
        return {}
    return {name: module for name, module in model.named_modules() if isinstance(module, LoraLayer)}


def _print(message: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(message)


def is_lora_enabled(cfg) -> bool:
    lora_cfg = _lora_cfg(cfg)
    return bool(lora_cfg is not None and lora_cfg.get("enable", False))


def lora_resume_mode(cfg) -> str:
    """How a resumed run should treat the adapters that produced the checkpoint.

    ``adapter``
        Reload the saved ``A``/``B`` and undo the merge on the base weights, so
        training picks up the same low-rank subspace it left off in.
    ``restart``
        Ignore them and open a fresh subspace on top of the merged weights. The
        model function is identical either way; this only changes which
        directions the next stretch of training can move in.
    """
    lora_cfg = _lora_cfg(cfg)
    mode = "adapter" if lora_cfg is None else str(lora_cfg.get("resume_mode", "adapter"))
    if mode not in LORA_RESUME_MODES:
        raise ValueError(f"trainer.lora.resume_mode must be one of {LORA_RESUME_MODES}, got `{mode}`.")
    return mode


def apply_lora(model: nn.Module, cfg) -> bool:
    """Freeze the configured scopes and inject LoRA adapters into them in place.

    Must run after the pretrained checkpoint is loaded (the base weights are what
    the adapters sit on top of) and before the optimizer is built (so the adapter
    parameters end up in the LR groups). Returns ``True`` when adapters were
    injected, ``False`` when LoRA is disabled.
    """
    if not is_lora_enabled(cfg):
        return False

    from peft import LoraConfig, inject_adapter_in_model

    lora_cfg = _lora_cfg(cfg)
    resume_mode = lora_resume_mode(cfg)  # validated up front so a typo fails at startup
    scopes = _as_list(lora_cfg.get("target_scopes", None), DEFAULT_TARGET_SCOPES)
    leaves = _as_list(lora_cfg.get("target_modules", None), DEFAULT_TARGET_MODULES)
    excludes = _as_list(lora_cfg.get("exclude_modules", None), "")
    if not scopes or not leaves:
        raise ValueError("trainer.lora requires non-empty `target_scopes` and `target_modules`.")

    # Freeze every base parameter inside the scopes; anything outside them
    # (action head, JEPA projectors, reasoning queries, ...) keeps training fully.
    for scope in scopes:
        module = _resolve(model, scope)
        if module is None:
            raise ValueError(f"trainer.lora.target_scopes references a missing module path: `{scope}`")
        for param in module.parameters():
            param.requires_grad = False

    scope_re = "|".join(re.escape(scope) for scope in scopes)
    leaf_re = "|".join(re.escape(leaf) for leaf in leaves)
    target_pattern = rf"^(?:{scope_re})\.(?:.*\.)?(?:{leaf_re})$"

    lora_config = LoraConfig(
        r=int(lora_cfg.get("r", 32)),
        lora_alpha=int(lora_cfg.get("alpha", 64)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias="none",
        target_modules=target_pattern,
        exclude_modules=excludes or None,
    )

    # peft freezes *every* non-adapter parameter of the model it is handed, which
    # would silently take the action head and the JEPA projectors down with it.
    # Snapshot the layout we want and restore it once the adapters are in.
    desired_grad = {name: param.requires_grad for name, param in model.named_parameters()}

    inject_adapter_in_model(lora_config, model)

    lora_layers = _named_lora_layers(model)
    if not lora_layers:
        raise RuntimeError(
            f"trainer.lora matched no module with pattern `{target_pattern}`. "
            "Check `target_scopes` / `target_modules`."
        )

    for name, param in model.named_parameters():
        if _LORA_PARAM_RE.search(name):
            param.requires_grad = True
        else:
            # Wrapping renames a targeted leaf `<path>.weight` to
            # `<path>.base_layer.weight`; both spellings map to the same intent.
            param.requires_grad = desired_grad.get(
                name, desired_grad.get(name.replace(".base_layer.", "."), False)
            )

    # `freeze_modules` runs after this and would freeze the adapters too, which
    # looks like a healthy run that learns nothing in the scoped backbones.
    frozen_cfg = cfg.trainer.get("freeze_modules", "")
    frozen_paths = _as_list(frozen_cfg, "") if isinstance(frozen_cfg, str) else []
    clashes = [p for p in frozen_paths if any(p == s or p.startswith(f"{s}.") for s in scopes)]
    if clashes:
        raise ValueError(
            f"trainer.freeze_modules={clashes} overlaps trainer.lora.target_scopes={scopes}; "
            "the adapters would be frozen and nothing would train there. "
            "Drop those paths from freeze_modules or from target_scopes."
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_params = sum(p.numel() for n, p in model.named_parameters() if _LORA_PARAM_RE.search(n))
    _print(
        f"🧩 LoRA enabled: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
        f"dropout={lora_config.lora_dropout}\n"
        f"   scopes={scopes} targets={leaves} resume_mode={resume_mode}\n"
        f"   wrapped {len(lora_layers)} linear layers, "
        f"{adapter_params / 1e6:.2f}M adapter params, "
        f"{trainable / 1e6:.2f}M trainable params total"
    )
    return True


def _adapter_delta(module: nn.Module, reference: torch.Tensor) -> torch.Tensor:
    """Summed ``B @ A * scaling`` of a LoRA layer's active adapters, in fp32."""
    delta = torch.zeros(reference.shape, dtype=torch.float32, device=reference.device)
    for adapter in module.active_adapters:
        if adapter not in getattr(module, "lora_A", {}):
            continue
        delta += module.get_delta_weight(adapter).detach().to(device=reference.device, dtype=torch.float32)
    return delta


def merge_lora_into_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Fold LoRA deltas into the base weights and drop the adapter entries.

    The result has the same key layout as a full fine-tune checkpoint, so eval and
    ``is_resume`` keep working without knowing LoRA was used. Returns
    ``state_dict`` untouched when the model carries no adapters.
    """
    lora_layers = _named_lora_layers(model)
    if not lora_layers:
        return state_dict

    merged = {}
    for key, value in state_dict.items():
        if _LORA_PARAM_RE.search(key):
            continue
        merged[key.replace(".base_layer.", ".")] = value

    for name, module in lora_layers.items():
        weight_key = f"{name}.weight"
        if weight_key not in merged:
            continue
        base = merged[weight_key]
        merged[weight_key] = (base.to(torch.float32) + _adapter_delta(module, base)).to(base.dtype)

    return merged


def lora_sidecar_path(checkpoint_path: str) -> str:
    """Adapter file that sits next to a merged checkpoint.

    ``.../steps_5000_pytorch_model.pt`` -> ``.../steps_5000_lora.pt``
    """
    directory, filename = os.path.split(checkpoint_path)
    stem = filename
    for suffix in ("pytorch_model.pt", "model.safetensors"):
        if filename.endswith(suffix):
            stem = filename[: -len(suffix)]
            break
    return os.path.join(directory, f"{stem}lora.pt")


def extract_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pull just the adapter tensors out of a full state dict."""
    return {key: value for key, value in state_dict.items() if _LORA_PARAM_RE.search(key)}


def load_lora_state_dict(model: nn.Module, path: str) -> bool:
    """Restore adapters from a sidecar and rebase the frozen weights underneath.

    The companion checkpoint stores ``W_merged = W_0 + B @ A * scaling``. Putting
    the saved ``A``/``B`` back without touching the base weights would apply that
    delta a second time, so the merge is undone here. The subtraction runs in
    fp32 but lands back in the checkpoint dtype, so each resume costs the frozen
    weights at most one unit in the last place.

    Returns ``False`` when there is nothing to restore (no adapters on the model,
    no sidecar on disk, or a sidecar that never finished being written), leaving
    the caller to fall back to fresh adapters on top of the merged weights.
    """
    lora_layers = _named_lora_layers(model)
    if not lora_layers or not os.path.exists(path):
        return False

    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        # A sidecar truncated by a killed job should not sink the run: the merged
        # weights next to it are still complete and usable.
        _print(f"⚠️  LoRA sidecar `{path}` is unreadable ({type(exc).__name__}: {exc}); ignoring it.")
        return False

    current = {name: param for name, param in model.named_parameters() if _LORA_PARAM_RE.search(name)}

    missing = sorted(set(current) - set(saved))
    unexpected = sorted(set(saved) - set(current))
    if missing or unexpected:
        raise RuntimeError(
            f"LoRA sidecar `{path}` does not match the configured adapters "
            f"({len(missing)} missing, {len(unexpected)} unexpected; e.g. missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}). The run's trainer.lora settings most likely differ "
            "from the ones that produced this checkpoint."
        )

    with torch.no_grad():
        for name, param in current.items():
            value = saved[name]
            if value.shape != param.shape:
                raise RuntimeError(
                    f"LoRA sidecar `{path}` has shape {tuple(value.shape)} for `{name}`, "
                    f"but the model expects {tuple(param.shape)}. Check trainer.lora.r."
                )
            param.copy_(value.to(device=param.device, dtype=param.dtype))

        # Deltas must be recomputed only after every A/B is in place.
        for module in lora_layers.values():
            base = module.base_layer.weight
            base.copy_((base.to(torch.float32) - _adapter_delta(module, base)).to(base.dtype))

    _print(f"🧩 Restored {len(current)} LoRA adapter tensors from {path}")
    return True
