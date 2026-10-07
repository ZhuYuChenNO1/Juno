# Copyright 2026 Juno contributors.
# Licensed under the MIT License.
"""Juno policy: QwenGR00T with JEPA predictive latents and MoT reasoning."""
import copy
import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from PIL import Image
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from deployment.model_server.tools.image_tools import to_pil_preserve
from juno.model.framework.share_tools import merge_framework_config
from juno.model.framework.VLM4A.QwenGR00T import Qwen_GR00T, QwenGR00TDefaultConfig
from juno.model.modules.world_model.JEPA import JEPAEncoderForAlignment
from juno.model.tools import FRAMEWORK_REGISTRY
from juno.training.trainer_utils import initialize_overwatch
from juno.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)
IGNORE_INDEX = -100


def _juno_jepa_defaults() -> dict:
    return {
        "ckpt_path": "",
        "ckpt_source": "lewm",
        "encoder_scale": "base",
        "image_size": 224,
        "patch_size": 14,
        "embed_dim": None,
        "projector_hidden_dim": 2048,
        "num_reasoning_queries": 8,
        "num_future_frames": 8,
        "frame_stride": 2,
        "loss_type": "cosine_smoothl1",
        "smoothl1_weight": 0.1,
        "align_loss_weight_max": 1.0,
        "align_loss_warmup_steps": 2000,
        "align_loss_plateau_steps": 4000,
        "align_loss_decay_steps": 20000,
        "align_loss_min": 0.05,
        "input_fusion_gate_init": 0.0,
        "input_xattn_scale": 0.2,
        "input_xattn_heads": 8,
        "input_xattn_dropout": 0.0,
        "use_input_fusion": False,
        "use_fm_jepa_prefix": False,
    }


@dataclass
class JunoPolicyDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "juno-policy"
    jepa: dict = field(default_factory=_juno_jepa_defaults)


def _alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str = "cosine_smoothl1",
    smoothl1_weight: float = 0.1,
    detach_target: bool = True,
) -> torch.Tensor:
    """Loss between predicted latent reasoning embeddings and alignment targets.
    Args:
        pred:   (B, K, D) — predicted embeddings (will be normalised).
        target: (B, K, D) — target embeddings (will be normalised; optionally detached).
        detach_target: If True, stop gradient through target (default, for frozen targets).
    """
    target = F.normalize(target.float(), dim=-1)
    if detach_target:
        target = target.detach()
    pred_norm = F.normalize(pred.float(), dim=-1)
    if loss_type == "mse":
        return F.mse_loss(pred_norm, target)
    if loss_type == "cosine_smoothl1":
        cos = (pred_norm * target).sum(dim=-1)
        cos_loss = (1.0 - cos).mean()
        l1 = F.smooth_l1_loss(pred_norm, target, beta=0.1)
        return cos_loss + smoothl1_weight * l1
    raise ValueError(f"Unknown loss_type: {loss_type}")


class JEPAInputResidualFuser(nn.Module):
    """Normalize projected JEPA latents and apply a learnable residual gate."""

    def __init__(self, hidden_size: int, gate_init: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    @property
    def fusion_weight(self) -> torch.Tensor:
        return torch.tanh(self.gate)

    def forward(self, projected_jepa: torch.Tensor) -> torch.Tensor:
        return self.fusion_weight * self.norm(projected_jepa)


class JEPAPatchCrossAttentionFuser(nn.Module):
    """Fuse Qwen image slots with projected JEPA patch tokens."""

    def __init__(
        self, hidden_size: int, *, num_heads: int = 8, dropout: float = 0.0, scale: float = 0.2
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "scale", torch.tensor(float(scale), dtype=torch.float32), persistent=True
        )

    def forward(
        self, qwen_image_tokens: torch.Tensor, jepa_patch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_norm(qwen_image_tokens)
        kv = self.kv_norm(jepa_patch_tokens)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        delta = self.dropout(attn_out) * self.scale.to(device=attn_out.device, dtype=attn_out.dtype)
        return (qwen_image_tokens + delta, delta)


def _make_mot_layer(qwen_layer: nn.Module) -> nn.Module:
    """Deep-copy the reasoning-specific submodules from one Qwen decoder layer.
    Using ``deepcopy`` warm-starts the reasoning branch from the pretrained
    weights and guarantees identical shapes/dtypes without hard-coding config.
    """
    attn = qwen_layer.self_attn
    mot = nn.Module()
    mot.input_layernorm = copy.deepcopy(qwen_layer.input_layernorm)
    mot.q_proj = copy.deepcopy(attn.q_proj)
    mot.k_proj = copy.deepcopy(attn.k_proj)
    mot.v_proj = copy.deepcopy(attn.v_proj)
    mot.o_proj = copy.deepcopy(attn.o_proj)
    mot.q_norm = copy.deepcopy(attn.q_norm)
    mot.k_norm = copy.deepcopy(attn.k_norm)
    mot.post_attention_layernorm = copy.deepcopy(qwen_layer.post_attention_layernorm)
    mot.mlp = copy.deepcopy(qwen_layer.mlp)
    return mot


def _make_mot_forward(framework, layer_idx: int, num_q: int, orig_forward):
    """Build a per-layer forward that routes the last ``num_q`` tokens through
    the reasoning branch while sharing one joint attention with the V+L tokens.
    """

    def mot_forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        if not getattr(framework, "_mot_active", False) or hidden_states.size(1) <= num_q:
            return orig_forward(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        mot = framework.mot_reasoning_layers[layer_idx]
        attn = self.self_attn
        head_dim = attn.head_dim
        k = num_q
        residual = hidden_states
        vl_in = self.input_layernorm(hidden_states[:, :-k, :])
        rq_in = mot.input_layernorm(hidden_states[:, -k:, :])

        def _project(x, q_proj, k_proj, v_proj, q_norm, k_norm):
            b, s, _ = x.shape
            shape = (b, s, -1, head_dim)
            q = q_norm(q_proj(x).view(shape))
            k_ = k_norm(k_proj(x).view(shape))
            v = v_proj(x).view(shape)
            return (q, k_, v)

        q_vl, k_vl, v_vl = _project(
            vl_in, attn.q_proj, attn.k_proj, attn.v_proj, attn.q_norm, attn.k_norm
        )
        q_rq, k_rq, v_rq = _project(
            rq_in, mot.q_proj, mot.k_proj, mot.v_proj, mot.q_norm, mot.k_norm
        )
        query_states = torch.cat([q_vl, q_rq], dim=1).transpose(1, 2)
        key_states = torch.cat([k_vl, k_rq], dim=1).transpose(1, 2)
        value_states = torch.cat([v_vl, v_rq], dim=1).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx, cache_kwargs
            )
        attention_interface = eager_attention_forward
        if attn.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]
        attn_output, _ = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            **kwargs,
        )
        b, s = hidden_states.shape[:2]
        attn_output = attn_output.reshape(b, s, -1).contiguous()
        o_vl = attn.o_proj(attn_output[:, :-k, :])
        o_rq = mot.o_proj(attn_output[:, -k:, :])
        hidden_states = residual + torch.cat([o_vl, o_rq], dim=1)
        residual = hidden_states
        p_vl = self.mlp(self.post_attention_layernorm(hidden_states[:, :-k, :]))
        p_rq = mot.mlp(mot.post_attention_layernorm(hidden_states[:, -k:, :]))
        hidden_states = residual + torch.cat([p_vl, p_rq], dim=1)
        return hidden_states

    return mot_forward


@FRAMEWORK_REGISTRY.register("juno-policy")
class JunoPolicy(Qwen_GR00T):
    """Juno policy with predictive visual latents and a MoT reasoning branch.

    The implementation is intentionally self-contained: JEPA encoding, future-frame
    alignment, input patch fusion, and MoT layer surgery all live in this class.
    Only the general QwenGR00T action-policy interface is inherited.
    """

    default_config_cls = JunoPolicyDefaultConfig
    detach_alignment_target = True

    def __init__(self, config=None, **kwargs):
        nn.Module.__init__(self)
        self.config = merge_framework_config(self.default_config_cls, config)
        from juno.model.modules.vlm import get_vlm_model
        from juno.model.modules.action_model.GR00T_ActionHeader import (
            FlowmatchingActionHead,
            get_action_model,
        )

        # Build the Qwen3-VL backbone and let its hidden size determine the action head.
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
        jepa_cfg = self.config.framework.jepa
        self.use_fm_jepa_prefix = bool(jepa_cfg.get("use_fm_jepa_prefix", False))
        self.config.framework.action_model.use_jepa_fm_prefix = self.use_fm_jepa_prefix
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.num_reasoning_queries = int(jepa_cfg.get("num_reasoning_queries", 8))
        self.num_future_frames = int(jepa_cfg.get("num_future_frames", 8))
        self.frame_stride = int(jepa_cfg.get("frame_stride", 2))
        self.loss_type = str(jepa_cfg.get("loss_type", "cosine_smoothl1"))
        self.smoothl1_weight = float(jepa_cfg.get("smoothl1_weight", 0.1))
        self._align_w_max = float(jepa_cfg.get("align_loss_weight_max", 1.0))
        self._align_w_min = float(jepa_cfg.get("align_loss_min", 0.05))
        self._warmup_steps = int(jepa_cfg.get("align_loss_warmup_steps", 2000))
        self._plateau_steps = int(jepa_cfg.get("align_loss_plateau_steps", 4000))
        self._decay_steps = int(jepa_cfg.get("align_loss_decay_steps", 20000))
        if self.frame_stride * self.num_future_frames > self.action_horizon:
            logger.warning(
                f"jepa.frame_stride * num_future_frames ({self.frame_stride}*{self.num_future_frames}) exceeds action_horizon ({self.action_horizon})."
            )
        ckpt_src = str(jepa_cfg.get("ckpt_source", "lewm")).lower()
        ckpt_path = str(jepa_cfg.get("ckpt_path", "") or "")
        jepa_kwargs = dict(
            encoder_scale=str(jepa_cfg.get("encoder_scale", "base")),
            image_size=int(jepa_cfg.get("image_size", 224)),
            patch_size=int(jepa_cfg.get("patch_size", 14)),
            embed_dim=jepa_cfg.get("embed_dim", None),
            projector_hidden_dim=int(jepa_cfg.get("projector_hidden_dim", 2048)),
        )
        # The target encoder is frozen. It supplies future visual latents but receives
        # no gradients from either the action loss or the alignment loss.
        if ckpt_src == "lewm" and ckpt_path:
            self.jepa_target = JEPAEncoderForAlignment.load_from_lewm_ckpt(ckpt_path, **jepa_kwargs)
        elif ckpt_src == "juno" and ckpt_path:
            self.jepa_target = JEPAEncoderForAlignment.load_from_juno_ckpt(ckpt_path, **jepa_kwargs)
        else:
            logger.warning(
                "No JEPA checkpoint provided; using a random JEPA target. Use this only for plumbing tests."
            )
            self.jepa_target = JEPAEncoderForAlignment(**jepa_kwargs)
        for p in self.jepa_target.parameters():
            p.requires_grad = False
        self.jepa_target.eval()
        vlm_hidden = self.qwen_vl_interface.model.config.hidden_size
        jepa_embed = self.jepa_target.embed_dim
        if self.num_reasoning_queries <= 0:
            self.reasoning_queries = nn.Parameter(torch.zeros(0))
        else:
            self.reasoning_queries = nn.Parameter(
                torch.zeros(self.num_reasoning_queries, vlm_hidden)
            )
            nn.init.normal_(self.reasoning_queries, mean=0.0, std=0.02)
        # Learnable query tokens are appended to the Qwen sequence and later aligned
        # with the corresponding future-frame JEPA embeddings.
        self.reasoning_query_norm = nn.LayerNorm(vlm_hidden)
        self.jepa_projector = nn.Sequential(
            nn.Linear(jepa_embed, vlm_hidden), nn.GELU(), nn.Linear(vlm_hidden, vlm_hidden)
        )
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.long), persistent=False)
        gate_init = float(self.config.framework.jepa.get("input_fusion_gate_init", 0.0))
        vlm_hidden = int(self.qwen_vl_interface.model.config.hidden_size)
        self.jepa_input_fuser = JEPAInputResidualFuser(vlm_hidden, gate_init=gate_init)
        self._last_jepa_input_diagnostics = {}
        self.jepa_input_projector = self.jepa_projector
        del self.jepa_projector
        vlm_hidden = int(self.qwen_vl_interface.model.config.hidden_size)
        jepa_embed = int(self.jepa_target.embed_dim)
        self.reasoning_projector = nn.Sequential(
            nn.Linear(vlm_hidden, vlm_hidden), nn.GELU(), nn.Linear(vlm_hidden, jepa_embed)
        )
        if hasattr(self, "jepa_input_fuser"):
            del self.jepa_input_fuser
        if hasattr(self, "jepa_input_projector"):
            del self.jepa_input_projector
        jepa_cfg = self.config.framework.jepa
        vlm_hidden = int(self.qwen_vl_interface.model.config.hidden_size)
        jepa_hidden = int(self.jepa_target.hidden_size)
        self.jepa_patch_projector = nn.Sequential(
            nn.Linear(jepa_hidden, vlm_hidden), nn.GELU(), nn.Linear(vlm_hidden, vlm_hidden)
        )
        # JEPA patch tokens condition the native Qwen image-token slots through a
        # rectangular cross-attention block; no extra sequence tokens are added.
        self.jepa_input_xattn_fuser = JEPAPatchCrossAttentionFuser(
            vlm_hidden,
            num_heads=int(jepa_cfg.get("input_xattn_heads", 8)),
            dropout=float(jepa_cfg.get("input_xattn_dropout", 0.0)),
            scale=float(jepa_cfg.get("input_xattn_scale", 0.2)),
        )
        self._last_jepa_input_diagnostics = {}
        self.use_jepa_input_fusion = bool(self.config.framework.jepa.get("use_input_fusion", False))
        logger.info("JEPA6 use_input_fusion: %s", self.use_jepa_input_fusion)
        # The standalone launcher supplies the shared Bridge/Fractal checkpoint at runtime.
        self.use_jepa_input_fusion = True
        # Clone Qwen decoder parameters for the reasoning-token branch. The patched
        # layer keeps one joint attention operation while using branch-specific norms,
        # projections, and MLPs.
        self._mot_active = False
        text_layers = self._qwen_text_layers()
        self.mot_reasoning_layers = nn.ModuleList([_make_mot_layer(layer) for layer in text_layers])
        num_q = self.num_reasoning_queries
        for layer_idx, layer in enumerate(text_layers):
            orig_forward = layer.forward
            layer.forward = types.MethodType(
                _make_mot_forward(self, layer_idx, num_q, orig_forward), layer
            )
        logger.info(
            "juno-policy installed %d MoT reasoning layers (K=%d reasoning queries), warm-started from Qwen weights; V+L and reasoning branches train jointly.",
            len(self.mot_reasoning_layers),
            num_q,
        )

    def _register_reasoning_input_hooks(self, qwen_inputs, batch_images) -> List:
        """Register temporary hooks that modify Qwen inputs for one forward pass."""
        embed_layer = self.qwen_vl_interface.model.get_input_embeddings()
        num_q = self.num_reasoning_queries
        if num_q <= 0:
            return []
        queries = self.reasoning_query_norm(self.reasoning_queries)

        def _inject_reasoning_embeddings(module, input, output):
            q = queries.unsqueeze(0).expand(output.size(0), -1, -1)
            q = q.to(device=output.device, dtype=output.dtype)
            output = output.clone()
            output[:, -num_q:, :] = q
            return output

        return [embed_layer.register_forward_hook(_inject_reasoning_embeddings)]

    def set_global_step(self, step: int) -> None:
        self._global_step.fill_(int(step))

    def _curriculum_weight(self, step: Optional[int] = None) -> float:
        s = int(self._global_step.item() if step is None else step)
        w_max = self._align_w_max
        w_min = self._align_w_min
        warm = self._warmup_steps
        plateau = self._plateau_steps
        decay = self._decay_steps
        if s < warm and warm > 0:
            return w_max * (s / max(1, warm))
        if s < warm + plateau:
            return w_max
        if s < warm + plateau + decay and decay > 0:
            t = (s - warm - plateau) / decay
            return w_min + 0.5 * (w_max - w_min) * (1.0 + math.cos(math.pi * t))
        return w_min

    def _append_reasoning_queries(self, qwen_inputs) -> dict:
        """Extend the multimodal Qwen sequence with reasoning-query placeholders.
        Keeping the ``input_ids`` path lets Qwen3-VL compute multimodal M-RoPE
        from image placeholders. The actual reasoning-query embeddings are
        injected by a temporary hook in ``_qwen_forward_with_reasoning_tokens``.
        """
        if self.num_reasoning_queries <= 0:
            return qwen_inputs
        qwen_inputs = {k: v for k, v in qwen_inputs.items()}
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs.get("attention_mask", None)
        batch_size = input_ids.size(0)
        pad_token_id = getattr(self.qwen_vl_interface.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        dummy_ids = torch.full(
            (batch_size, self.num_reasoning_queries),
            int(pad_token_id),
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        qwen_inputs["input_ids"] = torch.cat([input_ids, dummy_ids], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        query_mask = torch.ones(
            (batch_size, self.num_reasoning_queries),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        qwen_inputs["attention_mask"] = torch.cat([attention_mask, query_mask], dim=1)
        if "labels" in qwen_inputs and qwen_inputs["labels"] is not None:
            labels = qwen_inputs["labels"]
            query_labels = torch.full(
                (batch_size, self.num_reasoning_queries),
                IGNORE_INDEX,
                device=labels.device,
                dtype=labels.dtype,
            )
            qwen_inputs["labels"] = torch.cat([labels, query_labels], dim=1)
        qwen_inputs.pop("position_ids", None)
        qwen_inputs.pop("cache_position", None)
        qwen_inputs.pop("rope_deltas", None)
        return qwen_inputs

    def _extract_examples(self, examples: List[dict]):
        batch_images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None
        future_images = (
            [ex["future_image"] for ex in examples] if "future_image" in examples[0] else None
        )
        return (batch_images, instructions, actions, state, future_images)

    def _encode_future_frames(
        self, future_images_per_sample: List[List[Image.Image]], device: torch.device
    ) -> torch.Tensor:
        import torchvision.transforms.functional as TVF

        target_size = int(self.config.framework.jepa.image_size)
        frames: List[torch.Tensor] = []
        for sample in future_images_per_sample:
            if not sample:
                raise ValueError("future_image is empty; cannot compute JEPA alignment target.")
            if len(sample) > self.num_future_frames:
                sample = sample[: self.num_future_frames]
            elif len(sample) < self.num_future_frames:
                sample = list(sample) + [sample[-1]] * (self.num_future_frames - len(sample))
            per_sample_frames = []
            for img in sample:
                if not isinstance(img, Image.Image):
                    img = to_pil_preserve(img)
                tensor = TVF.to_tensor(img.convert("RGB"))
                tensor = TVF.resize(tensor, [target_size, target_size], antialias=True)
                per_sample_frames.append(tensor)
            frames.append(torch.stack(per_sample_frames, dim=0))
        batch = torch.stack(frames, dim=0).to(device=device, dtype=torch.float32)
        return self.jepa_target(batch)

    def _register_qwen_input_hooks(self, qwen_inputs, batch_images) -> List:
        """Inject reasoning queries and, when enabled, JEPA patch residuals."""
        hook_handles = []
        if getattr(self, "use_reasoning_tokens", True):
            hook_handles = self._register_reasoning_input_hooks(qwen_inputs, batch_images)
        if not getattr(self, "use_jepa_input_fusion", True):
            return hook_handles
        input_ids = qwen_inputs["input_ids"]
        image_token_id = getattr(self.qwen_vl_interface.model.config, "image_token_id", None)
        if image_token_id is None:
            raise RuntimeError("Qwen backbone config does not expose image_token_id.")
        image_runs = self._find_image_token_runs(input_ids, int(image_token_id))
        num_image_runs = sum((len(sample_runs) for sample_runs in image_runs))
        expected_frames = len(self._flatten_current_images(batch_images))
        if num_image_runs != expected_frames:
            raise ValueError(
                f"JEPA input cross-attention expected {expected_frames} image-token runs, but Qwen inputs contain {num_image_runs}."
            )
        projector_param = next(self.jepa_patch_projector.parameters())
        jepa_patches = self._encode_input_patch_tokens(batch_images, device=projector_param.device)
        jepa_patches = jepa_patches.to(dtype=projector_param.dtype)
        projected_patches = self.jepa_patch_projector(jepa_patches)
        if projected_patches.size(0) != num_image_runs:
            raise RuntimeError(
                f"JEPA input cross-attention produced {projected_patches.size(0)} patch groups for {num_image_runs} image-token runs."
            )
        qwen_run_lengths = [end - start for sample_runs in image_runs for start, end in sample_runs]
        with torch.no_grad():
            self._last_jepa_input_diagnostics = {
                "jepa_input_xattn_scale": self.jepa_input_xattn_fuser.scale.detach(),
                "jepa_input_patch_norm": projected_patches.detach().float().norm(dim=-1).mean(),
                "jepa_input_qwen_tokens_per_image": torch.tensor(
                    sum(qwen_run_lengths) / max(1, len(qwen_run_lengths)),
                    device=projected_patches.device,
                    dtype=torch.float32,
                ),
                "jepa_input_jepa_patches_per_image": torch.tensor(
                    projected_patches.size(1), device=projected_patches.device, dtype=torch.float32
                ),
            }
        language_model = getattr(
            getattr(self.qwen_vl_interface.model, "model", None), "language_model", None
        )
        if language_model is None:
            raise RuntimeError("JEPA6 requires a Qwen2.5-VL or Qwen3-VL language_model input.")

        def _inject_jepa_cross_attention(module, args, kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is None:
                raise RuntimeError("Qwen language_model did not receive inputs_embeds.")
            fused_embeds = inputs_embeds.clone()
            delta_norms = []
            patch_index = 0
            fuser_param = next(self.jepa_input_xattn_fuser.parameters())
            for sample_index, sample_runs in enumerate(image_runs):
                for start, end in sample_runs:
                    image_tokens = inputs_embeds[sample_index : sample_index + 1, start:end, :]
                    image_tokens_for_attn = image_tokens.to(
                        device=fuser_param.device, dtype=fuser_param.dtype
                    )
                    patch_tokens = projected_patches[patch_index : patch_index + 1].to(
                        device=fuser_param.device, dtype=fuser_param.dtype
                    )
                    fused_tokens, delta = self.jepa_input_xattn_fuser(
                        image_tokens_for_attn, patch_tokens
                    )
                    fused_embeds[sample_index : sample_index + 1, start:end, :] = fused_tokens.to(
                        device=fused_embeds.device, dtype=fused_embeds.dtype
                    )
                    delta_norms.append(delta.detach().float().norm(dim=-1).mean())
                    patch_index += 1
            with torch.no_grad():
                self._last_jepa_input_diagnostics["jepa_input_xattn_delta_norm"] = (
                    torch.stack(delta_norms).mean()
                    if delta_norms
                    else torch.zeros((), device=projected_patches.device)
                )
            kwargs = dict(kwargs)
            kwargs["inputs_embeds"] = fused_embeds
            return (args, kwargs)

        try:
            hook_handles.append(
                language_model.register_forward_pre_hook(
                    _inject_jepa_cross_attention, with_kwargs=True
                )
            )
        except Exception:
            for handle in reversed(hook_handles):
                handle.remove()
            raise
        return hook_handles

    def _qwen_forward_with_reasoning_tokens(
        self, batch_images, instructions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Qwen once and return the full sequence plus its reasoning suffix."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        qwen_inputs = self._append_reasoning_queries(qwen_inputs)
        hook_handles = self._register_qwen_input_hooks(qwen_inputs, batch_images)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                if qwenvl_outputs.hidden_states is None:
                    raise RuntimeError("Qwen forward did not return hidden_states.")
                last_hidden = qwenvl_outputs.hidden_states[-1]
        finally:
            for handle in reversed(hook_handles):
                handle.remove()
        reasoning_hidden = last_hidden[:, -self.num_reasoning_queries :, :]
        return (last_hidden, reasoning_hidden)

    def _prepare_alignment_embeddings(
        self, reasoning_hidden: torch.Tensor, future_images, last_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project Qwen reasoning tokens into the frozen JEPA target space."""
        pred_emb = self.reasoning_projector(reasoning_hidden.to(torch.float32))
        with torch.no_grad():
            tgt_emb = self._encode_future_frames(future_images, device=last_hidden.device)
        return (pred_emb, tgt_emb)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        """Compute action loss and optional future-frame JEPA alignment loss."""
        self._last_jepa_input_diagnostics = {}
        batch_images, instructions, actions, state, future_images = self._extract_examples(examples)
        last_hidden, reasoning_hidden = self._qwen_forward_with_reasoning_tokens(
            batch_images, instructions
        )
        with torch.autocast("cuda", dtype=torch.float32):
            actions_t = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )
            actions_target = actions_t[:, -self.action_horizon :, :]
            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            state_repeated = None
            if state is not None:
                state_t = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state_t.repeat(repeated_diffusion_steps, 1, 1)
            jepa_tokens_repeated = None
            if self.use_fm_jepa_prefix:
                jepa_tokens_repeated = reasoning_hidden.repeat(repeated_diffusion_steps, 1, 1)
            action_loss = self.action_model(
                last_hidden_repeated,
                actions_target_repeated,
                state_repeated,
                jepa_tokens=jepa_tokens_repeated,
            )
        out = {"action_loss": action_loss}
        # Future images are present during training. Their JEPA targets supervise the
        # reasoning suffix; action-only examples still optimize the policy normally.
        if future_images is not None:
            pred_emb, tgt_emb = self._prepare_alignment_embeddings(
                reasoning_hidden, future_images, last_hidden
            )
            k = min(pred_emb.size(1), tgt_emb.size(1))
            pred_aligned = pred_emb[:, :k]
            tgt_aligned = tgt_emb[:, :k]
            align_loss = _alignment_loss(
                pred_aligned,
                tgt_aligned,
                loss_type=self.loss_type,
                smoothl1_weight=self.smoothl1_weight,
                detach_target=self.detach_alignment_target,
            )
            with torch.no_grad():
                pred_diag = pred_aligned.float()
                tgt_diag = tgt_aligned.float()
                pred_flat = pred_diag.reshape(-1, pred_diag.size(-1))
                tgt_flat = tgt_diag.reshape(-1, tgt_diag.size(-1))
                pred_norm = F.normalize(pred_diag, dim=-1)
                tgt_norm = F.normalize(tgt_diag, dim=-1)
                diagnostics = {
                    "jepa_pred_norm": pred_diag.norm(dim=-1).mean(),
                    "jepa_pred_std": pred_flat.std(dim=0, unbiased=False).mean(),
                    "jepa_target_norm": tgt_diag.norm(dim=-1).mean(),
                    "jepa_target_std": tgt_flat.std(dim=0, unbiased=False).mean(),
                    "jepa_align_cosine": (pred_norm * tgt_norm).sum(dim=-1).mean(),
                }
            alpha = self._curriculum_weight()
            total = action_loss + alpha * align_loss
            out.update(
                {
                    "align_loss": align_loss,
                    "align_weight": torch.tensor(alpha, device=action_loss.device),
                    "loss": total,
                    **diagnostics,
                }
            )
        else:
            out["loss"] = action_loss
        if self.training:
            self._global_step += 1
        out.update(self._last_jepa_input_diagnostics)
        return out

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(ex["image"]) for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        last_hidden, reasoning_hidden = self._qwen_forward_with_reasoning_tokens(
            batch_images, instructions
        )
        state_t = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        jepa_tokens = reasoning_hidden if self.use_fm_jepa_prefix else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden, state_t, jepa_tokens=jepa_tokens
            )
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    @staticmethod
    def _find_image_token_runs(input_ids: torch.Tensor, image_token_id: int) -> List[List[tuple]]:
        """Return contiguous image-token ranges for each batch row."""
        batch_runs = []
        for row in input_ids:
            sample_runs = []
            start = None
            for index, is_image_token in enumerate(row.eq(image_token_id).tolist() + [False]):
                if is_image_token and start is None:
                    start = index
                elif not is_image_token and start is not None:
                    sample_runs.append((start, index))
                    start = None
            batch_runs.append(sample_runs)
        return batch_runs

    @staticmethod
    def _flatten_current_images(batch_images) -> List[Image.Image]:
        frames = []
        for sample in batch_images:
            sample_images = sample if isinstance(sample, (list, tuple)) else [sample]
            if not sample_images:
                raise ValueError("Current observation has no image views for JEPA input fusion.")
            for image in sample_images:
                image = to_pil_preserve(image)
                if not isinstance(image, Image.Image):
                    raise TypeError(f"Expected a PIL image leaf, got {type(image)}")
                frames.append(image.convert("RGB"))
        return frames

    def _encode_input_frames(self, batch_images, device: torch.device) -> torch.Tensor:
        """Encode all current views independently into frozen JEPA latents."""
        target_size = int(self.config.framework.jepa.image_size)
        frames = [
            TVF.resize(TVF.to_tensor(image), [target_size, target_size], antialias=True)
            for image in self._flatten_current_images(batch_images)
        ]
        if not frames:
            raise ValueError("No current frames available for JEPA input fusion.")
        batch = torch.stack(frames, dim=0).unsqueeze(0)
        batch = batch.to(device=device, dtype=torch.float32)
        return self.jepa_target(batch).squeeze(0)

    def _get_jepa_input_projector(self) -> nn.Module:
        return self.jepa_input_projector

    def _encode_input_patch_tokens(self, batch_images, device: torch.device) -> torch.Tensor:
        """Encode all current views into frozen JEPA raw patch tokens."""
        target_size = int(self.config.framework.jepa.image_size)
        frames = [
            TVF.resize(TVF.to_tensor(image), [target_size, target_size], antialias=True)
            for image in self._flatten_current_images(batch_images)
        ]
        if not frames:
            raise ValueError("No current frames available for JEPA input cross-attention.")
        batch = torch.stack(frames, dim=0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            batch = self.jepa_target._preprocess(batch)
            encoder_out = self.jepa_target.encoder(batch, interpolate_pos_encoding=True)
            patch_tokens = encoder_out.last_hidden_state[:, 1:, :]
        return patch_tokens

    def _qwen_text_layers(self) -> List[nn.Module]:
        """Return the Qwen text decoder layers (Qwen3-VL layout)."""
        language_model = getattr(
            getattr(self.qwen_vl_interface.model, "model", None), "language_model", None
        )
        if language_model is None or not hasattr(language_model, "layers"):
            raise RuntimeError(
                "juno-policy requires a Qwen3-VL backbone exposing model.model.language_model.layers."
            )
        return list(language_model.layers)
