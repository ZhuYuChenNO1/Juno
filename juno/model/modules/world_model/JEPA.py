# Copyright 2025 juno community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
JEPA (Joint-Embedding Predictive Architecture) World Model — Training-only Interface.

Ports the `le-wm` JEPA implementation (https://github.com/<le-wm>) into juno.
This file contains everything needed to *train* the JEPA world model from scratch:

  - ViT encoder: pixels → frame embeddings (CLS token)
  - Action encoder (Embedder): per-step action → action embedding
  - Predictor (ARPredictor): autoregressive transformer with AdaLN-zero
      that predicts next-frame embeddings conditioned on action embeddings
  - Projector + pred_proj: MLPs aligning encoder/predictor feature dims
  - SIGReg: Sketch Isotropic-Gaussian Regularizer (anti-collapse)

The training objective (per `le-wm/train.py::lejepa_forward`):

    emb            = encode(pixels)               # (B, T, D)
    act_emb        = action_encoder(action)        # (B, T, D)
    ctx_emb        = emb[:, :history_size]         # context window
    ctx_act        = act_emb[:, :history_size]
    tgt_emb        = emb[:, num_preds:]            # shifted target
    pred_emb       = predictor(ctx_emb, ctx_act)   # next-step prediction
    pred_loss      = MSE(pred_emb, tgt_emb)
    sigreg_loss    = SIGReg(emb)
    loss           = pred_loss + lambda * sigreg_loss

Inference / planning utilities (rollout / criterion / get_cost) from the
original `jepa.py` are intentionally NOT ported — this file is training-only.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from juno.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


# ---------------------------------------------------------------------------
# Sketch Isotropic-Gaussian Regularizer (single-GPU implementation)
# ---------------------------------------------------------------------------
class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Pushes the marginal distribution of features along random 1-D projections
    toward the standard normal — a tractable surrogate for full isotropy and
    a strong anti-collapse signal for JEPA-style training.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: (T, B, D) — features stacked along the leading time axis.
        """
        # Sample random projections on the unit sphere.
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))

        # Epps–Pulley statistic vs. the standard normal CF.
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------
def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with optional causal masking."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal: bool = True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class Block(nn.Module):
    """Standard pre-norm Transformer block."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    """Transformer trunk with optional AdaLN-zero conditioning blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        )

        self.layers = nn.ModuleList(
            [block_class(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self._is_conditional = block_class is ConditionalBlock

    def forward(self, x, c=None):
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x, c) if self._is_conditional else block(x)
        x = self.norm(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Action embedder and MLP projector
# ---------------------------------------------------------------------------
class Embedder(nn.Module):
    """Per-step action embedder used as conditioning for the predictor."""

    def __init__(
        self,
        input_dim: int = 10,
        smoothed_dim: int = 10,
        emb_dim: int = 10,
        mlp_scale: int = 4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, emb_dim)."""
        x = x.float().permute(0, 2, 1)
        x = self.patch_embed(x).permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    """Two-layer MLP with optional normalization and activation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Autoregressive Predictor
# ---------------------------------------------------------------------------
class ARPredictor(nn.Module):
    """Autoregressive next-state predictor (causal AdaLN-zero Transformer)."""

    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim or input_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) — context embeddings
            c: (B, T, input_dim) — action conditioning
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        return self.transformer(x, c)


# ---------------------------------------------------------------------------
# JEPA — main training module
# ---------------------------------------------------------------------------
# Preset ViT configurations, matching the sizes used by `spt.backbone.utils.vit_hf`.
_VIT_PRESETS = {
    "tiny":  dict(hidden_size=192,  num_hidden_layers=12, num_attention_heads=3,  intermediate_size=768),
    "small": dict(hidden_size=384,  num_hidden_layers=12, num_attention_heads=6,  intermediate_size=1536),
    "base":  dict(hidden_size=768,  num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072),
    "large": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096),
    "huge":  dict(hidden_size=1280, num_hidden_layers=32, num_attention_heads=16, intermediate_size=5120),
}


def _build_vit_encoder(
    scale: str = "base",
    image_size: int = 224,
    patch_size: int = 16,
    pretrained: bool = False,
):
    """Build a HuggingFace ViTModel matching `spt.backbone.utils.vit_hf` defaults."""
    from transformers import ViTConfig, ViTModel

    if scale not in _VIT_PRESETS:
        raise ValueError(f"Unknown ViT scale '{scale}'. Choose from {list(_VIT_PRESETS)}.")

    cfg = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        **_VIT_PRESETS[scale],
    )

    if pretrained:
        return ViTModel.from_pretrained(
            f"google/vit-{scale}-patch{patch_size}-{image_size}",
            add_pooling_layer=False,
        )
    return ViTModel(cfg, add_pooling_layer=False, use_mask_token=False)


class _JEPA_Interface(nn.Module):
    """
    JEPA world model — training interface.

    Components are built from `config.framework.world_model` (with sensible
    defaults matching `le-wm/config/train/lewm_seg02_vitbase.yaml`):

        world_model:
          encoder_scale: "base"           # tiny / small / base / large / huge
          image_size:    224
          patch_size:    14
          history_size:  3                # context window length
          num_preds:     1                # target shift (emb[:, num_preds:])
          embed_dim:     192              # shared latent dim (act / projector)
          action_dim:    2                # raw action dimensionality
          frameskip:     1                # effective_action_dim = frameskip*action_dim
          predictor:
            depth:     6
            heads:     16
            mlp_dim:   2048
            dim_head:  64
            dropout:   0.1
            emb_dropout: 0.0
          sigreg:
            weight:    0.2
            knots:     17
            num_proj:  1024

    The `forward(batch)` method runs one training step and returns a dict
    containing `loss`, `pred_loss`, `sigreg_loss`, `emb`, `act_emb`, `pred_emb`.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config

        wm_cfg = {}
        if config is not None and hasattr(config, "framework"):
            wm_cfg = config.framework.get("world_model", {}) or {}

        def _get(key, default):
            return wm_cfg.get(key, default) if hasattr(wm_cfg, "get") else default

        # ----- top-level hyperparameters -----
        encoder_scale = _get("encoder_scale", "base")
        image_size    = int(_get("image_size", 224))
        patch_size    = int(_get("patch_size", 14))
        pretrained    = bool(_get("pretrained", False))

        self.history_size = int(_get("history_size", 3))
        self.num_preds    = int(_get("num_preds", 1))
        action_dim        = int(_get("action_dim", 7))
        frameskip         = int(_get("frameskip", 1))
        effective_act_dim = frameskip * action_dim

        # ----- encoder -----
        logger.info(
            f"Building JEPA ViT-{encoder_scale} encoder "
            f"(image={image_size}, patch={patch_size}, pretrained={pretrained})"
        )
        self.encoder = _build_vit_encoder(
            scale=encoder_scale,
            image_size=image_size,
            patch_size=patch_size,
            pretrained=pretrained,
        )
        hidden_dim = self.encoder.config.hidden_size
        embed_dim  = int(_get("embed_dim", hidden_dim))

        # ----- predictor -----
        pred_cfg = _get("predictor", {})
        if hasattr(pred_cfg, "get"):
            pred_kwargs = dict(
                depth      = int(pred_cfg.get("depth", 6)),
                heads      = int(pred_cfg.get("heads", 16)),
                mlp_dim    = int(pred_cfg.get("mlp_dim", 2048)),
                dim_head   = int(pred_cfg.get("dim_head", 64)),
                dropout    = float(pred_cfg.get("dropout", 0.1)),
                emb_dropout= float(pred_cfg.get("emb_dropout", 0.0)),
            )
        else:
            pred_kwargs = dict(depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0)

        self.predictor = ARPredictor(
            num_frames=self.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **pred_kwargs,
        )

        # ----- action encoder + MLP projectors -----
        self.action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
        self.projector = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=nn.BatchNorm1d,
        )
        self.pred_proj = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=nn.BatchNorm1d,
        )

        # ----- SIGReg regularizer -----
        sig_cfg = _get("sigreg", {})
        if hasattr(sig_cfg, "get"):
            self.sigreg_weight = float(sig_cfg.get("weight", 0.2))
            self.sigreg = SIGReg(
                knots=int(sig_cfg.get("knots", 17)),
                num_proj=int(sig_cfg.get("num_proj", 1024)),
            )
        else:
            self.sigreg_weight = 0.2
            self.sigreg = SIGReg(knots=17, num_proj=1024)

        # Config-like shim so downstream code can read hidden_size.
        self._hidden_size = hidden_dim

        class _FakeConfig:
            pass

        self._model_config = _FakeConfig()
        self._model_config.hidden_size = hidden_dim

    # ------------------------------------------------------------------ #
    # Compatibility shim — downstream code reads backbone.model.config.hidden_size
    # ------------------------------------------------------------------ #
    @property
    def model(self):
        class _ModelShim:
            pass

        shim = _ModelShim()
        shim.config = self._model_config
        return shim

    # ------------------------------------------------------------------ #
    # Core training APIs
    # ------------------------------------------------------------------ #
    def encode(self, info: dict) -> dict:
        """Encode observations (and optionally actions) into latent embeddings.

        Args:
            info: dict with at least `pixels`: (B, T, C, H, W).
                Optionally `action`: (B, T, action_dim).

        Returns:
            info dict mutated in-place with new keys:
                - `emb`:     (B, T, embed_dim)
                - `act_emb`: (B, T, embed_dim)   [only if `action` present]
        """
        pixels = info["pixels"].float()
        b = pixels.size(0)

        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        cls_emb = output.last_hidden_state[:, 0]  # CLS token: ((b t), hidden_dim)

        emb = self.projector(cls_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Predict next-state embeddings.

        Args:
            emb:     (B, T, embed_dim)
            act_emb: (B, T, embed_dim)

        Returns:
            preds:   (B, T, embed_dim)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    def forward(self, batch: dict, **kwargs) -> dict:
        """One JEPA training step.

        Args:
            batch: dict with
                - `pixels`: (B, T, C, H, W)
                - `action`: (B, T, action_dim) — NaNs at sequence boundaries are
                  replaced with 0 (matches the `le-wm` behaviour).

        Returns:
            dict with `loss`, `pred_loss`, `sigreg_loss`, and intermediate
            tensors (`emb`, `act_emb`, `pred_emb`) for downstream logging.
        """
        ctx_len = self.history_size
        n_preds = self.num_preds
        lambd   = self.sigreg_weight

        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        output = self.encode(batch)
        emb     = output["emb"]      # (B, T, D)
        act_emb = output["act_emb"]  # (B, T, D)

        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:]

        pred_emb = self.predict(ctx_emb, ctx_act)

        # Truncate to the common time horizon (defensive — handles non-default
        # ctx_len / num_preds combinations where the two windows are unequal).
        T_common = min(pred_emb.size(1), tgt_emb.size(1))
        pred_loss = (pred_emb[:, :T_common] - tgt_emb[:, :T_common]).pow(2).mean()

        sigreg_loss = self.sigreg(emb.transpose(0, 1))  # SIGReg expects (T, B, D)

        loss = pred_loss + lambd * sigreg_loss

        output.update(
            pred_emb=pred_emb,
            pred_loss=pred_loss,
            sigreg_loss=sigreg_loss,
            loss=loss,
        )
        return output


# ---------------------------------------------------------------------------
# JEPAEncoderForAlignment — frozen target encoder for latent-reasoning alignment
# ---------------------------------------------------------------------------
# ImageNet stats used by le-wm's `get_img_preprocessor`. Kept here so the target
# pipeline is self-contained and doesn't reuse the VLM processor.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class JEPAEncoderForAlignment(nn.Module):
    """Frozen JEPA target encoder for latent-reasoning supervision.

    This wraps only the **encoder + projector** subset of `_JEPA_Interface`
    (predictor / action_encoder / sigreg are discarded), exposes a clean
    `forward(frames) -> projector_output` API, and is intended to be held
    by downstream frameworks as a non-trainable target network.

    Typical use::

        target = JEPAEncoderForAlignment.load_from_lewm_ckpt(
            ckpt_path, encoder_scale="base", image_size=224, patch_size=14,
        )
        target = target.to(device).eval()
        with torch.no_grad():
            emb = target(future_frames)  # (B, K, embed_dim)

    Implementation notes:
      * `requires_grad_(False)` is enforced at construction time and `.eval()`
        is called whenever `.train(mode)` is invoked, so dropout / batchnorm
        statistics stay frozen even when the parent framework is in train mode.
      * Image preprocessing (ImageNet normalize + resize) is done here so the
        caller can pass `[B, K, 3, H, W]` of *raw* pixels in [0, 1] (or *any*
        H, W) — internally we resize to the configured `image_size`.
    """

    def __init__(
        self,
        *,
        encoder_scale: str = "base",
        image_size: int = 224,
        patch_size: int = 14,
        embed_dim: Optional[int] = None,
        projector_hidden_dim: int = 2048,
    ):
        super().__init__()

        logger.info(
            f"Building JEPAEncoderForAlignment ViT-{encoder_scale} "
            f"(image={image_size}, patch={patch_size})"
        )
        self.image_size = image_size

        self.encoder = _build_vit_encoder(
            scale=encoder_scale,
            image_size=image_size,
            patch_size=patch_size,
            pretrained=False,
        )
        hidden_dim = self.encoder.config.hidden_size
        embed_dim = int(embed_dim or hidden_dim)

        self.projector = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=projector_hidden_dim,
            norm_fn=nn.BatchNorm1d,
        )

        self._hidden_size = hidden_dim
        self._embed_dim = embed_dim

        self.register_buffer(
            "_norm_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_norm_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.requires_grad_(False)
        self.eval()

    # Properties for downstream code to inspect target dimensions.
    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def train(self, mode: bool = True):
        """Keep the target network in eval mode regardless of the parent's state."""
        return super().train(False)

    # ------------------------------------------------------------------ #
    # Checkpoint loading                                                  #
    # ------------------------------------------------------------------ #
    @classmethod
    def load_from_lewm_ckpt(
        cls,
        ckpt_path: str,
        *,
        encoder_scale: str = "base",
        image_size: int = 224,
        patch_size: int = 14,
        embed_dim: Optional[int] = None,
        projector_hidden_dim: int = 2048,
        strict: bool = False,
    ) -> "JEPAEncoderForAlignment":
        """Load weights from a `le-wm` checkpoint produced by `ModelObjectCallBack`.

        `le-wm` saves the **entire `JEPA` nn.Module via `torch.save(model, ...)`**.
        We therefore load with `weights_only=False` and pull weights out by name.

        Only `encoder.*` and `projector.*` entries are consumed; everything else
        (predictor, action_encoder, pred_proj, sigreg) is intentionally dropped.
        """
        instance = cls(
            encoder_scale=encoder_scale,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            projector_hidden_dim=projector_hidden_dim,
        )

        logger.info(f"Loading le-wm JEPA checkpoint from {ckpt_path}")
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Tolerate three packaging styles:
        #   1. raw nn.Module (le-wm ModelObjectCallBack default)
        #   2. dict containing a state_dict (e.g. lightning ckpt)
        #   3. plain state_dict
        if isinstance(obj, nn.Module):
            source_state = obj.state_dict()
        elif isinstance(obj, dict) and "state_dict" in obj:
            source_state = obj["state_dict"]
        elif isinstance(obj, dict):
            source_state = obj
        else:
            raise TypeError(
                f"Unsupported ckpt object type {type(obj)}; expected nn.Module / dict."
            )

        # le-wm's JEPA module wraps fields directly (encoder, projector, ...);
        # spt.Module wraps further with a 'model.' prefix. Strip whichever is present.
        cleaned = {}
        for k, v in source_state.items():
            key = k
            if key.startswith("model."):
                key = key[len("model."):]
            if key.startswith("encoder.") or key.startswith("projector."):
                cleaned[key] = v

        missing, unexpected = instance.load_state_dict(cleaned, strict=False)
        logger.info(
            f"le-wm ckpt: loaded {len(cleaned)} tensors; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Strict load failed. missing={missing[:5]} ... unexpected={unexpected[:5]} ..."
            )

        instance.requires_grad_(False)
        instance.eval()
        return instance

    @classmethod
    def load_from_juno_ckpt(
        cls,
        ckpt_path: str,
        *,
        encoder_scale: str = "base",
        image_size: int = 224,
        patch_size: int = 14,
        embed_dim: Optional[int] = None,
        projector_hidden_dim: int = 2048,
        strict: bool = False,
    ) -> "JEPAEncoderForAlignment":
        """Load weights saved by `juno/training/pretrain_jepa.py`.

        The juno-side pretraining saves a state_dict whose keys directly
        match `_JEPA_Interface` (no `model.` prefix). We filter to the
        encoder/projector subset just like `load_from_lewm_ckpt`.
        """
        instance = cls(
            encoder_scale=encoder_scale,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            projector_hidden_dim=projector_hidden_dim,
        )

        logger.info(f"Loading juno JEPA checkpoint from {ckpt_path}")
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(obj, dict) and "state_dict" in obj:
            source_state = obj["state_dict"]
        elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
            source_state = obj["model"]
        elif isinstance(obj, dict):
            source_state = obj
        elif isinstance(obj, nn.Module):
            source_state = obj.state_dict()
        else:
            raise TypeError(f"Unsupported ckpt object type {type(obj)}")

        cleaned = {
            k: v
            for k, v in source_state.items()
            if k.startswith("encoder.") or k.startswith("projector.")
        }
        missing, unexpected = instance.load_state_dict(cleaned, strict=False)
        logger.info(
            f"juno ckpt: loaded {len(cleaned)} tensors; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Strict load failed. missing={missing[:5]} ... unexpected={unexpected[:5]} ..."
            )

        instance.requires_grad_(False)
        instance.eval()
        return instance

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #
    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """Resize + normalize frames so callers can pass raw [0, 1] tensors.

        Args:
            frames: (N, 3, H, W) float tensor, values either in [0, 1] or
                already normalised (we re-normalise either way).
        """
        if frames.shape[-2:] != (self.image_size, self.image_size):
            frames = F.interpolate(
                frames, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )

        # Values outside ~[0, 1] are interpreted as already-normalised; skip.
        if frames.min() >= -0.01 and frames.max() <= 1.01:
            frames = (frames - self._norm_mean) / self._norm_std
        return frames

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames into JEPA target embeddings.

        Args:
            frames: (B, K, 3, H, W) tensor of pixel observations.
                Accepts either [0, 1] (will be normalised) or pre-normalised.

        Returns:
            embeddings: (B, K, embed_dim) — projector output, NOT L2-normalised.
                Caller is expected to apply `F.normalize` + `.detach()` before
                using as supervision target.
        """
        assert frames.dim() == 5, (
            f"Expected (B, K, C, H, W); got {tuple(frames.shape)}"
        )
        b, k = frames.shape[:2]
        x = rearrange(frames.float(), "b k c h w -> (b k) c h w")
        x = self._preprocess(x)

        encoder_out = self.encoder(x, interpolate_pos_encoding=True)
        cls_emb = encoder_out.last_hidden_state[:, 0]  # ((B*K), hidden)
        emb = self.projector(cls_emb)                    # ((B*K), embed_dim)
        return rearrange(emb, "(b k) d -> b k d", b=b, k=k)
