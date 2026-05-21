from __future__ import annotations

from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Conv3×3 → GN → GELU → Conv3×3 → GN → GELU"""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Bilinear 2× upsample → (optional concat with skip) → DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # Handle odd-dimension mismatches
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RefinerBlock(nn.Module):
    """
    Residual dilated block for multi-scale context at full resolution.

    x + Conv1×1(GELU(GN(Conv3×3_dilated(x))))

    See DESIGN.md §3.3.
    """

    def __init__(self, channels: int, dilation: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, 3,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.gn = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.gn(self.conv1(x))))


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class AVSegmenter(nn.Module):
    """
    4-class retinal A/V segmenter.

    Architecture:
        ConvNeXt-S encoder → UNet decoder (up1–up5) → Refiner (dil 2, 4)
        → Dropout2d → Conv2d(48, 4) → logits (B, 4, H, W)
    """

    # Encoder channel sizes (ConvNeXt-Small)
    _ENC_CH = (96, 192, 384, 768)

    def __init__(
        self,
        encoder_name: str = "convnext_small.dinov3_lvd1689m",
        pretrained: bool = True,
        num_classes: int = 4,
        refiner_channels: int = 48,
        refiner_dilations: List[int] = (2, 4),
        decoder_dropout: float = 0.1,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes

        # Store config for checkpoint serialisation
        self.config = dict(
            encoder_name=encoder_name,
            pretrained=pretrained,
            num_classes=num_classes,
            refiner_channels=refiner_channels,
            refiner_dilations=list(refiner_dilations),
            decoder_dropout=decoder_dropout,
            use_checkpoint=use_checkpoint,
        )

        # ── Encoder ──────────────────────────────────────────────────────
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[0, 1, 2, 3],
        )
        self._init_chromatic_stem()

        # ── UNet Decoder (up1–up5) ───────────────────────────────────────
        C = self._ENC_CH
        self.up1 = UpBlock(C[3], C[2], C[2])           # 768→384 @ H/16
        self.up2 = UpBlock(C[2], C[1], C[1])            # 384→192 @ H/8
        self.up3 = UpBlock(C[1], C[0], C[0])            # 192→96  @ H/4
        self.up4 = UpBlock(C[0], 0, refiner_channels)   # 96→48   @ H/2  (no skip)
        self.up5 = UpBlock(refiner_channels, 0, refiner_channels)  # 48→48 @ H (no skip)

        # ── Refiner ──────────────────────────────────────────────────────
        self.refiner = nn.ModuleList([
            RefinerBlock(refiner_channels, d) for d in refiner_dilations
        ])

        # ── Head ─────────────────────────────────────────────────────────
        self.drop = nn.Dropout2d(decoder_dropout)
        self.head = nn.Conv2d(refiner_channels, num_classes, 1)
        self._init_head()

    # ── Initialisation helpers ───────────────────────────────────────────

    def _init_chromatic_stem(self):
        """Replace blue-channel stem weights with (R − G) / 2 initialisation.
        See DESIGN.md §4.3."""
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d) and m.weight.shape[1] == 3:
                with torch.no_grad():
                    m.weight[:, 2, :, :] = (
                        m.weight[:, 0, :, :] - m.weight[:, 1, :, :]
                    ) / 2.0
                break  # Only the first 3-channel conv (stem)

    def _init_head(self):
        """Near-zero head initialisation for balanced initial predictions."""
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    # ── Encoder / refiner with optional gradient checkpointing ───────────

    def _encode(self, x: torch.Tensor) -> list:
        if self.use_checkpoint and self.training:
            # Wrap full encoder call in checkpoint
            def _enc_fn(inp):
                return self.encoder(inp)
            # checkpoint requires at least one tensor input
            feats = checkpoint(_enc_fn, x, use_reentrant=False)
        else:
            feats = self.encoder(x)
        return feats

    def _refine(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.refiner:
            if self.use_checkpoint and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: dict with key ``"image"`` → (B, 3, H, W) float32.

        Returns:
            logits: (B, 4, H, W) raw logits (softmax applied externally).
        """
        x = batch["image"]
        H_in, W_in = x.shape[2:]

        # Encoder
        f1, f2, f3, f4 = self._encode(x)  # H/4, H/8, H/16, H/32

        # Decoder
        d = self.up1(f4, skip=f3)   # H/16
        d = self.up2(d,  skip=f2)   # H/8
        d = self.up3(d,  skip=f1)   # H/4
        d = self.up4(d)             # H/2
        d = self.up5(d)             # H

        # Ensure output matches input spatial size exactly
        if d.shape[2:] != (H_in, W_in):
            d = F.interpolate(d, size=(H_in, W_in), mode="bilinear", align_corners=False)

        # Refiner
        d = self._refine(d)

        # Head
        logits = self.head(self.drop(d))
        return logits

    # ── Freeze / unfreeze helpers (train.py calls these) ─────────────────

    def freeze_encoder(self):
        """Freeze all encoder parameters (warmup phase)."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters (after warmup)."""
        for p in self.encoder.parameters():
            p.requires_grad = True

    # ── Parameter groups for layerwise LR ────────────────────────────────

    def parameter_groups(
        self,
        encoder_lr: float = 5e-5,
        decoder_lr: float = 5e-4,
        lr_decay: float = 0.8,
        weight_decay: float = 1e-4,
    ) -> List[dict]:
        """
        Build AdamW parameter groups with layerwise encoder LR decay.

        Encoder stages 3→0 get ``encoder_lr * lr_decay^(3-stage)``.
        Encoder stem gets ``encoder_lr * lr_decay^4``.
        Decoder, refiner, and head all get ``decoder_lr``.
        """
        # Identify encoder stage parameters
        # timm ConvNeXt uses stages_0, stages_1, ... and stem_0, stem_1
        enc_stem_params = []
        enc_stage_params = {i: [] for i in range(4)}

        for name, param in self.encoder.named_parameters():
            if not param.requires_grad:
                continue
            assigned = False
            for stage_idx in range(4):
                if name.startswith(f"stages_{stage_idx}"):
                    enc_stage_params[stage_idx].append(param)
                    assigned = True
                    break
            if not assigned:
                enc_stem_params.append(param)  # stem / other

        groups = []

        # Encoder stem: deepest decay
        if enc_stem_params:
            groups.append({
                "params": enc_stem_params,
                "lr": encoder_lr * (lr_decay ** 4),
                "weight_decay": weight_decay,
                "name": "encoder_stem",
            })

        # Encoder stages 0→3 (stage 3 = highest LR = encoder_lr)
        for stage_idx in range(4):
            params = enc_stage_params[stage_idx]
            if params:
                stage_lr = encoder_lr * (lr_decay ** (3 - stage_idx))
                groups.append({
                    "params": params,
                    "lr": stage_lr,
                    "weight_decay": weight_decay,
                    "name": f"encoder_stage{stage_idx}",
                })

        # Decoder + refiner + head: all decoder_lr
        decoder_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and not name.startswith("encoder"):
                decoder_params.append(param)

        if decoder_params:
            groups.append({
                "params": decoder_params,
                "lr": decoder_lr,
                "weight_decay": weight_decay,
                "name": "decoder",
            })

        return groups


