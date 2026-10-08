"""SVTRv2 CTC-only adaptation with the feature-rearrangement (RCTC) head.

Architecture references: OpenOCR SVTRv2LNConvTwo33 and RCTCDecoder.
See NOTICE.md for attribution and differences from the published recipe.
The forward method ALWAYS returns logits, including in evaluation mode.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    image_height: int = 64
    image_width: int = 512
    num_classes: int = 96  # 95 printable ASCII characters plus CTC blank.
    dims: tuple[int, ...] = (64, 128, 192)
    depths: tuple[int, ...] = (3, 4, 3)
    heads: tuple[int, ...] = (2, 4, 6)
    stage2_conv_blocks: int = 2
    dropout: float = 0.1
    drop_path: float = 0.1
    format_version: int = 1

    def __post_init__(self):
        self.dims, self.depths, self.heads = map(tuple, (self.dims, self.depths, self.heads))
        if self.format_version != 1:
            raise ValueError("Unsupported model configuration version")
        if not (len(self.dims) == len(self.depths) == len(self.heads) == 3):
            raise ValueError("SVTRv2 requires three stages")
        if self.image_height < 16 or self.image_height % 8 or self.image_width < 16 or self.image_width % 4:
            raise ValueError("Height must be >=16 and divisible by 8; width >=16 and divisible by 4")
        if any(d < 1 for d in self.depths) or any(h < 1 or d % h for d, h in zip(self.dims, self.heads)):
            raise ValueError("Positive stage dimensions must be divisible by their head counts")
        if self.dims[0] < 2 or not 0 <= self.stage2_conv_blocks < self.depths[1]:
            raise ValueError("Stage two must contain at least one global-attention block")
        if not 0 <= self.dropout < 1 or not 0 <= self.drop_path < 1 or self.num_classes < 2:
            raise ValueError("Invalid dropout, drop-path, or class count")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def preset(cls, name: str, **kwargs) -> "ModelConfig":
        if name == "compact":
            return cls(**kwargs)
        if name == "reference":
            return cls(dims=(128, 256, 384), depths=(6, 6, 6), heads=(4, 8, 12), **kwargs)
        raise ValueError(f"Unknown model preset: {name}")


class DropPath(nn.Module):
    """Randomly drop a residual branch per sample during training."""
    def __init__(self, probability: float):
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device) < keep).to(x.dtype)
        return x * mask / keep


def mlp(dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(dim * 4, dim), nn.Dropout(dropout))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, qkv_bias: bool = True):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.projection = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # PyTorch selects its supported attention implementation for each device.
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(batch, length, dim)
        return self.projection(x)


class ConvMixBlock(nn.Module):
    """Two grouped 3x3 convolutions, then per-position channel normalization."""
    def __init__(self, dim: int, heads: int, dropout: float, drop_path: float):
        super().__init__()
        self.mixer = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=heads),
                                   nn.Conv2d(dim, dim, 3, padding=1, groups=heads))
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = mlp(dim, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.mixer(x))
        x = self.norm1(x.permute(0, 2, 3, 1))  # [B, H, W, C]
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x.permute(0, 3, 1, 2).contiguous()


class GlobalMixBlock(nn.Module):
    """SVTRv2 post-normalized attention across the full two-dimensional map."""
    def __init__(self, dim: int, heads: int, dropout: float, drop_path: float):
        super().__init__()
        self.attention = SelfAttention(dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = mlp(dim, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x + self.drop_path(self.attention(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x.transpose(1, 2).reshape(batch, channels, height, width)


class Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, height_stride: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, stride=(height_stride, 1), padding=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x).permute(0, 2, 3, 1)
        return self.norm(x).permute(0, 3, 1, 2).contiguous()


class FeatureRearrangementHead(nn.Module):
    """Row-wise attention, learned height pooling, then the CTC classifier."""
    def __init__(self, dim: int, heads: int, num_classes: int):
        super().__init__()
        self.row_norm1 = nn.LayerNorm(dim)
        self.row_norm2 = nn.LayerNorm(dim)
        self.row_attention = SelfAttention(dim, heads, dropout=0, qkv_bias=False)
        self.row_mlp = mlp(dim, dropout=0)
        self.char_query = nn.Parameter(torch.zeros(1, 1, dim))
        self.key_value = nn.Linear(dim, 2 * dim)
        self.classifier = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.char_query, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        # Mix features horizontally before learning which height positions matter.
        rows = x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        rows = rows + self.row_attention(self.row_norm1(rows))
        rows = rows + self.row_mlp(self.row_norm2(rows))
        features = rows.reshape(batch, height, width, channels)
        keys, values = self.key_value(features).chunk(2, dim=-1)
        scores = (keys.float() * self.char_query.float()).sum(dim=-1)
        weights = scores.softmax(dim=1).to(values.dtype)  # Normalize along height.
        sequence = (weights.unsqueeze(-1) * values).sum(dim=1)  # [B, W/4, C]
        return self.classifier(sequence)  # Logits, never probabilities.


class SVTRv2CTC(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d0 = config.dims[0]
        self.stem = nn.Sequential(
            nn.Conv2d(1, d0 // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d0 // 2), nn.GELU(),
            nn.Conv2d(d0 // 2, d0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d0), nn.GELU(),
        )
        blocks = []
        total_blocks = sum(config.depths)
        block_number = 0
        for stage in range(3):
            for index in range(config.depths[stage]):
                local = stage == 0 or (stage == 1 and index < config.stage2_conv_blocks)
                block_class = ConvMixBlock if local else GlobalMixBlock
                probability = config.drop_path * block_number / max(total_blocks - 1, 1)
                blocks.append(block_class(config.dims[stage], config.heads[stage],
                                          config.dropout, probability))
                block_number += 1
            if stage < 2:
                blocks.append(Downsample(config.dims[stage], config.dims[stage + 1],
                                         height_stride=1 if stage == 0 else 2))
        self.encoder = nn.Sequential(*blocks)
        self.head = FeatureRearrangementHead(config.dims[-1], config.heads[-1], config.num_classes)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def time_steps(self) -> int:
        return self.config.image_width // 4

    def forward(self, images: Tensor) -> Tensor:
        expected = (1, self.config.image_height, self.config.image_width)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"Expected [batch, {expected}], got {tuple(images.shape)}")
        return self.head(self.encoder(self.stem(images)))
