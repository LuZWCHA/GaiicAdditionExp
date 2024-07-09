import copy
import torch.nn as nn

import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn

from torchvision.ops.misc import Conv2dNormActivation, MLP
from torchvision.utils import _log_api_usage_once

__all__ = [
    "VisionTransformerWithoutHeader",
    "ViT_B_16_Weights",
    "ViT_B_32_Weights",
    "ViT_L_16_Weights",
    "ViT_L_32_Weights",
    "ViT_H_14_Weights",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
]


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformerWithoutHeader(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        input_channels: int,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = input_channels
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=input_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        return x

from torchvision.models._utils import _ovewrite_named_param
def _vision_transformer(
    input_channels: int,
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights,
    progress: bool,
    **kwargs: Any,
) -> VisionTransformerWithoutHeader:
    if weights is not None:
        checkpoint = weights.get_state_dict()
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    backbone = VisionTransformerWithoutHeader(
        input_channels=input_channels,
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        backbone.load_state_dict(checkpoint)

    return backbone

from torchvision.models import vit_l_32, ViT_L_32_Weights, ViT_B_32_Weights

def get_vit_base(input_channels, num_class=2, progress=True, weights = ViT_B_32_Weights.IMAGENET1K_V1, strict=False, **kwargs):
    model = _vision_transformer(
        input_channels=input_channels,
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=None,
        progress=progress,
        num_classes=num_class,
        **kwargs,
    )
    
    if weights is not None:
        checkpoint = weights.get_state_dict()
        if "heads.head.weight" in checkpoint and "heads.head.weight" in model.state_dict():
            if checkpoint["heads.head.weight"].shape != model.state_dict()["heads.head.weight"].shape:
                # del checkpoint["heads.head.weight"]
                # del checkpoint["heads.head.bias"]
                # init heads
                # import torch.nn as nn
                nn.init.zeros_(model.state_dict()["heads.head.bias"] )
                nn.init.normal_(model.state_dict()["heads.head.weight"] )
            model.load_state_dict(checkpoint, strict=strict)

    return model

def get_vit_large(input_channels, num_class=2, progress=True, weights = ViT_L_32_Weights.IMAGENET1K_V1, strict=False, **kwargs):
    model = _vision_transformer(
        input_channels=input_channels,
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=None,
        progress=progress,
        num_classes=num_class,
        **kwargs,
    )
    
    if weights is not None:
        checkpoint = weights.get_state_dict()
        if "heads.head.weight" in checkpoint and "heads.head.weight" in model.state_dict():
            if checkpoint["heads.head.weight"].shape != model.state_dict()["heads.head.weight"].shape:
                # del checkpoint["heads.head.weight"]
                # del checkpoint["heads.head.bias"]
                # init heads
                # import torch.nn as nn
                nn.init.zeros_(model.state_dict()["heads.head.bias"] )
                nn.init.normal_(model.state_dict()["heads.head.weight"] )
            model.load_state_dict(checkpoint, strict=strict)

    return model

class DualStreamViT(nn.Module):

    def __init__(self, 
                input_channels=3, 
                num_classes=6, 
                patch_size=32,
                num_layers=12,
                num_heads=12,
                hidden_dim=768,
                mlp_dim=3072,
                **kwargs) -> None:
        super().__init__()
        pretrained_weight = kwargs.pop("pretrained_weights", None)
        if pretrained_weight is None:
            weights = ViT_B_32_Weights.IMAGENET1K_V1

        self.vit1 = get_vit_base(
            input_channels=input_channels,
            num_class=num_classes,
            progress=False,
            weights=weights
        )

        self.vit2 = get_vit_base(
            input_channels=input_channels,
            num_class=num_classes,
            progress=False,
            weights=weights
        )

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()

        heads_layers["head"] = nn.Linear(hidden_dim, num_classes)

        self.heads = nn.Sequential(heads_layers)
        # self.vit2.conv_proj
        if isinstance(self.vit2.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.vit2.conv_proj.in_channels * self.vit2.conv_proj.kernel_size[0] * self.vit2.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.vit2.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.vit2.conv_proj.bias is not None:
                nn.init.zeros_(self.vit2.conv_proj.bias)
        elif self.vit2.conv_proj.conv_last is not None and isinstance(self.vit2.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.vit2.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.vit2.conv_proj.conv_last.out_channels)
            )
            if self.vit2.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.vit2.conv_proj.conv_last.bias)

        if isinstance(self.vit1.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.vit1.conv_proj.in_channels * self.vit1.conv_proj.kernel_size[0] * self.vit1.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.vit1.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.vit1.conv_proj.bias is not None:
                nn.init.zeros_(self.vit1.conv_proj.bias)
        elif self.vit1.conv_proj.conv_last is not None and isinstance(self.vit1.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.vit1.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.vit1.conv_proj.conv_last.out_channels)
            )
            if self.vit1.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.vit1.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def forward(self, x: torch.Tensor, return_features=False):
        rgb = x[:, :3]
        tir = x[:, 3:]
        # Reshape and permute the input tensor
        x = self.vit1(rgb)
        y = self.vit2(tir)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        y = y[:, 0]
        z = self.heads(x + y)
        if return_features:
            return z, x + y
        return z