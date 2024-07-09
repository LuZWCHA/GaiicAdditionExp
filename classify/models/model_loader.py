import torch
from torchvision.models import vit_l_32, ViT_L_32_Weights, ViT_B_32_Weights
from torchvision.models.vision_transformer import VisionTransformer, _vision_transformer

from models.dual_stream_vit import DualStreamViT

def get_default_model():
    model = vit_l_32(ViT_L_32_Weights.IMAGENET1K_V1)
    return model

def get_dual_stream_vit(num_class=2):
    return DualStreamViT(num_classes=num_class)

def get_vit_large(num_class=2, progress=True, weights = ViT_L_32_Weights.IMAGENET1K_V1, strict=False, **kwargs):
    model = _vision_transformer(
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=None,
        progress=progress,
        num_classes=num_class,
        **kwargs,
    )
    if weights is not None:
        checkpoint = weights.get_state_dict(progress)
        if "heads.head.weight" in checkpoint and "heads.head.weight" in model.state_dict():
            if checkpoint["heads.head.weight"].shape != model.state_dict()["heads.head.weight"].shape:
                del checkpoint["heads.head.weight"]
                del checkpoint["heads.head.bias"]
                # init heads
                import torch.nn as nn
                nn.init.zeros_(model.state_dict()["heads.head.bias"] )
                nn.init.normal_(model.state_dict()["heads.head.weight"] )
                
        model.load_state_dict(checkpoint, strict=strict)
    return model
    
def get_vit_base(num_class=2, progress=True, weights = ViT_B_32_Weights.IMAGENET1K_V1, strict=False, **kwargs):
    model = _vision_transformer(
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
        checkpoint = weights.get_state_dict(progress)
        if "heads.head.weight" in checkpoint and "heads.head.weight" in model.state_dict():
            if checkpoint["heads.head.weight"].shape != model.state_dict()["heads.head.weight"].shape:
                del checkpoint["heads.head.weight"]
                del checkpoint["heads.head.bias"]
                # init heads
                import torch.nn as nn
                nn.init.zeros_(model.state_dict()["heads.head.bias"] )
                nn.init.normal_(model.state_dict()["heads.head.weight"] )
                
        model.load_state_dict(checkpoint, strict=strict)
    return model

def get_resnet18(num_class=2, progress=False, weights=None, strict=False):
    from torchvision.models.resnet import _resnet, BasicBlock, ResNet
    from torchvision.models import resnet18, ResNet18_Weights
    if weights is None:
        weights = ResNet18_Weights.IMAGENET1K_V1

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_class)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress), strict=strict)
        print(model.state_dict().keys())
        import torch.nn as nn
        nn.init.zeros_(model.state_dict()["fc.bias"] )
        nn.init.normal_(model.state_dict()["fc.weight"])

    return model

# def get_convnext_v2_p(num_class=2, weights="/nasdata/private/zwlu/segmentation/segment-anything/classify/models/convnext/convnextv2_pico_1k_224_ema.pt", strict=False, **kwargs):
#     from classify.models.convnext.convnextv2 import convnext_pico, convnextv2_tiny
#     if weights is not None:
#         weights = torch.load(weights)
    
#     model = convnext_pico(num_classes=num_class)
    
#     if weights is not None:
#         model.load_state_dict(weights, strict=strict)
        
#     return model

# def get_convnext_v2_p(num_class=2, weights="/nasdata/private/zwlu/segmentation/segment-anything/classify/models/convnext/convnextv2_pico_1k_224_ema.pt", strict=False, **kwargs):
#     from .convnext.convnextv2 import convnextv2_pico, convnextv2_tiny, convnextv2_base
#     if weights is not None:
#         weights = torch.load(weights)
    
#     model = convnextv2_pico(num_classes=num_class)
    
#     if weights is not None:
#         del weights["model"]["head.weight"]
#         del weights["model"]["head.bias"]
#         res = model.load_state_dict(weights["model"], strict=strict)
        
#     return model

def get_efficientnet_b2(num_class=2, in_channels=3):
    from .efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_class, in_channels=in_channels)
    
    return model

def get_efficientnet_b0(num_class=2, in_channels=3):
    from .efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_class, in_channels=in_channels)
    
    return model

def get_efficientnet_b4(num_class=2, in_channels=3):
    from .efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_class, in_channels=in_channels)
    
    return model

def get_efficientnet_b8(num_class=2, in_channels=3):
    from .efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b8', num_classes=num_class, in_channels=in_channels)
    
    return model

# def get_swiftformer_s(num_class=2):
#     from .swiftformer.swiftformer import SwiftFormer_S
#     model = SwiftFormer_S(pretrained="classify/models/swiftformer/SwiftFormer_S.pth", num_classes=num_class)
    
#     return model