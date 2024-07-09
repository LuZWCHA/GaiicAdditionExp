from monai.transforms import (
        EnsureChannelFirstd,
        Compose, 
        SpatialPadd, 
        RandZoomd, 
        LoadImaged, 
        RandFlipd, 
        Resized, 
        Rotate90d, 
        RandGaussianSmoothd, 
        RandAdjustContrastd, 
        RandRotated, 
        SpatialPad, 
        RandCoarseDropoutd, 
        RandSimulateLowResolutiond, 
        NormalizeIntensityd,
        EnsureTyped
    )

train_transforms = Compose(
        [
            LoadImaged(keys=["tir", "rgb"]), 
            EnsureChannelFirstd(keys=["tir", "rgb"]),
            RandRotated(keys=["tir", "rgb"], prob=0.5),
            RandFlipd(keys=["tir", "rgb"], prob=0.5),
            RandGaussianSmoothd(keys=["tir"], prob=0.1,),
            RandCoarseDropoutd(keys=["rgb"], spatial_size=[4, 4], holes=1, max_holes=4, max_spatial_size=[32, 32], prob=0.2),
            Resized(keys=["tir", "rgb"], spatial_size=[224, 224]),
            SpatialPadd(keys=["tir", "rgb"], spatial_size=[224, 224]),
            RandZoomd(["tir", "rgb"], min_zoom=0.8, max_zoom=1.25),
            NormalizeIntensityd(keys=["tir", "rgb"], channel_wise=True),
            EnsureTyped(keys=["tir", "rgb", "label"],)
        ]
    )

val_transforms = Compose(
        [
            LoadImaged(keys=["tir", "rgb"]), 
            EnsureChannelFirstd(keys=["tir", "rgb"]),
            
            Resized(keys=["tir", "rgb"], spatial_size=[224, 224]),
            SpatialPadd(keys=["tir", "rgb"], spatial_size=[224, 224]),
            NormalizeIntensityd(keys=["tir", "rgb"], channel_wise=True),
            EnsureTyped(keys=["tir", "rgb", "label"],)
        ]
    )