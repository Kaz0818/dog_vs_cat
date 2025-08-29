import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_image_transforms(
    image_size: int,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
):
    training_transform = A.Compose([
        
        A.RandomResizedCrop((image_size, image_size), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    validation_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    return training_transform, validation_transform



