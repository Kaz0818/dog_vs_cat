import timm
import torch.nn as nn

def build_model(model_name: str, num_classes: int, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)