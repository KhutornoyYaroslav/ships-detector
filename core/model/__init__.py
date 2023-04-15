from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
import torchvision
import torch


def build_model():
    # TODO: implement
    model = maskrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                                     progress=True,
                                     num_classes=2,
                                     trainable_backbone_layers=5)
    model.eval()
    return model
