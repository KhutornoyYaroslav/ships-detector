import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights


def build_model(num_classes: int, pretrained: bool = True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                                                 progress=True,
                                                                 pretrained=pretrained,
                                                                 trainable_bacbone_layers=5,
                                                                 num_classes=num_classes)
    return model
