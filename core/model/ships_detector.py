from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights

class ShipsDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.model = fasterrcnn_resnet50_fpn_v2(num_classes=num_classes, trainable_backbone_layers=5)
        self.model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=num_classes, trainable_backbone_layers=6)

    def train_forward(self, images, targets):
        return self.model(images, targets)

    def infer_forward(self, images):
        return self.model(images)
