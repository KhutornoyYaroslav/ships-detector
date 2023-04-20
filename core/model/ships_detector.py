from torch import nn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

class ShipsDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=num_classes, trainable_backbone_layers=6)

    def train_forward(self, images, targets):
        return self.model(images, targets)

    def infer_forward(self, images):
        return self.model(images)
