from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


class ShipsDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        # self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights)
        self.model = fasterrcnn_resnet50_fpn_v2(num_classes=num_classes)

    def forward(self, images):
        return self.model(images)
