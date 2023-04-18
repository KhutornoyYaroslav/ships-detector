import torch
import torch.nn as nn
import torch.optim as optim
from core.model import build_model
from core.data import make_data_loader


class Train:

    def __init__(self, num_classes: int,
                 num_epochs: int,
                 dataloader: dict,
                 rpn_loss_fn,
                 classification_loss,
                 regression_loss):

        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.dataloader = dataloader
        self.rpn_loss_fn = rpn_loss_fn
        self.classification_loss = classification_loss
        self.regression_loss = regression_loss

    def train_model(self):

        # Change number of classes maybe
        model = build_model(num_classes=self.num_classes, pretrained=True)

        model = nn.Sequential(*list(model.children())[:-2])

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        dataloader = make_data_loader(root_dir=self.dataloader['root_dir'],
                                      type=self.dataloader['type'],
                                      model_input_size=self.dataloader['model_input_size'],
                                      is_train=self.dataloader['is_train'],
                                      batch_size=self.dataloader['batch_size'])

        # Обучение модели
        for epoch in range(self.num_epochs):
            for images, targets in dataloader:
                # Clear gradients
                optimizer.zero_grad()

                # Counting loss func
                rpn_loss = rpn_loss_fn(output_rpn, targets)
                classification_loss = classification_loss_fn(output_classification, targets)
                regression_loss = regression_loss_fn(output_regression, targets)

                # Sum loss func
                total_loss = rpn_loss + classification_loss + regression_loss

                # Gradient backward spreading
                total_loss.backward()

                # Update weights
                optimizer.step()

        # Saving model
        torch.save(model.state_dict(), 'faster_rcnn.pth')