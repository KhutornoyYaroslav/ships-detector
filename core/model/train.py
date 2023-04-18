import torch
import torch.nn as nn
import torch.optim as optim
from core.model import build_model
from core.data import make_data_loader
from core.solver import make_optimizer
from core.utils import checkpoint


class Trainer(object):

    def __init__(self, num_classes: int,
                 num_epochs: int,
                 dataloader: dict,
                 rpn_loss_fn,
                 classification_loss,
                 regression_loss):

        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.rpn_loss_fn = rpn_loss_fn
        self.classification_loss = classification_loss
        self.regression_loss = regression_loss

    def train_step(self, model, data_loader, optimizer, checkpointer, device, arguments, lr, i, data):
        for images, targets in data_loader:
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(data_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def train(self, model, data_loader, optimizer, checkpointer, device, arguments, lr):
        for i, data in enumerate(data_loader):
            self.train_step(data_loader=data_loader, lr=lr, model=model, optimizer=optimizer, i=i, data=data)

    def main(self):
        # Parameters
        TRAIN_DATASET_ROOT = './data/output1/Furzikov_01/'
        INPUT_SIZE = (512, 512)
        BATCH_SIZE = 8
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        LEARNING_RATE = 1e-3
        OUTPUT_DIR = './output/test/'

        # Create device
        print(f"Set device to '{DEVICE}'")
        device = torch.device(DEVICE)

        # Create model
        model = build_model('ShipsDetector', self.num_classes)
        model.to(device)

        # Create train data
        data_loader = make_data_loader(TRAIN_DATASET_ROOT, 'ShipsDataset', INPUT_SIZE, True, BATCH_SIZE)

        # Create optimizer
        optimizer = make_optimizer(LEARNING_RATE, model)

        # Create checkpointer
        arguments = {"epoch": 0}
        checkpointer = checkpoint.CheckPointer(model, optimizer, None, OUTPUT_DIR, True)
        extra_checkpoint_data = checkpointer.load()
        arguments.update(extra_checkpoint_data)
        # Saving model
        # torch.save(model.state_dict(), 'faster_rcnn.pth')

        # Train model
        return self.train(data_loader=data_loader, lr=LEARNING_RATE, model=model, optimizer=optimizer)
