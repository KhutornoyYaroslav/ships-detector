import torch
from core.data import make_data_loader
from core.model import build_model
from core.solver import make_optimizer
from core.utils.checkpoint import CheckPointer


def train_step():
    # TODO: implement
    return None


def train(model, data_loader, optimizer, checkpointer, device, arguments):
    # TODO: implement
    return None


def main():
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

    # Create train data
    data_loader = make_data_loader(TRAIN_DATASET_ROOT, 'ShipsDataset', INPUT_SIZE, True, BATCH_SIZE)

    # Create model
    model = build_model('ShipsDetector', 5+1)
    model.to(device)

    # Create optimizer
    optimizer = make_optimizer(LEARNING_RATE, model)

    # Create checkpointer
    arguments = {"epoch": 0}
    checkpointer = CheckPointer(model, optimizer, None, OUTPUT_DIR, True)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    # Train model
    return train()


if __name__ == '__main__':
    main()