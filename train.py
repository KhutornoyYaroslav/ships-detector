import os
import torch
import numpy as np
from tqdm import tqdm
from core.data import make_data_loader
from core.model import build_model
from core.solver import make_optimizer
from core.utils.checkpoint import CheckPointer


def prepare_data(data_entry, device):
    imgs, labels, bboxes = data_entry

    # Prepare images
    imgs = list(img.to(device) for img in imgs)

    # Prepare targets
    targets = []
    for batch_idx in range(len(imgs)):
        d = {}

        items_length = 0
        for bbox in bboxes[batch_idx]:
            if not all(np.isclose(x, 0) for x in bbox):
                items_length += 1

        if items_length == 0:
            d['boxes'] = torch.empty((0, 4), dtype=torch.float32).to(device)
            d['labels'] = torch.empty((0,), dtype=torch.int64).to(device)
        else:
            bboxes_ = bboxes[batch_idx][0:items_length]
            labels_ = labels[batch_idx][0:items_length]
            d['boxes'] = bboxes_.to(device)
            d['labels'] = labels_.to(device)

        targets.append(d)

    return imgs, targets


def eval(model, data_loader, device):
    torch.cuda.empty_cache()
    # model.eval() # No need

    stats = {
        'sample_count': 0.0,
        'loss_sum': 0.0,
        'loss_classifier': 0.0,
        'loss_box_reg': 0.0,
        'loss_objectness': 0.0,
        'loss_rpn_box_reg': 0.0
    }

    for data_entry in tqdm(data_loader):
        imgs, targets = prepare_data(data_entry, device)

        # Forward images
        with torch.no_grad():
            loss_dict = model.train_forward(imgs, targets)

        # Calculate loss
        loss = sum(loss for loss in loss_dict.values())
        stats['loss_sum'] += loss.item()
        stats['loss_classifier'] += loss_dict["loss_classifier"].item()
        stats['loss_box_reg'] += loss_dict["loss_box_reg"].item()
        stats['loss_objectness'] += loss_dict["loss_objectness"].item()
        stats['loss_rpn_box_reg'] += loss_dict["loss_rpn_box_reg"].item()
        stats['sample_count'] += 1

    # Return results
    stats['loss_sum'] /= stats['sample_count']
    stats['loss_classifier'] /= stats['sample_count']
    stats['loss_box_reg'] /= stats['sample_count']
    stats['loss_objectness'] /= stats['sample_count']
    stats['loss_rpn_box_reg'] /= stats['sample_count']

    result_dict = {
        'loss_sum': stats['loss_sum'],
        'loss_classifier': stats['loss_classifier'],
        'loss_box_reg': stats['loss_box_reg'],
        'loss_objectness': stats['loss_objectness'],
        'loss_rpn_box_reg': stats['loss_rpn_box_reg'],
    }

    torch.cuda.empty_cache()
    return result_dict


def train_step(model, device, data_entry, optimizer, loss_stat):
    # Prepare training sample
    imgs, targets = prepare_data(data_entry, device)

    # Do prediction
    model.train()
    loss_dict = model.train_forward(imgs, targets)

    # Save loss stat
    for key, value in loss_dict.items():
        loss_stat[key] += value.item()

    # Calculate loss
    loss = sum(loss for loss in loss_dict.values())

    # Do optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train(model, data_loader, valid_data_loader, optimizer, checkpointer, device, arguments, max_epochs):
    iters_per_epoch = len(data_loader)
    total_steps = iters_per_epoch * max_epochs
    start_epoch = arguments["epoch"]
    print(f"Iterations per epoch: {iters_per_epoch}. Total steps: {total_steps}. Start epoch: {start_epoch}")

    # Epoch loop
    for epoch in range(start_epoch, max_epochs):
        arguments["epoch"] = epoch + 1

        # Create progress bar
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'lr', 'loss', 'loss_cls', 'loss_box', 'loss_obj', 'loss_rpn'))
        pbar = enumerate(data_loader)
        pbar = tqdm(pbar, total=len(data_loader))

        # Iteration loop
        loss_stat = {
            'loss_classifier': 0,
            'loss_box_reg': 0,
            'loss_objectness': 0,
            'loss_rpn_box_reg': 0
        }

        # Train one epoch
        for iteration, data_entry in pbar:
            global_step = epoch * iters_per_epoch + iteration

            # Do training step
            train_step(model, device, data_entry, optimizer, loss_stat)

            # Update progress bar
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0) # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                                                '%g/%g' % (epoch, max_epochs - 1),
                                                mem,
                                                optimizer.param_groups[0]['lr'],
                                                sum(loss_stat.values()) / (iteration + 1),
                                                loss_stat['loss_classifier'] / (iteration + 1),
                                                loss_stat['loss_box_reg'] / (iteration + 1),
                                                loss_stat['loss_objectness'] / (iteration + 1),
                                                loss_stat['loss_rpn_box_reg'] / (iteration + 1))
            pbar.set_description(s)

        # Save epoch results
        print("Saving results to 'model_{:06d}'.".format(global_step))
        checkpointer.save("model_{:06d}".format(global_step), **arguments)

        # Evaluate model
        print('\nEvaluation ...')
        eval_loss_stat = eval(model, valid_data_loader, device)

        print(('\n' + 'Evaluation results:' + '%10s' * 5) % ('loss', 'loss_cls', 'loss_box', 'loss_obj', 'loss_rpn'))
        print('                   ' + '%10.4g%10.4g%10.4g%10.4g%10.4g' % (eval_loss_stat['loss_sum'],
                                                                          eval_loss_stat['loss_classifier'],
                                                                          eval_loss_stat['loss_box_reg'],
                                                                          eval_loss_stat['loss_objectness'],
                                                                          eval_loss_stat['loss_rpn_box_reg']))

    return model


def main():
    # Parameters
    TRAIN_DATASET_ROOT = './data/output1/Furzikov_01/'
    VALID_DATASET_ROOT = './data/output1/Furzikov_01/'
    INPUT_SIZE = (512, 512)
    BATCH_SIZE = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE = 1e-3
    OUTPUT_DIR = './output/test/'
    MAX_EPOCHS = 100

    # Enable cudnn auto-tuner to find the best algorithm to use for your hardware
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Create device
    print(f"Set device to '{DEVICE}'")
    device = torch.device(DEVICE)

    # Create datasets
    data_loader = make_data_loader(TRAIN_DATASET_ROOT, 'ShipsDataset', INPUT_SIZE, True, BATCH_SIZE)
    valid_data_loader = make_data_loader(VALID_DATASET_ROOT, 'ShipsDataset', INPUT_SIZE, False, BATCH_SIZE)

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
    return train(model, data_loader, valid_data_loader, optimizer, checkpointer, device, arguments, MAX_EPOCHS)


if __name__ == '__main__':
    main()