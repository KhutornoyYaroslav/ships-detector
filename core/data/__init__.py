import torch
from .shipsdataset import ShipsDataset
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, BatchSampler, DataLoader, ConcatDataset


_DATASET_TYPES = {
    "ShipsDataset": ShipsDataset,
}


def build_dataset(type, root_dir, image_size, is_train):
    dataset = _DATASET_TYPES[type]
    return dataset(root_dir, image_size, is_train)


def create_loader(dataset: Dataset,
                  shuffle: bool,
                  batch_size: int,
                  num_workers: int = 1,
                  pin_memory: bool = True):
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=True)
    data_loader = DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler, pin_memory=pin_memory)
    return data_loader


def make_data_loader(root_dirs,
                     type,
                     model_input_size,
                     is_train,
                     batch_size) -> DataLoader:

    # Create datasets
    datasets = []

    for root_dir in root_dirs:
        dataset = build_dataset(type, root_dir, model_input_size, is_train=is_train)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    data_loader = create_loader(dataset, is_train, batch_size, 8, True)
    return data_loader
