from torch.utils.data import Dataset
from .transforms import TransformCompose, Resize, ConvertFromInts, Clip, Normalize, ToTensor


class ShipsDataset(Dataset):
    def __init__(self, root_dir: str, image_size, is_train: bool = True):
        # TODO: implement
        return None

    def __len__(self):
        # TODO: implement
        return None

    def build_transforms(self, image_size, is_train: bool = True):
        if is_train:
            transform = [
                Resize(image_size),
                ConvertFromInts(),
                Clip()
            ]
        else:
            transform = [
                Resize(image_size),
                ConvertFromInts(),
                Clip()
            ]

        transform = transform + [Normalize(), ToTensor()]
        transform = TransformCompose(transform)
        return transform

    def __getitem__(self, idx):
        # TODO: implement
        return None
