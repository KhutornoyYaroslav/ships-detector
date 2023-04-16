import json
import ast
from torch.utils.data import Dataset
from .transforms import TransformCompose, Resize, ConvertFromInts, Clip, Normalize, ToTensor


def parse_annotation(path: str):
    return ast.literal_eval(json.dumps(path))


class ShipsDataset(Dataset):
    def __init__(self, root_dir: str, image_size, is_train: bool = True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self, listing):
        return len(listing)

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

        transform += [Normalize(), ToTensor()]
        transform = TransformCompose(transform)
        return transform

    def __getitem__(self, idx):
        for i, j in parse_annotation(idx).items():
            for key, value in j.items():
                yield key
                yield value
