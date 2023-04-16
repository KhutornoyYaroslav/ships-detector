import os
import json
import ast
import cv2 as cv
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from .transforms import TransformCompose, Resize, ConvertFromInts, Clip, Normalize, ToTensor


def parse_annotation(path: str):
    return ast.literal_eval(json.dumps(path))


class ShipsDataset(Dataset):
    def __init__(self, root_dir: str, class_labels: list, transforms=None):
        self.root_dir = root_dir
        self.imgs = sorted(glob(root_dir + "/*/img/*"))
        self.json = sorted(glob(root_dir + "/*/ann/*"))
        self.labels = sorted(glob(root_dir + "/*/labels/*"))
        assert len(self.imgs) == len(self.labels)
        self.transforms = transforms

        self.ship_classes = {'military': 0,
                             'boat': 1,
                             'tanker': 2,
                             'civilian': 3,
                             'barge': 4}

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
        meta_item = parse_annotation(json.dumps(self.root_dir + '/ann/'))["data"][idx]
        img_path = os.path.join(self.imgs, meta_item[0])
        image = cv.imread(img_path)

        # Read bounding rects (from [x, y, w, h] to [x1, y1, x2, y2])
        rects = np.array([[r[0], r[1], r[0] + r[2], r[1] + r[3]] for r in meta_item["rects"]], dtype=np.int)

        # Pad rects array to max size
        rects_real_num = rects.shape[0]
        assert rects_real_num in range(0, self.max_rects + 1)
        if rects_real_num == self.max_rects:
            pass
        elif rects_real_num == 0:
            rects = np.zeros(shape=(self.max_rects, 4), dtype=np.int)
        else:
            dummy_rects = np.zeros(shape=(self.max_rects - rects_real_num, 4), dtype=np.int)
            rects = np.concatenate([rects, dummy_rects])
        assert rects.shape == (self.max_rects, 4)

        # Prepare data
        if self.transforms:
            image, rects = self.transforms(image, rects)

        return image, rects, rects_real_num