import os
import json, codecs
import ast
import cv2 as cv
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from .transforms import TransformCompose, Resize, ConvertFromInts, Clip, Normalize, ToTensor


def parse_annotation(path: str):
    return ast.literal_eval(json.dumps(path))


class ShipsDataset(Dataset):
    CLASSES_STR2INT = {'military': 0,
                       'boat': 1,
                       'tanker': 2,
                       'civilian': 3,
                       'barge': 4}

    CLASSES_INT2STR = {0: 'military',
                       1: 'boat',
                       2: 'tanker',
                       3: 'civilian',
                       4: 'barge'}

    MAX_PADDING = 32

    def __init__(self, root_dir, image_size, is_train):
        self.root_dir = root_dir
        self.imgs = sorted(glob(root_dir + "/img/*"))
        self.annos = sorted(glob(root_dir + "/ann/*"))
        assert len(self.imgs) == len(self.annos)

        print('Read dataset {0}. Size: {1}.'.format(root_dir, len(self.imgs)))

        for i in range(len(self.annos)):
            json_data = codecs.open(self.annos[i], 'r').read()
            self.annos[i] = json.loads(json_data)

        self.transforms = self.build_transforms(image_size, is_train)

    def __len__(self):
        return len(self.imgs)

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
        anno = self.annos[idx]

        # Read image
        img = cv.imread(self.imgs[idx])
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Read labels
        labels = []
        for obj in anno['objects']:
            labels.append(self.CLASSES_STR2INT[obj['classTitle']])

        # Read bboxes
        bboxes = []
        for obj in anno['objects']:
            x1y1, x2y2 = obj['points']['exterior']
            bboxes.append([*x1y1, *x2y2])

        # Pad bboxes, labels to fix size
        bboxes = bboxes + [[0, 0, 0, 0]] * (self.MAX_PADDING - len(bboxes))
        labels = labels + [-1] * (self.MAX_PADDING - len(labels))

        # Apply transforms
        img, labels, bboxes = self.transforms(img, labels, bboxes)

        return img, labels, bboxes
