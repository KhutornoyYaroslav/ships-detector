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

    def __init__(self, root_dir, image_size, is_train):
        self.root_dir = root_dir
        self.imgs = sorted(glob(root_dir + "/img/*"))
        self.annos = sorted(glob(root_dir + "/ann/*"))
        assert len(self.imgs) == len(self.annos)

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

        # Apply transforms
        img, labels, bboxes = self.transforms(img, labels, bboxes)

        return img, labels, bboxes

        # anno_path = self.annos[idx]
        # anno = json.dumps(anno_path)

        # class_id = self.ship_classes[anno[]]

        # meta_item = parse_annotation(json.dumps(anno_path))["data"][idx]
        # img_path = os.path.join(self.imgs, meta_item[0])
        # image = cv.imread(img_path)

        # # Read bounding rects (from [x, y, w, h] to [x1, y1, x2, y2])
        # rects = np.array([[r[0], r[1], r[0] + r[2], r[1] + r[3]] for r in meta_item["rects"]], dtype=np.int)

        # # Pad rects array to max size
        # rects_real_num = rects.shape[0]
        # assert rects_real_num in range(0, self.max_rects + 1)
        # if rects_real_num == self.max_rects:
        #     pass
        # elif rects_real_num == 0:
        #     rects = np.zeros(shape=(self.max_rects, 4), dtype=np.int)
        # else:
        #     dummy_rects = np.zeros(shape=(self.max_rects - rects_real_num, 4), dtype=np.int)
        #     rects = np.concatenate([rects, dummy_rects])
        # assert rects.shape == (self.max_rects, 4)

        # # Prepare data
        # if self.transforms:
        #     image, rects = self.transforms(image, rects)

        # return image, rects, rects_real_num