import cv2 as cv
import json, codecs
from glob import glob
from torch.utils.data import Dataset
from .transforms import TransformCompose, Resize, ConvertFromInts, Clip, Normalize, ToTensor, RandomHue, RandomGamma


class ShipsDataset(Dataset):
    CLASSES_STR2INT = {
        'military':         1,
        'boat':             2,
        'tanker':           3,
        'civilian':         4,
        'barge':            5,
        'small_military':   6,
        'civilian_small':   7
    }
    CLASSES_INT2STR = {
        1: 'military',
        2: 'boat',
        3: 'tanker',
        4: 'civilian',
        5: 'barge',
        6: 'small_military',
        7: 'civilian_small'
    }
    MAX_PADDING = 32

    def __init__(self, root_dir, image_size, is_train):
        self.root_dir = root_dir
        self.imgs = sorted(glob(root_dir + "/img/*"))
        self.annos = sorted(glob(root_dir + "/ann/*"))
        assert len(self.imgs) == len(self.annos)

        print("Read dataset from: '{0}'. Size: {1}.".format(root_dir, len(self.imgs)))

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
                RandomHue(30, 0.5),
                RandomGamma(0.3, 3.0, 0.5),
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

        # Read labels, bboxes
        labels = []
        bboxes = []
        for obj in anno['objects']:
            x1y1, x2y2 = obj['points']['exterior']

            if x2y2[0] <= x1y1[0] or x2y2[1] <= x1y1[1]:
                # print("Skip sample '{0}' due to invalid bbox: {1}".format(self.imgs[idx], (x1y1, x2y2)))
                continue

            if obj['classTitle'] not in self.CLASSES_STR2INT:
                # print("Skip sample '{0}' due to invalid label: {1}".format(self.imgs[idx], obj['classTitle']))
                continue

            bboxes.append([*x1y1, *x2y2])
            labels.append(self.CLASSES_STR2INT[obj['classTitle']])

        # Pad bboxes, labels to fix size
        bboxes = bboxes + [[0, 0, 0, 0]] * (self.MAX_PADDING - len(bboxes))
        labels = labels + [-1] * (self.MAX_PADDING - len(labels))

        # Apply transforms
        img, labels, bboxes = self.transforms(img, labels, bboxes)

        return img, labels, bboxes
