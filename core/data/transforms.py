import torch
import cv2 as cv
import numpy as np


class TransformCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, labels=None, bboxes=None):
        for t in self.transforms:
            images, labels, bboxes = t(images, labels, bboxes)

        return images, labels, bboxes


class Resize(object):
    def __init__(self, size=(512, 512), interpolation=cv.INTER_NEAREST):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, labels=None, bboxes=None):
        if bboxes is not None:
            width_k = self.size[0] / image.shape[1]
            height_k = self.size[1] / image.shape[0]
            for i in range(len(bboxes)):
                bboxes[i][0] = int(width_k * bboxes[i][0])
                bboxes[i][1] = int(height_k * bboxes[i][1])
                bboxes[i][2] = int(width_k * bboxes[i][2])
                bboxes[i][3] = int(height_k * bboxes[i][3])
        image = cv.resize(image, self.size, interpolation=cv.INTER_AREA)
        return image, labels, bboxes


class ConvertFromInts:
    def __call__(self, image, labels=None, bboxes=None):
        image = image.astype(np.float32)
        if bboxes is not None:
            for i in range(len(bboxes)):
                bboxes[i] = [float(x) for x in bboxes[i]]
        return image, labels, bboxes


class Normalize:
    def __init__(self, norm_mask: bool = True):
        self.norm_mask = norm_mask

    def __call__(self, image, labels=None, bboxes=None):
        image = image.astype(np.float32) / 255.0
        return image, labels, bboxes


class Clip:
    def __init__(self, mmin: float = 0.0, mmax: float = 255.0):
        self.min = mmin
        self.max = mmax
        assert self.max >= self.min, "min val must be >= max val"

    def __call__(self, image, labels=None, bboxes=None):
        image = np.clip(image, self.min, self.max)
        return image, labels, bboxes


class ToTensor:
    def __call__(self, cvimage, labels=None, bboxes=None):
        img = torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)
        if labels is not None:
            labels = torch.LongTensor(labels)
        if bboxes is not None:
            bboxes = torch.FloatTensor(bboxes)
        return img, labels, bboxes


class ToCV2Image(object):
    def __call__(self, tensor):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0))
