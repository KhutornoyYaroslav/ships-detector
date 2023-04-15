import torch
import cv2 as cv
import numpy as np


# TODO: modify it
class TransformCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, labels=None, masks=None, bboxes=None):
        for t in self.transforms:
            images, labels, masks, bboxes = t(images, labels, masks, bboxes)

        return images, labels, masks, bboxes


class Resize(object):
    def __init__(self, size=(512, 512), interpolation=cv.INTER_NEAREST):
        # TODO: implement
        # self.size = size
        # self.interpolation = interpolation
        return None

    def __call__(self, image, label=None, mask=None, bboxes=None):
        # TODO: implement
        # image = cv.resize(image, self.size, interpolation=cv.INTER_AREA)
        # if label is not None:
        #     label = cv.resize(label, self.size, interpolation=self.interpolation)
        # if mask is not None:
        #     mask = cv.resize(mask, self.size, interpolation=cv.INTER_NEAREST)
        # if bboxes is not None:
        #     width_k = self.size[0] / label.shape[1]
        #     height_k = self.size[1] / label.shape[0]
        #     for i in range(len(bboxes)):
        #         bboxes[i][0] = int(width_k * bboxes[i][0])
        #         bboxes[i][1] = int(height_k * bboxes[i][1])
        #         bboxes[i][2] = int(width_k * bboxes[i][2])
        #         bboxes[i][3] = int(height_k * bboxes[i][3])
        # return image, label, mask, bboxes
        return None


class ConvertFromInts:
    def __call__(self, image, label=None, mask=None, bboxes=None):
        # TODO: implement
        # image = image.astype(np.float32)
        # if label is not None:
        #     label = label.astype(np.float32)
        # if mask is not None:
        #     mask = mask.astype(np.float32)
        # if bboxes is not None:
        #     for i in range(len(bboxes)):
        #         bboxes[i] = [float(x) for x in bboxes[i]]
        # return image, label, mask, bboxes
        return None


class Normalize(object):
    def __init__(self, norm_label: bool = True, norm_mask: bool = True):
        # TODO: implement
        # self.norm_label = norm_label
        # self.norm_mask = norm_mask
        return None

    def __call__(self, image, label=None, mask=None, bboxes=None):
        # TODO: implement
        # image = image.astype(np.float32) / 255.0
        # if (label is not None) and self.norm_label:
        #     label = label.astype(np.float32) / 255.0
        # if (mask is not None) and self.norm_mask:
        #     mask = mask.astype(np.float32) / 255.0
        # return image, label, mask, bboxes
        return None


class Clip(object):
    def __init__(self, mmin: float = 0.0, mmax: float = 255.0):
        # TODO: implement
        # self.min = mmin
        # self.max = mmax
        # assert self.max >= self.min, "min val must be >= max val"
        return None

    def __call__(self, image, label=None, mask=None, bboxes=None):
        # TODO: implement
        # image = np.clip(image, self.min, self.max)
        # return image, label, mask, bboxes
        return None


class ToTensor:
    def __call__(self, cvimage, label=None, mask=None, bboxes=None):
        # TODO: implement
        # img = torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)
        # if label is not None:
        #     label = torch.from_numpy(label.astype(np.float32)).permute(2, 0, 1)
        # if mask is not None:
        #     mask = torch.from_numpy(np.expand_dims(mask.astype(np.float32), 0))
        #     # mask = torch.from_numpy(mask.astype(np.float32))
        # if bboxes is not None:
        #     bboxes = torch.FloatTensor(bboxes)
        # return img, label, mask, bboxes
        return None


class ToCV2Image(object):
    def __call__(self, tensor):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0))
