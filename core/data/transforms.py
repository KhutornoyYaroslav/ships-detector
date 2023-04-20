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


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, labels=None, bboxes=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv.cvtColor(image, cv.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, labels, bboxes


class RandomHue(object):
    def __init__(self, delta:float=30.0, probabilty:float=0.5):
        self.delta = np.clip(delta, 0.0, 360.0)
        self.probabilty = np.clip(probabilty, 0.0, 1.0)

    def __call__(self, image, labels=None, bboxes=None):
        if np.random.choice([0, 1], size=1, p=[1-self.probabilty, self.probabilty]):
            cvt = ConvertColor(current="RGB", transform='HSV')
            image, _, _ = cvt(image)
            ru = np.random.uniform(-self.delta, self.delta)
            image[:, :, 0] += ru
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
            cvt = ConvertColor(current="HSV", transform='RGB')
            image, _, _ = cvt(image)
        return image, labels, bboxes


class RandomGamma(object):
    def __init__(self, lower:float=0.5, upper:float=2.0, probabilty:float=0.5):
        self.lower = lower
        self.upper = upper
        self.probabilty = np.clip(probabilty, 0.0, 1.0)
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, labels=None, bboxes=None):
        assert image.dtype == np.float32, "image dtype must be float"
        if np.random.choice([0, 1], size=1, p=[1-self.probabilty, self.probabilty]):
            gamma = np.random.uniform(self.lower, self.upper)
            if np.mean(image) > 100:
                image = pow(image / 255., gamma) * 255.
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
