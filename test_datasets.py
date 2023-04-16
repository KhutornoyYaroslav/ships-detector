from core.data import make_data_loader
import cv2 as cv
import numpy as np
from core.data.transforms import ToCV2Image
from core.data.shipsdataset import ShipsDataset


def test_dataset():
    ROOT_DIR = "./data/ships_dataset/"
    TYPE = "ShipsDataset"
    INPUT_SIZE = (512, 512)
    IS_TRAIN = False
    BATCH_SIZE = 1

    data_loader = make_data_loader(ROOT_DIR, TYPE, INPUT_SIZE, IS_TRAIN, BATCH_SIZE)
    toCVimg = ToCV2Image()

    for item in data_loader:
        img, labels, bboxes = item

        bboxes = bboxes[0].cpu().numpy()
        labels = labels[0].cpu().numpy()

        # Show image
        cv_img = 255 * toCVimg(img[0])
        cv_img = cv_img.astype(np.uint8).copy()

        # Draw bboxes
        for bbox, label in zip(bboxes, labels):
            cv.rectangle(cv_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
            label_str = ShipsDataset.CLASSES_INT2STR[str(label)]
            cv.putText(cv_img, label_str, (int(bbox[0]), int(bbox[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        cv.imshow('image', cv_img)
        cv.waitKey()


if __name__ == "__main__":
    test_dataset()
