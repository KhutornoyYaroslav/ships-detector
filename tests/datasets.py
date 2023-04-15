from core.data import make_data_loader
import cv2 as cv


def test_dataset():
    ROOT_DIR = "./data/ships_dataset/"
    TYPE = "ShipsDataset"
    INPUT_SIZE = (512, 512)
    IS_TRAIN = False
    BATCH_SIZE = 1

    data_loader = make_data_loader(ROOT_DIR, TYPE, INPUT_SIZE, IS_TRAIN, BATCH_SIZE)

    for i in data_loader:
        cv.namedWindow('image', 500)
        cv.imshow('image', i)
        cv.waitKey()
