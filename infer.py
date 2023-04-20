import enum
import torch
import cv2 as cv
import numpy as np
from tqdm import tqdm
from glob import glob
from core.model import build_model
from core.data import make_data_loader
from core.utils.checkpoint import CheckPointer
from core.data.transforms import ToCV2Image
from core.data import ShipsDataset


def infer(model, data_loader, device):
    torch.cuda.empty_cache()
    model.eval()

    for data_entry in tqdm(data_loader):
        imgs, _, _ = data_entry

        # Prepare images
        imgs = list(img.to(device) for img in imgs)

        cv_img = 255 * ToCV2Image()(imgs[0])
        cv_img = cv_img.astype(np.uint8).copy()

        with torch.no_grad():
            preds = model.infer_forward(imgs)
            preds = preds[0] # Remove batch dimension

            boxes = preds['boxes'].cpu().numpy()
            labels = preds['labels'].cpu().numpy()
            scores = preds['scores'].cpu().numpy()

            for i, _ in enumerate(boxes):
                if scores[i] < 0.7:
                    continue

                box = boxes[i]
                cv.rectangle(cv_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
                label_str = ShipsDataset.CLASSES_INT2STR[labels[i]]
                cv.putText(cv_img, label_str, (int(box[0]), int(box[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            cv.imshow('Frame', cv_img)
            if cv.waitKey(100) & 0xFF == ord('q'):
                return -1


def main():
    # Parameters
    TEST_DATASET_ROOTS = glob("./data/dataset_test/valid/*")
    INPUT_SIZE = (512, 512)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TRAINED_MODEL_DIR = './output/test/'

    # Enable cudnn auto-tuner to find the best algorithm to use for your hardware
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Create device
    print(f"Set device to '{DEVICE}'")
    device = torch.device(DEVICE)

    # Create datasets
    test_data_loader = make_data_loader(TEST_DATASET_ROOTS, 'ShipsDataset', INPUT_SIZE, False, 1)

    # Create model
    model = build_model('ShipsDetector', 6+1)
    model.to(device)

    # Create checkpointer
    arguments = {"epoch": 0}
    checkpointer = CheckPointer(model, None, None, TRAINED_MODEL_DIR, True)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    # Infer model
    return infer(model, test_data_loader, device)

if __name__ == '__main__':
    main()