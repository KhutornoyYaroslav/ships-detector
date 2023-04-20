import torch
import cv2 as cv
import numpy as np
from glob import glob
from core.model import build_model
from core.data import make_data_loader
from core.utils.checkpoint import CheckPointer
from core.data.transforms import ToCV2Image, Resize, ConvertFromInts, Clip, Normalize, ToTensor
from core.data import ShipsDataset


def init_model(model_dir):
    # Enable cudnn auto-tuner to find the best algorithm to use for your hardware
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Create device
    device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Set device to '{device_}'")
    device = torch.device(device_)

    # Create model
    model = build_model('ShipsDetector', 7+1)
    model.to(device)

    # Create checkpointer
    arguments = {"epoch": 0}
    checkpointer = CheckPointer(model, None, None, model_dir, True)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    # Set model to evaluation mode
    model.eval()

    return model, device


def prepare_frame(frame, input_size, device):
    # Apply transforms
    frame, _, _ = Resize(input_size)(frame)
    frame, _, _ = ConvertFromInts()(frame)
    frame, _, _ = Clip()(frame)
    frame, _, _ = Normalize()(frame)
    frame, _, _ = ToTensor()(frame)
    frame = frame.unsqueeze(0)

    # Prepare image
    imgs = list(frame.to(device))
    cv_img = 255 * ToCV2Image()(imgs[0])
    cv_img = cv_img.astype(np.uint8).copy()

    return imgs, cv_img


def main():
    # Parameters
    TRAINED_MODEL_DIR = "./output/test_3/"
    INPUT_SIZE = (768, 768)

    # Init model
    model, device = init_model(TRAINED_MODEL_DIR)

    # Connect to camera
    vidcap = cv.VideoCapture(1)

    # Process frames
    while(vidcap.isOpened()):
        ret, frame = vidcap.read()

        if ret:
            imgs, cv_img = prepare_frame(frame, INPUT_SIZE, device)

            with torch.no_grad():
                preds = model.infer_forward(imgs)
                preds = preds[0] # Remove batch dimension

                print(preds)

                boxes = preds['boxes'].cpu().numpy()
                labels = preds['labels'].cpu().numpy()
                scores = preds['scores'].cpu().numpy()

                for i, _ in enumerate(boxes):
                    if scores[i] < 0.85:
                        continue

                    box = boxes[i]
                    cv.rectangle(cv_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
                    label_str = ShipsDataset.CLASSES_INT2STR[labels[i]] + ", {0:.2f}".format(scores[i])
                    cv.putText(cv_img, label_str, (int(box[0]), int(box[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            cv.imshow('Frame', cv_img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                return 1
        else:
            print("Error : Failed to capture frame")
            break

    return 0


if __name__ == '__main__':
    main()