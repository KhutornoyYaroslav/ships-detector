import torch
import cv2 as cv
import numpy as np
from core.model import build_model
from core.utils.checkpoint import CheckPointer
from core.data.transforms import ToCV2Image, Resize, ConvertFromInts, Clip, Normalize, ToTensor
from core.data import ShipsDataset
from parameters import *


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
    model = build_model('ShipsDetector', NUM_CLASSES)
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


def resize_box(box, in_size, out_size):
    width_k = out_size[1] / in_size[1]
    height_k = out_size[0] / in_size[0]

    box[0] = int(width_k * box[0])
    box[1] = int(height_k * box[1])
    box[2] = int(width_k * box[2])
    box[3] = int(height_k * box[3])

    return box


def main():
    # Init model
    model, device = init_model(OUTPUT_DIR)

    # Connect to camera
    # vidcap = cv.VideoCapture(1)
    # print(vidcap.set(cv.CAP_PROP_FRAME_WIDTH, 1280))

    # Read video file
    vidcap = cv.VideoCapture("./data/videos/IMG_4111.MOV")

    # Process frames
    while(vidcap.isOpened()):
        ret, frame = vidcap.read()
        frame_orig_size = frame.shape[:2]

        if ret:
            imgs, cv_img = prepare_frame(frame, INPUT_SIZE, device)

            with torch.no_grad():
                preds = model.infer_forward(imgs)
                preds = preds[0] # Remove batch dimension

                boxes = preds['boxes'].cpu().numpy()
                labels = preds['labels'].cpu().numpy()
                scores = preds['scores'].cpu().numpy()

                for i, _ in enumerate(boxes):
                    if scores[i] < 0.85:
                        continue

                    box = resize_box(boxes[i], cv_img.shape[:2], frame_orig_size)
                    cv.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (30, 220, 20), 2)
                    label_str = ShipsDataset.CLASSES_INT2STR[labels[i]] + ", {0:.2f}".format(scores[i])
                    cv.putText(frame, label_str, (int(box[0]), int(box[1]) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            cv.imshow('Frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                return 1
        else:
            print("Error : Failed to capture frame")
            break

    return 0


if __name__ == '__main__':
    main()