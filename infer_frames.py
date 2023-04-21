import torch
import cv2 as cv
import numpy as np
from tqdm import tqdm
from glob import glob
from parameters import *
from core.data import ShipsDataset
from core.model import build_model
from core.utils.checkpoint import CheckPointer
from core.data.transforms import ToCV2Image, Resize, ConvertFromInts, Clip, Normalize, ToTensor


def resize_box(box, in_size, out_size):
    width_k = out_size[1] / in_size[1]
    height_k = out_size[0] / in_size[0]

    box[0] = int(width_k * box[0])
    box[1] = int(height_k * box[1])
    box[2] = int(width_k * box[2])
    box[3] = int(height_k * box[3])

    return box


def infer(model, frames, input_size, device):
    torch.cuda.empty_cache()
    model.eval()

    for frame_path in tqdm(frames):
        # Read input frame
        frame_cv = cv.imread(frame_path)
        frame_orig_size = frame_cv.shape[:2]

        # Transforms frame
        frame, _, _ = Resize(input_size)(frame_cv)
        frame, _, _ = ConvertFromInts()(frame)
        frame, _, _ = Clip()(frame)
        frame, _, _ = Normalize()(frame)
        frame, _, _ = ToTensor()(frame)
        frame = frame.unsqueeze(0)

        # Prepare images
        imgs = list(frame.to(device))

        cv_img = 255 * ToCV2Image()(imgs[0])
        cv_img = cv_img.astype(np.uint8).copy()

        with torch.no_grad():
            preds = model.infer_forward(imgs)
            preds = preds[0] # Remove batch dimension

            boxes = preds['boxes'].cpu().numpy()
            labels = preds['labels'].cpu().numpy()
            scores = preds['scores'].cpu().numpy()

            for i, _ in enumerate(boxes):
                if scores[i] < PROB_THRESH:
                    continue

                box = resize_box(boxes[i], cv_img.shape[:2], frame_orig_size)
                cv.rectangle(frame_cv, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (30, 220, 20), 2)
                label_str = ShipsDataset.CLASSES_INT2STR[labels[i]] + ", {0:.2f}".format(scores[i])
                cv.putText(frame_cv, label_str, (int(box[0]), int(box[1]) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            cv.imshow('Frame', frame_cv)
            if cv.waitKey(0) & 0xFF == ord('q'):
                return -1


def main():
    # Parameters
    FRAMES = sorted(glob("./data/frames_1/*")) + sorted(glob("./data/frames_2/*")) + sorted(glob("./data/frames_3/*"))

    # Enable cudnn auto-tuner to find the best algorithm to use for your hardware
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Create device
    print(f"Set device to '{DEVICE}'")
    device = torch.device(DEVICE)

    # Create model
    model = build_model('ShipsDetector', NUM_CLASSES)
    model.to(device)

    # Create checkpointer
    arguments = {"epoch": 0}
    checkpointer = CheckPointer(model, None, None, OUTPUT_DIR, True)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    # Infer model
    return infer(model, FRAMES, INPUT_SIZE, device)

if __name__ == '__main__':
    main()