import torch
from glob import glob

# Common
NUM_CLASSES = 8
INPUT_SIZE = (768, 768)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = './output/test_3/'

# Training
TRAIN_DATASET_ROOTS = glob("./data/dataset/train/*")
VALID_DATASET_ROOTS = glob("./data/dataset/valid/*")
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
MAX_EPOCHS = 400

# Testing
PROB_THRESH = 0.75