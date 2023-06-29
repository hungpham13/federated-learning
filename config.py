import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else 'cpu')  # Try "cuda" to train on GPU
NUM_CLIENTS = 10

CLASSES = ('benign', 'non-neoplastic', 'malignant')
CLASS_WEIGHTS = [0.3, 0.1, 0.6]

LABEL_KEY = 'three_partition_label'

RUN_ID = 'runs/centralize-weighted-loss-128-resnet152-adam'
