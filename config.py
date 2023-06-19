import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else 'cpu')  # Try "cuda" to train on GPU
NUM_CLIENTS = 10

CLASSES = ('benign', 'non-neoplastic', 'malignant')

LABEL_KEY = 'three_partition_label'

RUN_ID = 'runs/centralize-stratrified-skin-sampling'