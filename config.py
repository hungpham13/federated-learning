import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else 'cpu')  # Try "cuda" to train on GPU
NUM_CLIENTS = 10

CLASSES = ('benign', 'non-neoplastic', 'malignant')
CLASS_WEIGHTS = [0.14, 0.73, 0.13]

LABEL_KEY = 'three_partition_label'

RUN_ID = 'runs/centralize-stratrified-skin-sampling-img-232-efficientnet'
