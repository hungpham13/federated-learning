import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')  # Try "cuda" to train on GPU
NUM_CLIENTS = 10
