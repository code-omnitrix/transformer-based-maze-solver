import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Set seeds for scientific reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

# Grid & Vocabulary Configuration
GRID_SIZE = 8          # 8x8 Maze Grid
WALL_PROB = 0.28       # Probability of a cell being an obstacle

# Maze Input Vocab: 0=Free, 1=Wall, 2=Start, 3=Goal
MAZE_VOCAB = {"FREE": 0, "WALL": 1, "START": 2, "GOAL": 3}

# Action Output Vocab: PAD, BOS, EOS, and directional moves U, L, R, D
ACTION_VOCAB = {"PAD": 0, "BOS": 1, "EOS": 2, "U": 3, "L": 4, "R": 5, "D": 6}
IDX_TO_ACTION = {v: k for k, v in ACTION_VOCAB.items()}

# Coordinate deltas corresponding to U, L, R, D
MOVE_DELTAS = {
    "U": (-1, 0),
    "L": (0, -1),
    "R": (0, 1),
    "D": (1, 0)
}

