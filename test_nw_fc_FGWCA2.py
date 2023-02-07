import torch.nn as nn
import torch
from torch import Tensor
from typing import Type
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.optim as optim
import numpy as np
import random
import math
import torch.nn.functional as F
from aircraft import Aircraft

