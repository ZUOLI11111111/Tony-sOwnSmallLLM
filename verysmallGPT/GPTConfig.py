from timeit import main
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as DataSet
import torch.utils.data as DataLoader
import math
from dataclasses import dataclass
import numpy as np
import torch_directml
device = torch_directml.device()
torch.manual_seed(1024)
@dataclass
class GPTConfig:
    block_size: int = 512
    batch_size: int = 12
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    hidden_dim: int = n_embd
    dropout: float = 0.1
    head_size: int = n_embd // n_head
    vocab_size: int = 50257
    device: torch.device = device
if __name__ == "__main__":
    config = GPTConfig()
    print(config.device)