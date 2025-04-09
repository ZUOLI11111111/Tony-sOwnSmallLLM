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
from SingleHeadAttention import SingleHeadAttention
device = torch_directml.device()
from GPTConfig import GPTConfig
class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList(
            SingleHeadAttention(config)
            for _ in range(config.n_head)
        )
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.proj(torch.cat(
            [h(x) for h in (self.heads)],
            dim = -1
        )))    
if __name__ == "__main__":
    config = GPTConfig()
    multi_head_attention = MultiHeadAttention(config)
    batch_size = config.batch_size
    seq_len = config.block_size
    hidden_dim = config.hidden_dim
    x = torch.randn(batch_size, seq_len, hidden_dim)  
    output = multi_head_attention(x)
    print("Output:", output)