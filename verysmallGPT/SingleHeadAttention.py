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
from GPTConfig import GPTConfig
class SingleHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.head_size = config.head_size
        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.query = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)
        self.register_buffer(
            "attention_mask",
            torch.ones(config.block_size, config.block_size)
        )
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        batch_size, seq_len, dim_2_mt = x.size()
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        weight = query @ key.transpose(-2, -1)
        weight = weight.masked_fill(self.attention_mask[:seq_len, :seq_len] == 0, float("-inf"))
        weight = F.softmax(weight, dim=-1) / math.sqrt(self.head_size)
        weight = self.dropout(weight)
        out = weight @ value
        return out
if __name__ == "__main__":
    config = GPTConfig()
    single_head_attention = SingleHeadAttention(config)
    batch_size = config.batch_size
    seq_len = config.block_size
    hidden_dim = config.hidden_dim
    x = torch.randn(batch_size, seq_len, hidden_dim)  
    output = single_head_attention(x)
    print("Output shape:", output.shape)
