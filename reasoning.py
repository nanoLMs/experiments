import torch
import torch.nn as nn
import torch.nn.functional as F

class ReasoningAggregator(nn.Module):
    """Lightweight reasoning head: aggregates last-K token states to produce auxiliary prediction.
       Used to encourage chain-of-thought compression without full exposure."""
    def __init__(self, d_model: int, hidden: int, vocab_size: int):
        super().__init__()
        self.lin1 = nn.Linear(d_model, hidden)
        self.lin2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):  # x: B,T,C
        # Aggregate final 8 tokens average
        k = min(8, x.size(1))
        segment = x[:, -k:].mean(dim=1)  # B,C
        h = F.gelu(self.lin1(segment))  # FIXED: Use F.gelu instead of torch.gelu
        return self.lin2(h)  # B,V
