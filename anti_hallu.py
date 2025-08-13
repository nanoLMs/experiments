import torch
import torch.nn as nn

class LogitConstraint(nn.Module):
    def __init__(self, forbidden_ids=None, penalty_weight=0.02):
        super().__init__()
        self.register_buffer('forbidden', torch.tensor(forbidden_ids if forbidden_ids else [], dtype=torch.long))
        self.penalty_weight = penalty_weight

    def forward(self, logits):  # logits: B,T,V
        if self.forbidden.numel()==0:
            return logits, torch.zeros((), device=logits.device)
        penalty = torch.zeros_like(logits[...,0])
        for tid in self.forbidden:
            logits[..., tid] -= self.penalty_weight
            penalty = penalty + torch.clamp(-logits[..., tid], 0, 10).mean()
        return logits, penalty * self.penalty_weight
