import torch
import torch.nn as nn

class LogitConstraint(nn.Module):
    def __init__(self, forbidden_ids=None, penalty_weight=0.02):
        super().__init__()
        # Store both a tensor buffer for device placement and a python list for iteration
        forbidden_ids = list(forbidden_ids) if forbidden_ids else []
        self.register_buffer('forbidden_tensor', torch.tensor(forbidden_ids, dtype=torch.long) if forbidden_ids else torch.tensor([], dtype=torch.long))
        self.forbidden_list = forbidden_ids
        self.penalty_weight = penalty_weight

    def forward(self, logits):  # logits: B,T,V
        if len(self.forbidden_list) == 0:
            return logits, torch.zeros((), device=logits.device)
        penalty = torch.zeros((), device=logits.device)
        # subtract a small amount from forbidden token logits and accumulate a penalty
        for tid in self.forbidden_list:
            # defensively check range
            if tid < 0 or tid >= logits.size(-1):
                continue
            logits[..., int(tid)] = logits[..., int(tid)] - self.penalty_weight
            p = torch.softmax(logits, dim=-1)[..., int(tid)]
            penalty = penalty + p.mean()
        return logits, penalty * self.penalty_weight


class UnlikelihoodLoss(nn.Module):
    """Simple unlikelihood loss which penalizes probability mass on a set of undesired tokens.

    Lightweight implementation: expects a python list of forbidden ids or empty list.
    """
    def __init__(self, forbidden_ids=None, weight: float = 0.0, pad_id: int = -100):
        super().__init__()
        self.forbidden_list = list(forbidden_ids) if forbidden_ids else []
        self.weight = float(weight)
        self.pad_id = pad_id

    def forward(self, logits, targets=None):
        # logits: [B,T,V]
        if self.weight <= 0 or len(self.forbidden_list) == 0:
            return torch.zeros((), device=logits.device)
        probs = torch.softmax(logits, dim=-1)
        # build mask over forbidden tokens in vocab dimension
        mask = probs.new_zeros(probs.shape)
        for tid in self.forbidden_list:
            if 0 <= int(tid) < probs.size(-1):
                mask[..., int(tid)] = 1.0
        # per-position forbidden mass
        forbidden_mass = (probs * mask).sum(dim=-1)  # [B,T]
        # ignore padding positions if targets provided
        if targets is not None and self.pad_id is not None:
            pad_mask = (targets == self.pad_id)
            forbidden_mass = forbidden_mass.masked_fill(pad_mask, 0.0)
        # unlikelihood: encourage forbidden_mass -> 0
        ul = -torch.log(1.0 - forbidden_mass + 1e-9)
        return ul.mean() * self.weight


class ContrastiveLoss(nn.Module):
    """Lightweight contrastive loss placeholder.

    In-batch NT-Xent using cosine similarity. Returns 0 if weight == 0.
    """
    def __init__(self, temperature: float = 0.07, weight: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.weight = float(weight)

    def forward(self, embeddings_a, embeddings_b=None, labels=None):
        if self.weight <= 0:
            return torch.zeros((), device=embeddings_a.device)
        if embeddings_b is None:
            embeddings_b = embeddings_a
        a = nn.functional.normalize(embeddings_a, dim=-1)
        b = nn.functional.normalize(embeddings_b, dim=-1)
        logits = (a @ b.t()) / self.temperature
        labels = torch.arange(a.size(0), device=a.device)
        loss = nn.functional.cross_entropy(logits, labels)
        return loss * self.weight
