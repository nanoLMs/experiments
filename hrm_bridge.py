
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class HRMBridge(nn.Module):
    """
    Wraps a token-level model (e.g., Transformer decoder) to provide segmented 1-step-grad training
    in the spirit of the HRM paper:
      - Run (N*T - 1) recurrent "updates" under no_grad
      - Do the final update with grad
      - Repeat for M segments with state detaching between segments
    We treat the model's cached key/values + final hidden as the state.
    The wrapped model must implement:
      forward(input_ids, past_state=None, return_state=True) -> (logits, new_state)
    """
    def __init__(self, base_model: nn.Module, N_cycles: int = 2, T_steps: int = 2):
        super().__init__()
        self.base = base_model
        self.N_cycles = N_cycles
        self.T_steps = T_steps

    @torch.no_grad()
    def _step_nograd(self, x, state):
        logits, state = self.base(x, past_state=state, return_state=True)
        return logits, state

    def _step_grad(self, x, state):
        logits, state = self.base(x, past_state=state, return_state=True)
        return logits, state

    def segment(self, x, state=None):
        # x: [B, T]
        total = self.N_cycles * self.T_steps
        for _ in range(total-1):
            _, state = self._step_nograd(x, state)
        logits, state = self._step_grad(x, state)
        return logits, state

    def forward(self, x, segments: int = 1):
        logits_list: List[torch.Tensor] = []
        state = None
        for _ in range(segments):
            logits, state = self.segment(x, state)
            logits_list.append(logits)
            # detach state between segments
            if isinstance(state, tuple):
                state = tuple(s.detach() if torch.is_tensor(s) else s for s in state)
            elif torch.is_tensor(state):
                state = state.detach()
        return logits_list
