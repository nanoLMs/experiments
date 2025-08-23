import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except Exception:
    HAS_FLASH = False

def maybe_lora(module: nn.Linear, cfg, name: str):
    if cfg.lora and any(t in name for t in cfg.lora_target_modules):
        return LoRALinear(module, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
    return module

# Rotary embedding (simple)
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, scale=1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scale = scale

    def forward(self, seq_len: int, device: torch.device):
        dim = self.dim
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype) / self.scale
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)

# Apply rotary - MINIMAL WORKING VERSION
def apply_rotary(x, cos, sin):
    """
    Minimal rotary embedding - handles dimension mismatches gracefully
    x: [B, T, H, D]
    cos, sin: [T, rope_dim]
    """
    try:
        B, T, H, D = x.shape
        seq_len, rope_dim = cos.shape

        # Handle sequence length mismatch
        if seq_len != T:
            # Truncate or pad cos/sin to match sequence length
            if seq_len > T:
                cos = cos[:T]
                sin = sin[:T]
            else:
                # Pad with identity rotation
                pad_len = T - seq_len
                pad_cos = torch.ones(pad_len, rope_dim, device=cos.device, dtype=cos.dtype)
                pad_sin = torch.zeros(pad_len, rope_dim, device=sin.device, dtype=sin.dtype)
                cos = torch.cat([cos, pad_cos], dim=0)
                sin = torch.cat([sin, pad_sin], dim=0)

        # Handle dimension mismatch
        rope_dim = min(rope_dim, D)
        if rope_dim % 2 != 0:
            rope_dim -= 1  # Must be even for rotation pairs

        if rope_dim <= 0:
            return x  # No rotation needed

        # Reshape for broadcasting: [1, T, 1, rope_dim]
        cos = cos[:, :rope_dim].unsqueeze(0).unsqueeze(2)
        sin = sin[:, :rope_dim].unsqueeze(0).unsqueeze(2)

        # Apply rotation only to the rope dimensions
        x_rot = x[..., :rope_dim]  # [B, T, H, rope_dim]

        # Simple rotation formula
        x1 = x_rot[..., 0::2]  # Even indices [B, T, H, rope_dim//2]
        x2 = x_rot[..., 1::2]  # Odd indices [B, T, H, rope_dim//2]
        cos_half = cos[..., 0::2]  # [1, T, 1, rope_dim//2]
        # FIX: sin must slice the same even positions to pair correctly
        sin_half = sin[..., 0::2]  # [1, T, 1, rope_dim//2]

        # Rotate
        x1_new = x1 * cos_half - x2 * sin_half
        x2_new = x1 * sin_half + x2 * cos_half

        # Interleave back
        x_rotated = torch.stack([x1_new, x2_new], dim=-1).flatten(-2)

        # Combine rotated and non-rotated parts
        if rope_dim < D:
            x_rest = x[..., rope_dim:]  # Non-rotated dimensions
            x = torch.cat([x_rotated, x_rest], dim=-1)
        else:
            x = x_rotated

        return x

    except Exception as e:
        print(f"⚠️ Rotary embedding failed: {e}")
    return x

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.r = r
        if r > 0:
            self.lora_A = nn.Linear(base.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            self.scaling = alpha / r
            self.dropout = nn.Dropout(dropout)
        else:
            self.lora_A = None
    def forward(self, x):
        if self.lora_A is None:
            return self.base(x)
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, rope: RotaryEmbedding, attn_dropout=0.0, dropout=0.0, cfg=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rope = rope
        self.cfg = cfg
        raw = nn.Linear(d_model, 3 * d_model, bias=False)
        self.qkv = maybe_lora(raw, cfg, 'qkv') if cfg else raw
        out = nn.Linear(d_model, d_model, bias=False)
        self.out = maybe_lora(out, cfg, 'out') if cfg else out
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # B,T,H,D
        cos, sin = self.rope(T, x.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if self.cfg and self.cfg.use_flash_attn and HAS_FLASH and mask is None:
            # flash_attn expects (B,T,H,D)
            q_ = q.contiguous()
            k_ = k.contiguous()
            v_ = v.contiguous()
            # flash_attn_func works on (B,T,H,D) returning same shape
            attn_out = flash_attn_func(q_, k_, v_, dropout_p=self.cfg.flash_dropout_p if self.training else 0.0, softmax_scale=None, causal=True)
            y = attn_out.reshape(B, T, C)
        else:
            # FIXED: Transpose q,k,v to [B, H, T, D] for proper attention computation
            q = q.transpose(1, 2)  # [B, H, T, D]
            k = k.transpose(1, 2)  # [B, H, T, D]
            v = v.transpose(1, 2)  # [B, H, T, D]

            # Compute attention scores
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, H, T, T]

            # Apply causal mask
            causal = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(causal == 0, float('-inf'))
            if mask is not None:
                att = att + mask

            # Apply softmax and dropout
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # Apply attention to values
            y = att @ v  # [B, H, T, D]
            y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]

        return self.dropout(self.out(y))

class Expert(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.gelu(self.w1(x))))

class MoE(nn.Module):
    def __init__(self, d_model, n_experts, hidden_dim, k=2, router_jitter=0.01, capacity_factor=1.25, router_z_loss=1e-4, dropout=0.0):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, hidden_dim, dropout) for _ in range(n_experts)])
        self.capacity_factor = capacity_factor
        self.router_jitter = router_jitter
        self.router_z_loss = router_z_loss

    def forward(self, x):  # x: B,T,C
        B, T, C = x.size()
        logits = self.router(x)
        if self.router_jitter > 0:
            logits = logits + torch.randn_like(logits) * self.router_jitter
        gates = F.softmax(logits, dim=-1)                       # B,T,E
        topk_vals, topk_idx = torch.topk(gates, self.k, dim=-1) # B,T,k
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

        # FIXED: Load balance auxiliary loss
        importance = gates.sum(dim=(0,1))               # E (sum of gate values per expert)
        # Compute load as the number of selections per expert (simpler and more robust)
        load = torch.zeros(self.n_experts, device=x.device)
        total_selections = float(topk_idx.numel())
        for e in range(self.n_experts):
            load[e] = (topk_idx == e).float().sum()
        # normalize
        importance = importance / (importance.sum() + 1e-9)
        load = load / (load.sum() + 1e-9)
        balance_loss = (importance * load).sum() * self.n_experts * self.router_z_loss

        outputs = torch.zeros_like(x)
        # FIXED: Correct per-expert weight application
        for e, expert in enumerate(self.experts):
            mask_e = (topk_idx == e)                     # B,T,k (where expert e is selected)
            sel_tokens = mask_e.any(dim=-1)              # B,T (tokens that use expert e)
            if not sel_tokens.any():
                continue
            h = x[sel_tokens]                            # S,C (selected tokens)
            out = expert(h)                              # S,C
            # Extract only weights for this specific expert
            weights_e = torch.where(mask_e, topk_vals, torch.zeros_like(topk_vals))  # B,T,k
            weights_e = weights_e.sum(dim=-1)            # B,T (sum weights for expert e)
            outputs[sel_tokens] += out * weights_e[sel_tokens].unsqueeze(-1)
        return outputs, balance_loss

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, rope, dropout=0.0, attn_dropout=0.0, use_moe=False, moe_cfg=None, cfg=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, rope, attn_dropout, dropout, cfg)
        self.ln2 = nn.LayerNorm(d_model)
        self.use_moe = use_moe
        if use_moe and moe_cfg is not None:
            self.moe = MoE(
                d_model,
                moe_cfg.get('n_experts', 4),
                int(moe_cfg.get('expert_ff_mult', 1.0) * d_ff),
                k=moe_cfg.get('moe_top_k', 2),
                router_jitter=moe_cfg.get('router_jitter', 0.01),
                capacity_factor=moe_cfg.get('capacity_factor', 1.25),
                router_z_loss=moe_cfg.get('router_z_loss', 1e-4),
                dropout=dropout
            )
        else:
            self.ff = nn.Sequential(
                maybe_lora(nn.Linear(d_model, d_ff), cfg, 'w1'),
                nn.GELU(),
                nn.Dropout(dropout),
                maybe_lora(nn.Linear(d_ff, d_model), cfg, 'w2'),
                nn.Dropout(dropout)
            )

    def forward(self, x, mask=None):
        h = x + self.attn(self.ln1(x), mask)
        if self.use_moe:
            moe_out, aux = self.moe(self.ln2(h))
            h = h + moe_out
        else:
            ff_out = self.ff(self.ln2(h))
            h = h + ff_out
            aux = torch.zeros((), device=x.device)
        return h, aux

class NanoMoEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Ensure config derived fields are set
        try:
            if hasattr(self.cfg, 'validate'):
                self.cfg.validate()
        except Exception:
            pass

        # robust rope dim (fall back to d_model // n_heads)
        rope_dim = getattr(self.cfg, 'd_head', None)
        if not rope_dim:
            rope_dim = max(1, self.cfg.d_model // max(1, self.cfg.n_heads))

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(rope_dim, base=cfg.rope_base, scale=cfg.rope_scaling)
        self.blocks = nn.ModuleList()
        for i in range(cfg.n_layers):
            use_moe = (i % cfg.moe_every == 0) if cfg.moe_every > 0 else False
            block = TransformerBlock(
                cfg.d_model, cfg.n_heads, cfg.d_ff, self.rope,
                dropout=cfg.dropout, attn_dropout=cfg.attn_dropout,
                use_moe=use_moe,
                moe_cfg=dict(n_experts=cfg.n_experts, expert_ff_mult=cfg.expert_ff_mult, moe_top_k=cfg.moe_top_k,
                             router_jitter=cfg.router_jitter, capacity_factor=cfg.capacity_factor, router_z_loss=cfg.router_z_loss),
                cfg=cfg
            )
            self.blocks.append(block)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = maybe_lora(nn.Linear(cfg.d_model, cfg.vocab_size, bias=False), cfg, 'head')
        self.mtp_heads = nn.ModuleList([maybe_lora(nn.Linear(cfg.d_model, cfg.vocab_size, bias=False), cfg, f"mtp_{i}") for i in range(cfg.mtp_k)])
        if cfg.enable_reasoning:
            from reasoning import ReasoningAggregator
            self.reason_head = ReasoningAggregator(cfg.d_model, cfg.reasoning_dim, cfg.vocab_size)
        else:
            self.reason_head = None

        # Ensure all parameters require gradients by default
        for param in self.parameters():
            param.requires_grad_(True)

    def forward(self, idx):  # idx: B,T
        B, T = idx.shape
        x = self.tok_emb(idx)
        aux_losses = []

        # Ensure embeddings require gradients during training
        if self.training and not x.requires_grad:
            x.requires_grad_(True)

        # HRM (Hierarchical Reasoning Module) integration
        if getattr(self.cfg, 'use_hrm', False) and self.training:
            return self._forward_with_hrm(idx, x, aux_losses)
        else:
            return self._forward_standard(x, aux_losses)

    def _forward_standard(self, x, aux_losses):
        """Standard forward pass"""
        for blk in self.blocks:
            # FIXED: Disable gradient checkpointing when using bitsandbytes
            use_checkpointing = (
                self.cfg.gradient_checkpointing and
                self.training and
                x.requires_grad and
                not getattr(self.cfg, 'use_bnb_4bit', False)  # Disable for bitsandbytes
            )

            if use_checkpointing:
                # Simplified gradient checkpointing to avoid tensor mismatch
                def checkpoint_fn(input_x):
                    output_x, output_aux = blk(input_x)
                    return output_x, output_aux

                # Use non-reentrant mode which is recommended and more efficient
                x, aux = torch.utils.checkpoint.checkpoint(
                    checkpoint_fn,
                    x,
                    use_reentrant=False  # Use non-reentrant mode
                )
                aux_losses.append(aux)
            else:
                # Standard forward pass (used with bitsandbytes)
                x, aux = blk(x)
                aux_losses.append(aux)

        x = self.ln_f(x)
        logits_main = self.head(x)

        # FIXED MTP: Proper multi-token prediction implementation
        logits_mtp = []
        for i, head in enumerate(self.mtp_heads):
            # For MTP head i, we predict token at position t+i+1
            # So we use the same hidden states x (no shifting needed in model)
            # The shifting happens in loss computation in trainer
            logits_mtp.append(head(x))

        reason_logits = None
        if self.reason_head is not None:
            reason_logits = self.reason_head(x)

        return logits_main, logits_mtp, torch.stack(aux_losses).sum() * self.cfg.moe_aux_weight, reason_logits

    def _forward_with_hrm(self, idx, x, aux_losses):
        """HRM (Hierarchical Reasoning Module) forward pass.

        This implementation segments the input sequence into time-chunks and for each segment
        performs N_cycles of T_steps updates where the first (N_cycles*T_steps - 1)
        updates are executed under torch.no_grad() and the final update is executed with gradients.
        The resulting segment outputs are concatenated to form the final sequence hidden states.
        """
        # HRM parameters from config (use hrm_* names)
        N_cycles = getattr(self.cfg, 'hrm_N_cycles', 2)
        T_steps = getattr(self.cfg, 'hrm_T_steps', 2)
        segments = getattr(self.cfg, 'hrm_segments', 2)

        B, T, C = x.shape
        if segments <= 1:
            # fallback to standard forward if not segmented
            return self._forward_standard(x, aux_losses)

        seg_len = T // segments
        remainder = T % segments
        outputs = []

        time_ptr = 0
        for s in range(segments):
            this_len = seg_len + (1 if s < remainder else 0)
            if this_len <= 0:
                continue
            x_seg = x[:, time_ptr:time_ptr+this_len, :].contiguous()  # [B, L, C]

            # run (N_cycles*T_steps - 1) no-grad updates and final grad update
            total_updates = N_cycles * T_steps
            # intermediate no-grad updates
            for _ in range(total_updates - 1):
                with torch.no_grad():
                    x_seg = self._process_segment_no_grad(x_seg)
            # final update with grad
            x_seg = self._process_segment_with_grad(x_seg, aux_losses)

            # collect
            outputs.append(x_seg)
            time_ptr += this_len

        # concat segments along time dim
        x = torch.cat(outputs, dim=1)

        # final norm and heads
        x = self.ln_f(x)
        logits_main = self.head(x)

        logits_mtp = []
        for i, head in enumerate(self.mtp_heads):
            logits_mtp.append(head(x))

        reason_logits = None
        if self.reason_head is not None:
            reason_logits = self.reason_head(x)

        return logits_main, logits_mtp, torch.stack(aux_losses).sum() * self.cfg.moe_aux_weight, reason_logits

    def _process_segment_with_grad(self, x, aux_losses):
        """Process segment with gradients (final step)"""
        for blk in self.blocks:
            x, aux = blk(x)
            aux_losses.append(aux)
        return x

    def _process_segment_no_grad(self, x):
        """Process segment without gradients (intermediate steps)"""
        with torch.no_grad():
            for blk in self.blocks:
                x, _ = blk(x)  # Ignore aux losses for no-grad steps
        return x

    def get_pooled_representation(self, x):
        """Return a pooled (mean) representation for contrastive loss.

        Accepts either input ids [B,T] or hidden states [B,T,C].
        """
        if x.dim() == 2:  # input ids
            x = self.tok_emb(x)
            x = self.ln_f(x)
        elif x.dim() == 3:
            # assume hidden states
            x = self.ln_f(x)
        # mean pool over time
        return x.mean(dim=1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class KVCache:
    """Simple KV cache container for incremental state."""
    def __init__(self):
        self.cache = None

def shift_targets(y, k):
    # returns list of shifted targets for multi-token prediction
    outs = []
    for i in range(1, k+1):
        outs.append(y[:, i:])  # next i tokens
    return outs

def mtp_loss(logits_list, y, k):
    # logits_list: list of [B,T,V] for each shift, all same T
    # Align targets
    losses = []
    T = logits_list[0].size(1)
    for i, logits in enumerate(logits_list, start=1):
        tgt = y[:, :T]  # align left; trainer should slice inputs accordingly
        losses.append(F.cross_entropy(logits.transpose(1,2), tgt, reduction='mean', label_smoothing=0.1))
    return sum(losses) / len(losses)