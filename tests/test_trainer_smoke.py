import pytest
import torch
from config import TrainConfig
from model_moe import NanoMoEModel


def test_model_forward_shapes():
    cfg = TrainConfig()
    cfg.vocab_size = 256
    cfg.n_layers = 2
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.d_ff = 128
    cfg.mtp_heads = 1
    cfg.mtp_k = 2

    B = 2
    T = 16
    device = torch.device('cpu')

    model = NanoMoEModel(cfg).to(device)
    model.eval()

    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    with torch.no_grad():
        logits_main, logits_mtp, aux_loss, reason_logits = model(x)

    assert logits_main.shape == (B, T, cfg.vocab_size)
    assert isinstance(logits_mtp, list)
    if cfg.mtp_heads > 0:
        assert len(logits_mtp) == cfg.mtp_heads
        assert logits_mtp[0].shape[0] == B
        assert logits_mtp[0].shape[1] == T


def test_smoke_train_step():
    cfg = TrainConfig()
    cfg.vocab_size = 256
    cfg.n_layers = 2
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.d_ff = 128
    cfg.mtp_heads = 1
    cfg.mtp_k = 2

    B = 2
    T = 16
    device = torch.device('cpu')

    model = NanoMoEModel(cfg).to(device)
    model.train()

    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    y = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    logits_main, logits_mtp, aux_loss, reason_logits = model(x)
    criterion = torch.nn.CrossEntropyLoss()
    loss_main = criterion(logits_main.view(-1, logits_main.size(-1)), y.view(-1))
    total_loss = loss_main + aux_loss
    total_loss.backward()

    assert total_loss.item() >= 0.0
