import torch
from config import TrainConfig
from model_moe import NanoMoEModel


def main():
    cfg = TrainConfig()
    # tiny configuration for smoke test
    cfg.vocab_size = 512
    cfg.n_layers = 2
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.d_ff = 128
    cfg.mtp_heads = 1
    cfg.mtp_k = 2
    cfg.use_bnb_4bit = False
    cfg.use_fp4 = False
    cfg.gradient_checkpointing = False
    cfg.use_hrm = False

    B = 2
    T = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NanoMoEModel(cfg).to(device)
    model.train()

    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    y = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    logits_main, logits_mtp, aux_loss, reason_logits = model(x)

    criterion = torch.nn.CrossEntropyLoss()
    loss_main = criterion(logits_main.view(-1, logits_main.size(-1)), y.view(-1))

    # compute mtp loss if model helper exists
    mtp_loss_val = torch.tensor(0.0, device=device)
    try:
        from model_moe import mtp_loss as model_mtp_loss
        if logits_mtp and len(logits_mtp) > 0:
            mtp_loss_val = model_mtp_loss(logits_mtp, y, cfg.mtp_k)
    except Exception:
        mtp_loss_val = torch.tensor(0.0, device=device)

    total_loss = loss_main + mtp_loss_val + aux_loss
    total_loss.backward()

    print(f"SMOKE TEST OK - total_loss={total_loss.item():.6f}")


if __name__ == '__main__':
    main()
