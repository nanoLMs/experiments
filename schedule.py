import math

def cosine_with_warmup(step, warmup, max_steps, base_lr, min_lr):
    if step < warmup:
        return base_lr * step / max(warmup,1)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5*(base_lr - min_lr)*(1 + math.cos(math.pi*progress))
