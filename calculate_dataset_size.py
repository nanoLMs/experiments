
import os
import mmap
from transformers import PreTrainedTokenizerFast
from config import TrainConfig

# --- Simplified from data_loader.py ---
class MemoryEfficientDataset:
    def __init__(self, path: str, tokenizer, seq_len: int, max_samples: int | None = None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Corpus file not found: {path}")

        file_size = os.path.getsize(path)
        self.file = open(path, 'r', encoding='utf-8')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

        sample_size = min(10000, file_size // 2 if file_size > 0 else 0)
        sample_text = self.mmap[:sample_size].decode('utf-8', errors='ignore')
        sample_tokens = len(tokenizer(sample_text, add_special_tokens=False)['input_ids'])

        self.tokens_per_char = sample_tokens / len(sample_text) if len(sample_text) > 0 else 0
        self.estimated_tokens = int(file_size * self.tokens_per_char)
        self.estimated_samples = max(0, self.estimated_tokens - seq_len)

        if max_samples and max_samples < self.estimated_samples:
            self.estimated_samples = max_samples

    def __len__(self):
        return self.estimated_samples

    def __del__(self):
        self.mmap.close()
        self.file.close()
# --- End simplified section ---

try:
    # Instantiate configuration and tokenizer
    cfg = TrainConfig()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_dir)

    # Instantiate dataset to calculate its size
    dataset = MemoryEfficientDataset(
        path=cfg.train_corpus,
        tokenizer=tokenizer,
        seq_len=cfg.seq_len,
        max_samples=cfg.max_dataset_samples
    )

    num_samples = len(dataset)
    batches_per_epoch = num_samples // cfg.micro_batch_size
    total_optimizer_steps = (batches_per_epoch * cfg.num_epochs) // cfg.grad_accum_steps

    print(f"Configuration:")
    print(f"  - Training Corpus: {cfg.train_corpus}")
    print(f"  - Num Epochs: {cfg.num_epochs}")
    print(f"  - Max Dataset Samples: {cfg.max_dataset_samples:,}")
    print(f"  - Micro Batch Size: {cfg.micro_batch_size}")
    print(f"  - Gradient Accumulation Steps: {cfg.grad_accum_steps}")
    print("-" * 20)
    print(f"Calculations:")
    print(f"  - Estimated Samples in Dataset: {num_samples:,}")
    print(f"  - Batches per Epoch: {batches_per_epoch:,}")
    print(f"  - Total Optimizer Steps: {total_optimizer_steps:,}")

except Exception as e:
    print(f"An error occurred: {e}")

