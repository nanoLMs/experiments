import torch
from torch.utils.data import Dataset, DataLoader
import os
import mmap
import gc
from typing import List, Optional

class MemoryEfficientDataset(Dataset):
    """EXTREMELY memory-efficient dataset using memory mapping and streaming"""

    def __init__(self, path: str, tokenizer, seq_len: int, max_samples: Optional[int] = None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Corpus file not found: {path}")

        self.path = path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples

        print(f"📖 Memory-mapping corpus: {path}")

        # Get file size
        file_size = os.path.getsize(path)
        print(f"📊 Corpus size: {file_size / 1e6:.1f}MB")

        # Memory map the file (doesn't load into RAM!)
        self.file = open(path, 'r', encoding='utf-8')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

        # Sample the file to estimate tokens per character
        sample_size = min(10000, file_size // 2)
        sample_text = self.mmap[:sample_size].decode('utf-8', errors='ignore')
        sample_tokens = len(self.tokenizer(sample_text, add_special_tokens=False)['input_ids'])

        # Estimate total tokens and samples
        self.tokens_per_char = sample_tokens / len(sample_text)
        self.estimated_tokens = int(file_size * self.tokens_per_char)
        self.estimated_samples = max(0, self.estimated_tokens - seq_len)

        # Limit dataset size if specified
        if max_samples and max_samples < self.estimated_samples:
            self.estimated_samples = max_samples

        print(f"✅ Dataset ready: ~{self.estimated_tokens:,} tokens, ~{self.estimated_samples:,} samples")
        print(f"💾 Memory usage: ~{sample_size / 1e6:.1f}MB (memory-mapped)")

        # Cache for recent chunks
        self.chunk_cache = {}
        self.chunk_size = 50000  # Characters per chunk
        self.max_cached_chunks = 3  # Keep only 3 chunks in memory

    def __len__(self):
        return self.estimated_samples

    def __getitem__(self, idx):
        try:
            # Calculate which part of file we need
            chars_per_token = 1.0 / self.tokens_per_char
            start_char = int(idx * chars_per_token)

            # Determine chunk
            chunk_id = start_char // self.chunk_size

            # Get chunk from cache or load it
            if chunk_id not in self.chunk_cache:
                self._load_chunk(chunk_id)

            # Get text chunk
            chunk_start = chunk_id * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size + self.seq_len * 10, len(self.mmap))

            chunk_text = self.mmap[chunk_start:chunk_end].decode('utf-8', errors='ignore')

            # Tokenize chunk
            if chunk_id not in self.chunk_cache:
                tokens = self.tokenizer(chunk_text, add_special_tokens=False)['input_ids']
                self.chunk_cache[chunk_id] = torch.tensor(tokens, dtype=torch.long)

                # Limit cache size
                if len(self.chunk_cache) > self.max_cached_chunks:
                    oldest_chunk = min(self.chunk_cache.keys())
                    del self.chunk_cache[oldest_chunk]
                    gc.collect()

            tokens = self.chunk_cache[chunk_id]

            # Extract sequence
            local_idx = max(0, start_char - chunk_start)
            token_start = int(local_idx * self.tokens_per_char)

            # Ensure we don't exceed available tokens
            max_start = max(0, len(tokens) - self.seq_len - 1)
            token_start = min(token_start, max_start)

            if token_start + self.seq_len + 1 <= len(tokens):
                x = tokens[token_start:token_start + self.seq_len]
                y = tokens[token_start + 1:token_start + self.seq_len + 1]

                # Ensure exact length match
                if len(x) == self.seq_len and len(y) == self.seq_len:
                    return x, y

            # Fallback: create from beginning of chunk
            if len(tokens) >= self.seq_len + 1:
                x = tokens[:self.seq_len]
                y = tokens[1:self.seq_len + 1]
                return x, y
            else:
                # Pad if necessary
                pad_id = self.tokenizer.pad_token_id or 0
                x = torch.full((self.seq_len,), pad_id)
                y = torch.full((self.seq_len,), pad_id)
                return x, y

        except Exception as e:
            print(f"⚠️ Error loading sample {idx}: {e}")
            # Return padded sequence as fallback
            pad_id = self.tokenizer.pad_token_id or 0
            x = torch.full((self.seq_len,), pad_id)
            y = torch.full((self.seq_len,), pad_id)
            return x, y

    def _load_chunk(self, chunk_id):
        """Load a chunk of text and tokenize it"""
        try:
            chunk_start = chunk_id * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size + self.seq_len * 10, len(self.mmap))

            chunk_text = self.mmap[chunk_start:chunk_end].decode('utf-8', errors='ignore')
            tokens = self.tokenizer(chunk_text, add_special_tokens=False)['input_ids']
            self.chunk_cache[chunk_id] = torch.tensor(tokens, dtype=torch.long)

        except Exception as e:
            print(f"⚠️ Error loading chunk {chunk_id}: {e}")

    def __del__(self):
        """Clean up memory mapping"""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'file'):
            self.file.close()


def create_dataloader(cfg, tokenizer, world_size=1):
    """Create memory-efficient dataloader"""

    # Limit dataset size for memory safety
    max_samples = None
    if hasattr(cfg, 'max_dataset_samples'):
        max_samples = cfg.max_dataset_samples
    else:
        # Auto-limit based on available RAM
        import psutil
        available_ram_gb = psutil.virtual_memory().available / 1e9

        if available_ram_gb < 8:
            max_samples = 50000  # Very limited
            print(f"⚠️ Low RAM ({available_ram_gb:.1f}GB) - limiting dataset to {max_samples:,} samples")
        elif available_ram_gb < 16:
            max_samples = 100000  # Moderate
            print(f"⚠️ Moderate RAM ({available_ram_gb:.1f}GB) - limiting dataset to {max_samples:,} samples")
        else:
            max_samples = 500000  # More generous but still limited
            print(f"✅ Good RAM ({available_ram_gb:.1f}GB) - limiting dataset to {max_samples:,} samples")

    # Create memory-efficient dataset
    dataset = MemoryEfficientDataset(
        path=cfg.train_corpus,
        tokenizer=tokenizer,
        seq_len=cfg.seq_len,
        max_samples=max_samples
    )

    # Force garbage collection
    gc.collect()

    # Create dataloader with memory optimizations
    num_workers = min(cfg.dataloader_workers, 2)  # Limit workers to save memory

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=False,  # Don't shuffle to maintain locality
        num_workers=num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        persistent_workers=False,  # Don't keep workers alive
        prefetch_factor=1 if num_workers > 0 else None,  # Only set prefetch_factor with workers
        drop_last=True
    )

    print(f"🚀 DataLoader created:")
    print(f"  - Batch size: {cfg.micro_batch_size}")
    print(f"  - Workers: {min(cfg.dataloader_workers, 2)}")
    print(f"  - Memory efficient: ✅")
    print(f"  - Estimated batches per epoch: {len(dataloader):,}")

    return dataloader
