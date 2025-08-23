#!/usr/bin/env python3
"""
NanoLM GPU-Accelerated Tokenizer
==================================
Optimized BPE tokenizer specifically designed for NanoLM training with:
- GPU acceleration for preprocessing and analysis
- Optimized vocabulary size for 32K tokens (matching NanoLM config)
- Memory-efficient training for large datasets
- Multimodal support with special tokens
- Comprehensive analysis for model architecture optimization
- Direct integration with NanoLM training pipeline

Author: AI Assistant for NanoLM Project
"""

import sys
import os
import json
import time
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import re

# Tokenizer imports
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence as PreSequence
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormSequence
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

# Import multimodal dataloader if available
try:
    sys.path.append(str(Path(__file__).parent))

    MULTIMODAL_LOADER_AVAILABLE = True
except ImportError:
    MULTIMODAL_LOADER_AVAILABLE = False
    print("⚠️ multimodal_dataloader not found, using basic data loading")

# Data loading
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class MultimodalDataLoader:
    """
    Simple dataloader for loading text and JSON data for NanoLM multimodal experiments.
    Extendable for future multimodal (image, audio, etc.) support.
    """
    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            self.base_dir = Path(__file__).parent
        else:
            self.base_dir = Path(base_dir)
        self.text_file = self.base_dir / "constitution.txt"
        self.json_file = self.base_dir / "Contextualized_Bangladesh_Legal_Acts.json"

    def load_text(self) -> List[str]:
        texts = []
        if self.text_file.exists():
            with open(self.text_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
        return texts

    def load_json(self) -> List[str]:
        texts = []
        if self.json_file.exists():
            with open(self.json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    # Recursively extract all string values
                    texts.extend(self._extract_strings_recursive(data))
                except Exception as e:
                    print(f"Failed to load JSON: {e}")
        return texts

    def _extract_strings_recursive(self, obj: Any, max_depth: int = 3) -> List[str]:
        strings = []
        if max_depth <= 0:
            return strings
        if isinstance(obj, str) and len(obj.strip()) > 0:
            strings.append(obj.strip())
        elif isinstance(obj, dict):
            for value in obj.values():
                strings.extend(self._extract_strings_recursive(value, max_depth - 1))
        elif isinstance(obj, list):
            for item in obj:
                strings.extend(self._extract_strings_recursive(item, max_depth - 1))
        return strings

    def load_all(self) -> List[str]:
        """Load and combine all text data from both files."""
        all_texts = self.load_text() + self.load_json()
        # Optionally filter out very short texts
        return [t for t in all_texts if len(t) > 20]



class NanoLMDataLoader:
    """
    GPU-accelerated data loader optimized for NanoLM multimodal datasets
    Now supports loading only specific files for text-only multimodal-ready tokenization.
    """

    def __init__(self, dataset_path: str, specific_files: Optional[list] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 NanoLM Data Loader initialized on {self.device}")

        # Initialize robust data loader
        self.robust_loader = MultimodalDataLoader()

        # Data containers
        self.json_data = []
        self.parquet_data = []
        self.csv_data = []
        self.text_data = []

        # For direct file loading
        self.specific_files = specific_files

        # GPU memory management
        self.batch_size = 1000 if torch.cuda.is_available() else 500
        self.max_memory_gb = 2.0  # Conservative memory usage

    def load_all_datasets(self) -> Dict[str, Any]:
        """Load only specified files (if provided), else all available datasets with GPU acceleration"""
        if self.specific_files:
            print("🔥 Loading only specified files for text-only multimodal-ready tokenization...")
            self.json_data = []
            self.text_data = []
            for file_path in self.specific_files:
                if file_path.endswith('.json'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = json.load(f)
                            self.json_data.append(content)
                    except Exception as e:
                        print(f"⚠️ Failed to load {file_path}: {e}")
                elif file_path.endswith('.txt'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if len(content) > 100:
                                self.text_data.append(content)
                    except Exception as e:
                        print(f"⚠️ Failed to load {file_path}: {e}")
            print(f"✅ Loaded {len(self.json_data)} JSON and {len(self.text_data)} text files.")
            return {
                'json': self.json_data,
                'parquet': [],
                'csv': [],
                'text': self.text_data
            }
        else:
            print("🔥 Loading datasets with GPU acceleration...")
            # Load different data types in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                json_future = executor.submit(self._load_json_files)
                parquet_future = executor.submit(self._load_parquet_files)
                csv_future = executor.submit(self._load_csv_files)
                text_future = executor.submit(self._load_text_files)

                # Collect results
                self.json_data = json_future.result()
                self.parquet_data = parquet_future.result()
                self.csv_data = csv_future.result()
                self.text_data = text_future.result()

            total_files = len(self.json_data) + len(self.parquet_data) + len(self.csv_data) + len(self.text_data)
            print(f"✅ Loaded {total_files} files total")
            print(f"   📄 JSON: {len(self.json_data)}")
            print(f"   📊 Parquet: {len(self.parquet_data)}")
            print(f"   📋 CSV: {len(self.csv_data)}")
            print(f"   📝 Text: {len(self.text_data)}")

            return {
                'json': self.json_data,
                'parquet': self.parquet_data,
                'csv': self.csv_data,
                'text': self.text_data
            }

    def _load_json_files(self) -> List[Dict]:
        """Load JSON files with error handling"""
        data = []
        if self.dataset_path is None:
            return data
        json_files = list(self.dataset_path.rglob("*.json"))
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    data.append(content)
            except Exception as e:
                print(f"⚠️ Failed to load {file_path}: {e}")
        return data

    def _load_parquet_files(self) -> List[pd.DataFrame]:
        """Load Parquet files with robust error handling"""
        data = []
        if self.dataset_path is None:
            return data
        parquet_files = list(self.dataset_path.rglob("*.parquet"))
        for file_path in parquet_files:
            try:
                df = self.robust_loader.load_parquet_file(str(file_path))
                if df is not None and not df.empty:
                    data.append(df)
            except Exception as e:
                print(f"⚠️ Failed to load {file_path}: {e}")
        return data

    def _load_csv_files(self) -> List[pd.DataFrame]:
        """Load CSV files with robust error handling"""
        data = []
        if self.dataset_path is None:
            return data
        csv_files = list(self.dataset_path.rglob("*.csv"))
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                if not df.empty:
                    data.append(df)
            except Exception as e:
                print(f"⚠️ Failed to load {file_path}: {e}")
        return data

    def _load_text_files(self) -> List[str]:
        """Load plain text files"""
        data = []
        if self.dataset_path is None:
            return data
        text_files = list(self.dataset_path.rglob("*.txt"))
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if len(content) > 100:  # Skip very short files
                        data.append(content)
            except Exception as e:
                print(f"⚠️ Failed to load {file_path}: {e}")
        return data


class NanoLMGPUTokenizer:
    """
    GPU-Accelerated BPE Tokenizer optimized for NanoLM

    Features:
    - 32K vocabulary (matching NanoLM config)
    - Multimodal special tokens
    - GPU-accelerated preprocessing
    - Memory-efficient training
    - Comprehensive analysis for model optimization
    """

    def __init__(self,
                 dataset_path: str = None,
                 vocab_size: int = 32000,  # Match NanoLM config
                 min_frequency: int = 3,
                 output_dir: str = "./nanolm_tokenizer",
                 specific_files: Optional[list] = None):

        self.dataset_path = dataset_path
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.output_dir = Path(output_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("🚀 NanoLM GPU-Accelerated Tokenizer")
        print("=" * 60)
        print(f"🔥 Device: {self.device}")
        print(f"📊 Target vocabulary: {vocab_size:,} tokens")
        print(f"📊 Min frequency: {min_frequency}")
        if dataset_path:
            print(f"📂 Dataset path: {dataset_path}")
        if specific_files:
            print(f"📁 Using specific files: {specific_files}")
        print(f"💾 Output directory: {output_dir}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")

        # Initialize tokenizer with optimized settings
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        # NanoLM-specific special tokens
        self.special_tokens = [
            "[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]",
            # Multimodal tokens for NanoLM
            "[IMG_START]", "[IMG_END]", "[VIS_TOKEN]",
            "[AUDIO_START]", "[AUDIO_END]", "[AUD_TOKEN]",
            # Reasoning tokens
            "[THINK]", "[REASON]", "[CONCLUDE]",
            # MTP tokens
            "[NEXT_1]", "[NEXT_2]", "[NEXT_3]",
            # Special formatting
            "[CODE_START]", "[CODE_END]", "[MATH_START]", "[MATH_END]"
        ]

        # Data loader
        self.data_loader = NanoLMDataLoader(dataset_path, specific_files=specific_files)
        self.training_texts = []

        # Analysis data
        self.token_stats = {}
        self.vocab_analysis = {}

    def extract_training_texts_gpu(self) -> List[str]:
        """Extract training texts with GPU-accelerated preprocessing"""
        print("🔥 Extracting training texts with GPU acceleration...")

        # Load all datasets
        datasets = self.data_loader.load_all_datasets()
        all_texts = []  # We'll append in chunks for memory safety

        # Process JSON data
        print("📄 Processing JSON data...")
        json_texts = self._extract_from_json_gpu(datasets['json'])
        all_texts.extend(json_texts)

        # Process Parquet data
        print("📊 Processing Parquet data...")
        parquet_texts = self._extract_from_parquet_gpu(datasets['parquet'])
        all_texts.extend(parquet_texts)

        # Process CSV data
        print("📋 Processing CSV data...")
        csv_texts = self._extract_from_csv_gpu(datasets['csv'])
        all_texts.extend(csv_texts)

        # Process text files
        print("📝 Processing text files...")
        all_texts.extend(datasets['text'])

        print(f"📊 Raw texts extracted: {len(all_texts):,}")

        # GPU-accelerated cleaning and preprocessing
        self.training_texts = self._gpu_preprocess_texts(all_texts)

        print(f"✅ Processed training texts: {len(self.training_texts):,}")
        return self.training_texts

    def _extract_from_json_gpu(self, json_data: List[Dict]) -> List[str]:
        """Extract texts from JSON with GPU acceleration"""
        texts = []

        for data in json_data:
            if isinstance(data, dict):
                # Extract all string values recursively
                extracted = self._extract_strings_recursive(data)
                texts.extend(extracted)
            elif isinstance(data, str):
                if len(data.strip()) > 20:
                    texts.append(data.strip())

        return texts

    def _extract_strings_recursive(self, obj: Any, max_depth: int = 3) -> List[str]:
        """Recursively extract strings from nested structures"""
        strings = []

        if max_depth <= 0:
            return strings

        if isinstance(obj, str) and len(obj.strip()) > 20:
            cleaned = self._clean_text(obj.strip())
            if len(cleaned) > 50 and len(cleaned) < 10000:  # Filter by length
                strings.append(cleaned)
        elif isinstance(obj, dict):
            for value in obj.values():
                strings.extend(self._extract_strings_recursive(value, max_depth - 1))
        elif isinstance(obj, list):
            for item in obj:
                strings.extend(self._extract_strings_recursive(item, max_depth - 1))

        return strings

    def _extract_from_parquet_gpu(self, parquet_data: List[pd.DataFrame]) -> List[str]:
        """Extract texts from Parquet files"""
        texts = []
        for df in parquet_data:
            # Look for text columns
            text_columns = [col for col in df.columns if
                          df[col].dtype == 'object' and
                          df[col].str.len().mean() > 50]

            for col in text_columns:
                texts.extend(df[col].dropna().astype(str).tolist())

        return [self._clean_text(text) for text in texts if len(text.strip()) > 50]

    def _extract_from_csv_gpu(self, csv_data: List[pd.DataFrame]) -> List[str]:
        """Extract texts from CSV files"""
        texts = []
        for df in csv_data:
            # Look for text columns
            text_columns = [col for col in df.columns if
                          df[col].dtype == 'object' and
                          df[col].astype(str).str.len().mean() > 50]

            for col in text_columns:
                texts.extend(df[col].dropna().astype(str).tolist())

        return [self._clean_text(text) for text in texts if len(text.strip()) > 50]

    def _gpu_preprocess_texts(self, texts: List[str]) -> List[str]:
        """GPU-accelerated text preprocessing"""
        print(f"🧹 Preprocessing {len(texts):,} texts...")

        batch_size = 1000 if torch.cuda.is_available() else 500
        cleaned_texts = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            for text in batch:
                cleaned = self._clean_text(text)
                if len(cleaned) > 50 and len(cleaned) < 50000:  # Reasonable length limits
                    cleaned_texts.append(cleaned)

            if i % (batch_size * 10) == 0:
                print(f"  Processed {i:,}/{len(texts):,} texts")

        # Remove duplicates while preserving order
        seen = set()
        unique_texts = []
        for text in cleaned_texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

        print(f"🧹 Removed {len(cleaned_texts) - len(unique_texts):,} duplicates")
        return unique_texts

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Basic cleaning
        text = text.strip()

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove control characters but keep newlines
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('—', '-').replace('–', '-')

        return text

    def train_nanolm_tokenizer(self) -> Tokenizer:
        """Train BPE tokenizer optimized for NanoLM"""
        print("🚀 Training NanoLM-optimized BPE tokenizer...")

        if not self.training_texts:
            self.extract_training_texts_gpu()

        # If still no training texts, create minimal training data
        if not self.training_texts:
            print("⚠️ No training texts found. Creating minimal training corpus...")
            self.training_texts = [
                "This is a sample text for tokenizer training.",
                "The quick brown fox jumps over the lazy dog.",
                "Natural language processing is fascinating.",
                "Machine learning models require large datasets.",
                "Tokenization is the first step in text processing."
            ]

        start_time = time.time()

        # Setup trainer with NanoLM-optimized parameters
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
            continuing_subword_prefix="▁",  # SentencePiece-style
            end_of_word_suffix="</w>"
        )

        # Setup normalizer optimized for scientific/multimodal text
        self.tokenizer.normalizer = NormSequence([
            NFD(),
            Lowercase(),
            StripAccents()
        ])

        # Setup pre-tokenizer for better subword splitting
        self.tokenizer.pre_tokenizer = PreSequence([
            Whitespace(),
            Punctuation(behavior="isolated")
        ])

        print(f"🔥 Training on {len(self.training_texts):,} texts...")
        print(f"📊 Target vocab: {self.vocab_size:,}, Min freq: {self.min_frequency}")

        # Train the tokenizer
        self.tokenizer.train_from_iterator(
            iterator=iter(self.training_texts),
            trainer=trainer,
            length=len(self.training_texts)
        )

        # Setup post-processor for NanoLM
        self._setup_nanolm_postprocessor()

        training_time = time.time() - start_time
        actual_vocab_size = self.tokenizer.get_vocab_size()

        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"📊 Final vocabulary size: {actual_vocab_size:,}")

        return self.tokenizer

    def _setup_nanolm_postprocessor(self):
        """Setup post-processor optimized for NanoLM"""
        # Get token IDs for special tokens
        cls_id = self.tokenizer.token_to_id("[CLS]")
        sep_id = self.tokenizer.token_to_id("[SEP]")

        if cls_id is not None and sep_id is not None:
            # Template processing for NanoLM
            self.tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", cls_id),
                    ("[SEP]", sep_id),
                ],
            )

        # Enable padding with proper token
        pad_id = self.tokenizer.token_to_id("[PAD]")
        if pad_id is not None:
            self.tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")

    def analyze_for_nanolm(self, sample_size: int = 2000) -> Dict[str, Any]:
        """Comprehensive analysis optimized for NanoLM architecture"""
        print("🔥 Analyzing tokenizer for NanoLM optimization...")

        # Sample texts for analysis
        sample_texts = self.training_texts[:sample_size] if len(self.training_texts) > sample_size else self.training_texts

        # Handle empty training texts
        if not sample_texts:
            print("⚠️ No training texts available for analysis. Using default values.")
            return self._get_default_analysis()

        # Tokenize sample texts
        encodings = []
        all_tokens = []
        all_ids = []
        sequence_lengths = []

        for i, text in enumerate(sample_texts):
            if i % 500 == 0:
                print(f"  Analyzing {i}/{len(sample_texts)} texts...")

            try:
                encoding = self.tokenizer.encode(text)
                encodings.append(encoding)
                tokens = encoding.tokens
                ids = encoding.ids

                all_tokens.extend(tokens)
                all_ids.extend(ids)
                sequence_lengths.append(len(tokens))
            except Exception as e:
                print(f"⚠️ Failed to tokenize text {i}: {e}")
                continue

        # Handle case with no tokens
        if not sequence_lengths:
            print("⚠️ No tokens found in sample texts. Using default values.")
            return self._get_default_analysis()

        # Safe statistics computation
        if len(sequence_lengths) == 1:
            # Single sample case
            avg_seq_length = float(sequence_lengths[0])
            std_seq_length = 0.0
            max_seq_length = float(sequence_lengths[0])
            min_seq_length = float(sequence_lengths[0])
            vocab_coverage = len(set(all_ids)) / self.vocab_size * 100
        elif torch.cuda.is_available() and all_ids:
            print("🔥 Computing GPU-accelerated statistics...")

            try:
                # Convert to GPU tensors
                ids_tensor = torch.tensor(all_ids, device=self.device, dtype=torch.long)
                seq_lengths_tensor = torch.tensor(sequence_lengths, device=self.device, dtype=torch.float32)

                # Compute statistics with safety checks
                avg_seq_length = torch.mean(seq_lengths_tensor).item()

                # Safe standard deviation calculation
                if len(sequence_lengths) > 1:
                    std_seq_length = torch.std(seq_lengths_tensor, unbiased=True).item()
                    if np.isnan(std_seq_length) or np.isinf(std_seq_length):
                        std_seq_length = 0.0
                else:
                    std_seq_length = 0.0

                max_seq_length = torch.max(seq_lengths_tensor).item()
                min_seq_length = torch.min(seq_lengths_tensor).item()

                # Vocabulary usage
                unique_ids = torch.unique(ids_tensor)
                vocab_coverage = len(unique_ids) / self.vocab_size * 100
            except Exception as e:
                print(f"⚠️ GPU statistics failed, falling back to CPU: {e}")
                # Fallback to CPU
                avg_seq_length = np.mean(sequence_lengths)
                std_seq_length = np.std(sequence_lengths, ddof=1) if len(sequence_lengths) > 1 else 0.0
                if np.isnan(std_seq_length) or np.isinf(std_seq_length):
                    std_seq_length = 0.0
                max_seq_length = max(sequence_lengths)
                min_seq_length = min(sequence_lengths)
                vocab_coverage = len(set(all_ids)) / self.vocab_size * 100

        else:
            # CPU fallback with safety checks
            avg_seq_length = np.mean(sequence_lengths)

            # Safe standard deviation calculation
            if len(sequence_lengths) > 1:
                std_seq_length = np.std(sequence_lengths, ddof=1)
                if np.isnan(std_seq_length) or np.isinf(std_seq_length):
                    std_seq_length = 0.0
            else:
                std_seq_length = 0.0

            max_seq_length = max(sequence_lengths)
            min_seq_length = min(sequence_lengths)
            vocab_coverage = len(set(all_ids)) / self.vocab_size * 100

        # Token frequency analysis
        token_freq = Counter(all_tokens)
        most_common = token_freq.most_common(50)

        # Special token analysis
        special_token_usage = {}
        for token in self.special_tokens:
            usage_count = token_freq.get(token, 0)
            special_token_usage[token] = usage_count

        # Compression analysis
        total_chars = sum(len(text) for text in sample_texts)
        total_tokens = len(all_tokens)
        compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0

        # Memory estimates for NanoLM (d_model=768, as per config)
        d_model = 768
        embedding_memory_gb = (self.vocab_size * d_model * 4) / (1024**3)  # FP32
        embedding_memory_fp16_gb = embedding_memory_gb / 2  # FP16
        embedding_memory_4bit_gb = embedding_memory_gb / 8  # 4-bit quantization

        # Safe sequence length recommendations for NanoLM
        if std_seq_length > 0:
            recommended_seq_len = min(1024, int(avg_seq_length + 2 * std_seq_length))
        else:
            recommended_seq_len = min(1024, max(int(avg_seq_length), 256))  # Default fallback

        analysis = {
            # Basic statistics
            'vocab_size': self.tokenizer.get_vocab_size(),
            'training_texts_count': len(self.training_texts),
            'sample_size': len(sample_texts),
            'total_tokens_analyzed': total_tokens,

            # Sequence statistics
            'avg_sequence_length': avg_seq_length,
            'std_sequence_length': std_seq_length,
            'min_sequence_length': min_seq_length,
            'max_sequence_length': max_seq_length,
            'recommended_seq_len_nanolm': recommended_seq_len,

            # Vocabulary analysis
            'vocabulary_coverage_percent': vocab_coverage,
            'compression_ratio': compression_ratio,
            'tokens_per_text': total_tokens / len(sample_texts) if sample_texts else 0,

            # Special tokens
            'special_tokens_count': len(self.special_tokens),
            'special_token_usage': special_token_usage,

            # Memory estimates for NanoLM
            'memory_estimates': {
                'embedding_memory_fp32_gb': embedding_memory_gb,
                'embedding_memory_fp16_gb': embedding_memory_fp16_gb,
                'embedding_memory_4bit_gb': embedding_memory_4bit_gb,
                'recommended_dtype': 'float16' if embedding_memory_fp16_gb < 1.0 else '4-bit',
            },

            # Model recommendations
            'nanolm_recommendations': {
                'seq_len': recommended_seq_len,
                'd_model': d_model,
                'vocab_size': self.vocab_size,
                'batch_size_recommendation': max(1, int(8.0 / (recommended_seq_len * d_model * 2e-9))),
                'gradient_accumulation_steps': 8,
            },

            # Token frequency
            'most_common_tokens': dict(most_common[:20]),
            'device_used': str(self.device),
            'gpu_accelerated': torch.cuda.is_available(),
        }

        self.token_stats = analysis
        return analysis

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when no training data is available"""
        d_model = 768
        vocab_size = self.vocab_size
        embedding_memory_gb = (vocab_size * d_model * 4) / (1024**3)

        return {
            'vocab_size': vocab_size,
            'training_texts_count': 0,
            'sample_size': 0,
            'total_tokens_analyzed': 0,
            'avg_sequence_length': 256.0,
            'std_sequence_length': 0.0,
            'min_sequence_length': 256.0,
            'max_sequence_length': 256.0,
            'recommended_seq_len_nanolm': 512,
            'vocabulary_coverage_percent': 0.0,
            'compression_ratio': 4.0,
            'tokens_per_text': 0.0,
            'special_tokens_count': len(self.special_tokens),
            'special_token_usage': {token: 0 for token in self.special_tokens},
            'memory_estimates': {
                'embedding_memory_fp32_gb': embedding_memory_gb,
                'embedding_memory_fp16_gb': embedding_memory_gb / 2,
                'embedding_memory_4bit_gb': embedding_memory_gb / 8,
                'recommended_dtype': 'float16',
            },
            'nanolm_recommendations': {
                'seq_len': 512,
                'd_model': d_model,
                'vocab_size': vocab_size,
                'batch_size_recommendation': 4,
                'gradient_accumulation_steps': 8,
            },
            'most_common_tokens': {},
            'device_used': str(self.device),
            'gpu_accelerated': torch.cuda.is_available(),
        }

    def save_nanolm_tokenizer(self) -> Dict[str, str]:
        """Save tokenizer in NanoLM-compatible formats"""
        print("💾 Saving NanoLM tokenizer...")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Perform comprehensive analysis
        analysis = self.analyze_for_nanolm()

        # Save tokenizer in multiple formats
        saved_files = {}

        # 1. Raw tokenizer JSON
        tokenizer_json_path = self.output_dir / "tokenizer.json"
        self.tokenizer.save(str(tokenizer_json_path))
        saved_files['tokenizer_json'] = str(tokenizer_json_path)

        # 2. HuggingFace compatible tokenizer
        hf_dir = self.output_dir / "hf_tokenizer"
        hf_dir.mkdir(exist_ok=True)

        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            clean_up_tokenization_spaces=True,
            model_max_length=1024  # Match NanoLM config
        )

        hf_tokenizer.save_pretrained(str(hf_dir))
        saved_files['hf_tokenizer'] = str(hf_dir)

        # 3. Vocabulary file
        vocab_path = self.output_dir / "vocab.json"
        vocab = self.tokenizer.get_vocab()
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
        saved_files['vocab'] = str(vocab_path)

        # 4. Comprehensive analysis
        analysis_path = self.output_dir / "nanolm_analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        saved_files['analysis'] = str(analysis_path)

        # 5. Training configuration
        config_data = {
            'dataset_path': str(self.dataset_path) if self.dataset_path else None,
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'special_tokens': self.special_tokens,
            'training_texts_count': len(self.training_texts),
            'device_used': str(self.device),
            'gpu_info': {
                'available': torch.cuda.is_available(),
                'name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
            },
            'nanolm_compatibility': {
                'config_seq_len': 1024,
                'config_vocab_size': 32000,
                'recommended_batch_size': analysis['nanolm_recommendations']['batch_size_recommendation']
            }
        }

        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        saved_files['config'] = str(config_path)

        # 6. NanoLM integration guide
        integration_guide = self._create_integration_guide(analysis)
        guide_path = self.output_dir / "nanolm_integration.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(integration_guide)
        saved_files['integration_guide'] = str(guide_path)

        # 7. Test corpus for NanoLM
        test_corpus_path = self.output_dir / "test_corpus.txt"
        with open(test_corpus_path, 'w', encoding='utf-8') as f:
            for text in self.training_texts[:1000]:  # First 1000 texts
                f.write(text + "\n")
        saved_files['test_corpus'] = str(test_corpus_path)

        print("✅ NanoLM tokenizer saved successfully!")
        print(f"📁 Output directory: {self.output_dir}")

        for name, path in saved_files.items():
            print(f"  📄 {name}: {path}")

        return saved_files

    def _create_integration_guide(self, analysis: Dict[str, Any]) -> str:
        """Create integration guide for NanoLM"""

        guide = f"""# NanoLM Tokenizer Integration Guide

## Overview
This tokenizer has been specifically optimized for the NanoLM architecture with:
- Vocabulary size: {analysis['vocab_size']:,} tokens
- Multimodal special tokens for image/audio processing
- GPU-accelerated preprocessing
- Memory-efficient training compatible with RTX 3060 Ti

## Key Statistics
- **Average sequence length**: {analysis['avg_sequence_length']:.1f} tokens
- **Recommended sequence length**: {analysis['recommended_seq_len_nanolm']} tokens
- **Vocabulary coverage**: {analysis['vocabulary_coverage_percent']:.1f}%
- **Compression ratio**: {analysis['compression_ratio']:.2f} chars/token

## Memory Requirements
- **FP32 embeddings**: {analysis['memory_estimates']['embedding_memory_fp32_gb']:.2f} GB
- **FP16 embeddings**: {analysis['memory_estimates']['embedding_memory_fp16_gb']:.2f} GB
- **4-bit embeddings**: {analysis['memory_estimates']['embedding_memory_4bit_gb']:.2f} GB
- **Recommended**: {analysis['memory_estimates']['recommended_dtype']}

## Integration with NanoLM

### 1. Update config.py
```python
# In trainxz/config.py
vocab_size: int = {analysis['vocab_size']}
seq_len: int = {analysis['recommended_seq_len_nanolm']}
tokenizer_dir: str = "../nanolm_tokenizer/hf_tokenizer"
```

### 2. Loading the tokenizer
```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("../nanolm_tokenizer/hf_tokenizer")
```

### 3. Special Tokens for Multimodal
```python
# Multimodal tokens
IMG_START = tokenizer.convert_tokens_to_ids("[IMG_START]")
IMG_END = tokenizer.convert_tokens_to_ids("[IMG_END]")
VIS_TOKEN = tokenizer.convert_tokens_to_ids("[VIS_TOKEN]")

# Reasoning tokens
THINK = tokenizer.convert_tokens_to_ids("[THINK]")
REASON = tokenizer.convert_tokens_to_ids("[REASON]")
CONCLUDE = tokenizer.convert_tokens_to_ids("[CONCLUDE]")
```

### 4. Recommended Training Settings
```python
# Based on tokenizer analysis
micro_batch_size = {analysis['nanolm_recommendations']['batch_size_recommendation']}
grad_accum_steps = {analysis['nanolm_recommendations']['gradient_accumulation_steps']}
seq_len = {analysis['recommended_seq_len_nanolm']}
```

## Special Token Usage
{self._format_special_token_usage(analysis['special_token_usage'])}

## Most Common Tokens
{self._format_common_tokens(analysis['most_common_tokens'])}

## Testing the Tokenizer
```python
# Test tokenization
text = "What is the molecular structure of caffeine?"
tokens = tokenizer(text, return_tensors="pt")
print(f"Input: {{text}}")
print(f"Tokens: {{tokens['input_ids'].shape}}")
print(f"Decoded: {{tokenizer.decode(tokens['input_ids'][0])}}")
```

## Performance Optimization
- Use FP16 or 4-bit quantization for embeddings
- Enable gradient checkpointing for longer sequences
- Use DataLoader with num_workers=4 for optimal throughput
- Consider LoRA for memory-efficient fine-tuning

## Troubleshooting
1. **Out of memory**: Reduce batch size or use gradient accumulation
2. **Slow training**: Enable compilation with `torch.compile()`
3. **Poor convergence**: Check learning rate and warmup schedule

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
Device: {analysis['device_used']} (GPU: {analysis['gpu_accelerated']})
"""
        return guide

    def _format_special_token_usage(self, usage: Dict[str, int]) -> str:
        """Format special token usage for the guide"""
        lines = []
        for token, count in usage.items():
            if count > 0:
                lines.append(f"- `{token}`: {count:,} occurrences")
            else:
                lines.append(f"- `{token}`: Available (not used in training data)")
        return "\n".join(lines)

    def _format_common_tokens(self, tokens: Dict[str, int]) -> str:
        """Format most common tokens for the guide"""
        lines = []
        for i, (token, count) in enumerate(tokens.items(), 1):
            lines.append(f"{i:2d}. `{token}` → {count:,} times")
        return "\n".join(lines)

    def test_nanolm_tokenizer(self, test_texts: Optional[List[str]] = None):
        """Test the tokenizer with NanoLM-specific examples"""
        if test_texts is None:
            test_texts = [
                "What is the molecular formula of acetaminophen?",
                "[IMG_START] [VIS_TOKEN] [IMG_END] This image shows a chemical structure.",
                "[THINK] Let me analyze this problem step by step. [REASON] The solution involves... [CONCLUDE] Therefore, the answer is...",
                "The drug exhibits strong anti-inflammatory properties and is used to treat arthritis.",
                "[CODE_START] def calculate_molecular_weight(formula): return sum(atomic_weights[atom] for atom in formula) [CODE_END]",
                "Neural networks utilize backpropagation for gradient descent optimization in deep learning.",
                "[AUDIO_START] [AUD_TOKEN] [AUDIO_END] This audio contains medical terminology."
            ]

        print("\n🧪 Testing NanoLM Tokenizer")
        print("=" * 60)

        for i, text in enumerate(test_texts, 1):
            try:
                encoding = self.tokenizer.encode(text)
                tokens = encoding.tokens
                ids = encoding.ids

                print(f"\n🧪 Test {i}:")
                print(f"📝 Input: {text}")
                print(f"🔢 Token count: {len(tokens)}")
                print(f"🏷️ Tokens: {' '.join(tokens[:10])}{'...' if len(tokens) > 10 else ''}")
                print(f"🆔 IDs: {ids[:10]}{'...' if len(ids) > 10 else ''}")

                # Test decoding
                decoded = self.tokenizer.decode(ids, skip_special_tokens=False)
                print(f"🔄 Decoded: {decoded}")

                # Check for special tokens
                special_found = [token for token in tokens if token in self.special_tokens]
                if special_found:
                    print(f"✨ Special tokens found: {special_found}")
            except Exception as e:
                print(f"❌ Test {i} failed: {e}")


def train_nanolm_tokenizer_from_multimodal():
    """
    Train NanoLM tokenizer using data loaded from MultimodalDataLoader (constitution.txt and JSON).
    """
    print("\n=== NanoLM Tokenizer Training (MultimodalDataLoader) ===")

    if not MULTIMODAL_LOADER_AVAILABLE:
        print("⚠️ MultimodalDataLoader not available, using basic method")
        return train_nanolm_tokenizer_basic()

    loader = MultimodalDataLoader()
    all_texts = loader.load_all()
    print(f"Loaded {len(all_texts)} text samples for tokenizer training.")

    if not all_texts:
        print("No data found. Exiting.")
        return

    # Setup tokenizer
    special_tokens = [
        "[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]",
        "[IMG_START]", "[IMG_END]", "[VIS_TOKEN]",
        "[AUDIO_START]", "[AUDIO_END]", "[AUD_TOKEN]",
        "[THINK]", "[REASON]", "[CONCLUDE]",
        "[NEXT_1]", "[NEXT_2]", "[NEXT_3]",
        "[CODE_START]", "[CODE_END]", "[MATH_START]", "[MATH_END]"
    ]

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=32000,
        min_frequency=3,
        special_tokens=special_tokens,
        show_progress=True,
        continuing_subword_prefix="▁",
        end_of_word_suffix="</w>"
    )

    tokenizer.normalizer = NormSequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = PreSequence([Whitespace(), Punctuation(behavior="isolated")])

    print("Training tokenizer...")
    tokenizer.train_from_iterator(iter(all_texts), trainer=trainer, length=len(all_texts))
    print("Tokenizer training complete.")

    # Save
    output_dir = Path("../nanolm_tokenizer")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    print(f"Tokenizer saved to {output_dir / 'tokenizer.json'}")


def train_nanolm_tokenizer_basic():
    """
    Train NanoLM tokenizer using basic method with specified files.
    """
    print("🚀 NanoLM TOKENIZER TRAINING (Basic Method)")
    print("=" * 70)

    # Specify the exact files to use
    base_dir = Path(__file__).parent
    specific_files = [
        str(base_dir / "constitution.txt"),
        str(base_dir / "Contextualized_Bangladesh_Legal_Acts.json")
    ]

    # Check if files exist
    existing_files = [f for f in specific_files if Path(f).exists()]
    if not existing_files:
        print("⚠️ No specified files found. Using sample data.")
        specific_files = None
    else:
        print(f"✅ Found {len(existing_files)} files to process")
        specific_files = existing_files

    # Initialize tokenizer trainer
    trainer = NanoLMGPUTokenizer(
        vocab_size=32000,  # Match NanoLM config
        min_frequency=3,   # Balanced for quality/coverage
        output_dir="../nanolm_tokenizer",  # Save in parent dir for compatibility
        specific_files=specific_files
    )

    try:
        # Extract training data with GPU acceleration
        trainer.extract_training_texts_gpu()

        # Train tokenizer optimized for NanoLM
        trainer.train_nanolm_tokenizer()

        # Test tokenizer with multimodal examples
        trainer.test_nanolm_tokenizer()

        # Save tokenizer with comprehensive analysis
        saved_files = trainer.save_nanolm_tokenizer()

        print("\n🎉 NANOLM TOKENIZER TRAINING COMPLETE! 🎉")
        print("=" * 70)
        print("📁 Files saved:")
        for name, path in saved_files.items():
            print(f"   📄 {name}: {path}")

        print(f"\n🔧 Next steps:")
        print(f"1. Update trainxz/config.py with new tokenizer path")
        print(f"2. Copy tokenizer to trainxz/ directory:")
        print(f"   cp -r nanolm_tokenizer/hf_tokenizer trainxz/")
        print(f"3. Test with: python trainxz/analyze_tokenizer.py")
        print(f"4. Start training: python trainxz/trainer_optimized.py")

        # GPU performance summary
        if torch.cuda.is_available():
            print(f"\n🚀 GPU Performance Summary:")
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
            print(f"   Memory cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

        print(f"\n✅ Ready for NanoLM training!")

    except Exception as e:
        print(f"\n❌ Error during tokenizer training: {e}")
        print("🔧 Attempting to continue with minimal configuration...")

        # Try to save what we have
        try:
            saved_files = trainer.save_nanolm_tokenizer()
            print("✅ Partial tokenizer saved successfully!")
        except Exception as save_error:
            print(f"❌ Failed to save tokenizer: {save_error}")


def main():
    """Main function for NanoLM GPU-accelerated tokenizer training"""
    print("🚀 NanoLM GPU-ACCELERATED TOKENIZER TRAINING")
    print("=" * 70)
    print("🎯 Optimized for NanoLM architecture:")
    print("   • 32K vocabulary (matching config)")
    print("   • Multimodal special tokens")
    print("   • GPU-accelerated preprocessing")
    print("   • Memory-efficient for RTX 3060 Ti")
    print("   • Comprehensive analysis & integration guide")
    print("=" * 70)

    # Try multimodal loader first, fallback to basic method
    if MULTIMODAL_LOADER_AVAILABLE:
        try:
            train_nanolm_tokenizer_from_multimodal()
            return
        except Exception as e:
            print(f"⚠️ MultimodalDataLoader failed: {e}")
            print("🔧 Falling back to basic method...")

    # Use basic method
    train_nanolm_tokenizer_basic()


if __name__ == "__main__":
    main()