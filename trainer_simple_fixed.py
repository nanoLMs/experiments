#!/usr/bin/env python3
"""
Simple, robust trainer without complex quantization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import os
import json
from datetime import datetime

# Import our models
from model_moe import NanoMoEModel
from data_loader import create_dataloader
from loss_tracker import LossTracker
from config import TrainConfig

class SimpleTrainer:
    """Simple, robust trainer without complex quantization"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize loss tracker
        self.loss_tracker = LossTracker()

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')

    def _create_model(self):
        """Create a simple NanoMoE model without complex quantization"""
        # Create a simple config
        config = TrainConfig()
        # Disable complex features for stability
        config.use_quantization = False
        config.moe_every = 0  # Disable MoE for now
        config.use_hrm = False  # Disable HRM for now
        config.enable_reasoning = False  # Disable reasoning for now
        config.n_layers = 8  # Smaller model for testing
        config.d_model = 256  # Smaller model
        config.n_heads = 4   # Smaller model

        model = NanoMoEModel(config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"Model created:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f}MB")

        return model

    def _create_optimizer(self):
        """Create optimizer"""
        # Separate parameters for different learning rates
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'ln' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                'params': decay_params,
                'weight_decay': 0.01,
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0,
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=3e-4,
            betas=(0.9, 0.95),
            eps=1e-8
        )

        return optimizer

    def compute_loss(self, batch):
        """Compute loss for a batch"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        outputs = self.model(input_ids)
        logits = outputs['logits']
        aux_loss = outputs.get('aux_loss', 0.0)

        # Compute cross-entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Total loss
        total_loss = lm_loss + 0.01 * aux_loss

        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'aux_loss': aux_loss
        }

    def train_step(self, batch):
        """Single training step"""
        self.model.train()

        # Compute loss
        loss_dict = self.compute_loss(batch)
        loss = loss_dict['total_loss']

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Optimizer step
        self.optimizer.step()

        # Update tracking
        self.loss_tracker.update(loss_dict)
        self.global_step += 1

        return loss_dict

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        progress_bar = tqdm(dataloader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                loss_dict = self.train_step(batch)
                loss = loss_dict['total_loss'].item()
                total_loss += loss

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                    'step': self.global_step
                })

                # Log periodically
                if self.global_step % 100 == 0:
                    self.logger.info(f"Step {self.global_step}: Loss = {loss:.4f}")

                # Save checkpoint periodically
                if self.global_step % 1000 == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            except Exception as e:
                self.logger.error(f"Error in training step {batch_idx}: {e}")
                continue

        return total_loss / num_batches

    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                try:
                    loss_dict = self.compute_loss(batch)
                    total_loss += loss_dict['total_loss'].item()
                    num_batches += 1
                except Exception as e:
                    self.logger.error(f"Error in validation: {e}")
                    continue

        if num_batches > 0:
            avg_loss = total_loss / num_batches
        else:
            avg_loss = float('inf')

        return avg_loss

    def train(self, train_dataloader, val_dataloader=None, num_epochs=10):
        """Main training loop"""
        self.logger.info("Starting training...")

        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(train_dataloader)

            # Validate
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                self.logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best_model.pt")
            else:
                self.logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        self.logger.info("Training completed!")

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        os.makedirs("checkpoints", exist_ok=True)
        filepath = os.path.join("checkpoints", filename)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': self.config
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']

        self.logger.info(f"Checkpoint loaded: {filepath}")


def main():
    """Main training function"""
    print("🚀 Starting Simple NanoLM Training")

    # Create config
    config = TrainConfig()
    config.use_quantization = False
    config.moe_every = 0
    config.use_hrm = False
    config.enable_reasoning = False
    config.n_layers = 8
    config.d_model = 256
    config.n_heads = 4
    config.micro_batch_size = 8
    config.num_epochs = 3  # Shorter for testing
    config.train_corpus = "data/constitution.txt"

    print(f"Device: cuda" if torch.cuda.is_available() else "cpu")

    # Create trainer
    trainer = SimpleTrainer(config)

    # Load tokenizer
    print("📚 Loading tokenizer...")
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("nanolm_tokenizer")
    except:
        print("⚠️ Could not load tokenizer, creating simple one...")
        # Create a simple tokenizer for testing
        class SimpleTokenizer:
            def __init__(self):
                self.vocab_size = 29086
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.bos_token_id = 2

            def __call__(self, text, add_special_tokens=True):
                # Simple character-level encoding for testing
                tokens = [ord(c) % self.vocab_size for c in text[:config.seq_len]]
                return {'input_ids': tokens}

            def encode(self, text):
                return [ord(c) % self.vocab_size for c in text[:config.seq_len]]

            def decode(self, tokens):
                return ''.join([chr(t) for t in tokens if t > 0])

        tokenizer = SimpleTokenizer()

    # Create dataloader
    print("📚 Creating dataloader...")
    train_dataloader = create_dataloader(config, tokenizer)

    print(f"✅ Dataloader created: {len(train_dataloader)} batches")

    # Start training
    trainer.train(train_dataloader, num_epochs=config.num_epochs)

    # Generate loss plots
    trainer.loss_tracker.plot_losses("training_results.png")

    print("🎉 Training completed successfully!")


if __name__ == "__main__":
    main()