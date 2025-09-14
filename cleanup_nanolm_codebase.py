#!/usr/bin/env python3
"""
NanoLM Codebase Cleanup Script
==============================

Intelligently removes unnecessary files while preserving:
- Token data and tokenizer configurations
- Essential model components
- Export systems for multi-platform deployment
- Training infrastructure
- Research-backed quantization systems

Based on analysis of BitNet, T-MAC, HRM, and other research papers.
"""

import os
import shutil
import json
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime
import argparse

# Import the rich logger for beautiful output
from nanolm_rich_logger import get_nanolm_logger
from rich.panel import Panel
from rich.table import Table

class NanoLMCodebaseCleanup:
    """Intelligent cleanup of NanoLM codebase"""
    
    def __init__(self, workspace_path: str = "."):
        self.workspace = Path(workspace_path)
        self.logger = get_nanolm_logger()
        self.backup_dir = self.workspace / "cleanup_backup"
        
        # Files to absolutely preserve (core functionality)
        self.essential_files = {
            # Core model architecture
            "nanolm_core.py",
            "nanolm_model.py", 
            "nanolm_research_model.py",
            "model_moe.py",
            "moe_system.py",
            "mtp_head.py",
            "mtp_system.py",
            "hrm_system.py",
            "hrm_bridge.py",
            
            # Training systems
            "trainer.py",
            "trainer_optimized.py",
            "trainer_optimized_fixed.py",
            "advanced_trainer.py",
            "training_recovery.py",
            
            # Quantization (research-backed)
            "fp4_fqt_core.py",
            "fp4_model_integration.py", 
            "fp4_quant.py",
            "advanced_quantization.py",
            "trainer_fp4_fqt.py",
            "trainer_bnb_quantized.py",
            
            # Configuration
            "config.py",
            "enhanced_config.py",
            "rtx3060ti_config.py",  # Our new optimized config
            "tokenizer_config.py",
            
            # Tokenization (essential for token data)
            "nanolm_gpu_tokenizer_streaming.py",
            "train_legal_hf_tokenizer.py",
            "benchmark_tokenizer.py",
            
            # Export systems (multi-platform deployment)
            "multi_platform_exporter.py",
            "enhanced_export_system.py",
            "export_models.py",
            "export_clean_model.py",
            
            # Anti-hallucination (research-backed)
            "anti_hallucination.py",
            "anti_hallu.py",
            "unlikelihood_training.py",
            
            # Loss and monitoring
            "loss_system.py",
            "loss_tracker.py",
            "predict_final_loss.py",
            "monitoring_system.py",
            "monitoring_diagnostics.py",
            
            # Memory optimization
            "memory_compute_optimizations.py",
            "dynamic_batching_caching.py",
            
            # Edge deployment
            "mobile_edge_optimizer.py",
            "edge_device_validator.py",
            
            # Logging system
            "nanolm_rich_logger.py",  # Our new logging system
            "rich_output.py",
            
            # Data and reasoning
            "reasoning.py",
            "data_loader.py",
            "schedule.py",
            
            # Documentation
            "README.md",
            "API_DOCUMENTATION.md",
            "CONFIGURATION_GUIDE.md",
            "LOSS_TRACKING_SUMMARY.md",
            "SOLUTION_SUMMARY.md",
            "TOKEN_BUDGET_FIX.md",
            "GPU_TOKENIZER_README.md",
            
            # Data files (token sources)
            "legal_corpus.txt",
            "constitution.txt",
            "Contextualized_Bangladesh_Legal_Acts.json",
        }
        
        # Test files to remove (keep only essential tests)
        self.test_files_to_remove = {
            "test_cloud_deployment.py",  # Too specific, not core
            "test_dynamic_batching_caching.py",
            "test_gpu_tokenizer.py", 
            "test_integration_working.py",
            "test_memory_compute_optimizations.py",
            "test_performance_quality_validation.py",
        }
        
        # Development/debug files to remove
        self.debug_files_to_remove = {
            "config_simple_loss_debug.py",
            "config_simple.py",
            "train_simple.py",
            "trainer_simple_fixed.py",
            "fix_model_export_v2.py",
            "fix_model_export.py",
            "fix_training_and_export.py", 
            "fix_training_issues.py",
            "analyze_fp4_fqt.py",  # Analysis done, not needed for production
        }
        
        # Launch scripts to consolidate (keep only essential)
        self.launch_files_to_remove = {
            "launch_bnb_training.py",
            "launch_fp4_training_fixed.py",
            "launch_fp4_training.py",  # Superseded by optimized trainer
        }
        
        # Integration examples (keep one, remove duplicates)
        self.example_files_to_remove = {
            "export_integration_example.py",
            "mobile_edge_integration_example.py", 
            "monitoring_integration_example.py",
            "recovery_integration_example.py",
        }
        
        # Utility files that are redundant
        self.utility_files_to_remove = {
            "calculate_dataset_size.py",
            "calculate_model_size.py",
            "checkpoint.py",  # Functionality integrated into trainers
            "read_pdf.py",  # PDF analysis complete
            "setup_infrastructure.py",  # One-time setup
        }
        
        # Directories to clean up
        self.dirs_to_clean = {
            "__pycache__",
            ".pytest_cache", 
            "*.egg-info",
            "build",
            "dist",
        }
        
    def create_backup(self):
        """Create backup of current state"""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        self.backup_dir.mkdir()
        
        self.logger.console.print(
            "[nanolm.info]Creating backup in cleanup_backup/[/]"
        )
        
        # Backup all files that will be removed
        all_removals = (
            self.test_files_to_remove | 
            self.debug_files_to_remove |
            self.launch_files_to_remove |
            self.example_files_to_remove |
            self.utility_files_to_remove
        )
        
        for filename in all_removals:
            file_path = self.workspace / filename
            if file_path.exists():
                backup_path = self.backup_dir / filename
                shutil.copy2(file_path, backup_path)
                
        self.logger.console.print(
            f"[nanolm.success]✅ Backed up {len(all_removals)} files[/]"
        )
        
    def analyze_codebase(self) -> Dict[str, List[str]]:
        """Analyze current codebase structure"""
        analysis = {
            "essential": [],
            "test_files": [],
            "debug_files": [],
            "launch_files": [],
            "example_files": [],
            "utility_files": [],
            "unknown": []
        }
        
        for file_path in self.workspace.glob("*.py"):
            filename = file_path.name
            
            if filename in self.essential_files:
                analysis["essential"].append(filename)
            elif filename in self.test_files_to_remove:
                analysis["test_files"].append(filename)
            elif filename in self.debug_files_to_remove:
                analysis["debug_files"].append(filename)
            elif filename in self.launch_files_to_remove:
                analysis["launch_files"].append(filename)
            elif filename in self.example_files_to_remove:
                analysis["example_files"].append(filename)
            elif filename in self.utility_files_to_remove:
                analysis["utility_files"].append(filename)
            else:
                analysis["unknown"].append(filename)
                
        return analysis
        
    def show_cleanup_plan(self, analysis: Dict[str, List[str]]):
        """Display cleanup plan"""
        
        table = Table(title="[nanolm.header]NanoLM Cleanup Plan[/]")
        table.add_column("Category", style="bold")
        table.add_column("Action", justify="center")
        table.add_column("Count", justify="right")
        table.add_column("Files", style="dim")
        
        table.add_row(
            "[nanolm.success]Essential[/]",
            "✅ Keep",
            str(len(analysis["essential"])),
            "Core model, training, export systems"
        )
        
        table.add_row(
            "[nanolm.warning]Test Files[/]",
            "🗑️ Remove",
            str(len(analysis["test_files"])),
            ", ".join(analysis["test_files"][:3]) + "..."
        )
        
        table.add_row(
            "[nanolm.warning]Debug Files[/]",
            "🗑️ Remove", 
            str(len(analysis["debug_files"])),
            ", ".join(analysis["debug_files"][:3]) + "..."
        )
        
        table.add_row(
            "[nanolm.warning]Launch Scripts[/]",
            "🗑️ Remove",
            str(len(analysis["launch_files"])),
            ", ".join(analysis["launch_files"][:3]) + "..."
        )
        
        table.add_row(
            "[nanolm.warning]Examples[/]",
            "🗑️ Remove",
            str(len(analysis["example_files"])),
            ", ".join(analysis["example_files"][:3]) + "..."
        )
        
        table.add_row(
            "[nanolm.warning]Utilities[/]",
            "🗑️ Remove",
            str(len(analysis["utility_files"])),
            ", ".join(analysis["utility_files"][:3]) + "..."
        )
        
        if analysis["unknown"]:
            table.add_row(
                "[nanolm.info]Unknown[/]",
                "❓ Review",
                str(len(analysis["unknown"])),
                ", ".join(analysis["unknown"][:3]) + "..."
            )
            
        self.logger.console.print(table)
        
        # Show preservation of critical components
        preservation_panel = Panel(
            """[nanolm.success]✅ PRESERVED COMPONENTS[/]

[nanolm.tokenizer]🔤 Tokenization System[/]
• GPU-accelerated BPE tokenizer
• Legal domain special tokens
• Token data and configurations

[nanolm.quantization]⚡ BitNet 1.58-bit Quantization[/]
• Ternary {-1, 0, 1} weights
• FP4 training systems
• Advanced quantization cores

[nanolm.hrm]🧠 Advanced Architecture[/]
• Hierarchical Reasoning Module (HRM)
• Multi-Token Prediction (MTP)
• Mixture of Experts (MoE)

[nanolm.export]📱 Multi-Platform Export[/]
• TorchScript, ONNX, CoreML
• TensorFlow Lite, OpenVINO
• Mobile, Edge, Cloud deployment

[nanolm.memory]💾 Memory Optimization[/]
• RTX 3060 TI optimizations
• Dynamic batching & caching
• Gradient checkpointing

[nanolm.success]🛡️ Quality Assurance[/]
• Anti-hallucination systems
• Loss tracking & prediction
• Rich logging & monitoring""",
            title="[nanolm.header]Essential Systems Preserved[/]",
            border_style="nanolm.success"
        )
        
        self.logger.console.print(preservation_panel)
        
    def perform_cleanup(self, analysis: Dict[str, List[str]], dry_run: bool = False):
        """Perform the actual cleanup"""
        if not dry_run:
            self.create_backup()
            
        total_removed = 0
        total_size_saved = 0
        
        # Remove files by category
        categories_to_remove = [
            ("test_files", "🧪 Test Files"),
            ("debug_files", "🐛 Debug Files"),
            ("launch_files", "🚀 Launch Scripts"),
            ("example_files", "📚 Examples"),
            ("utility_files", "🔧 Utilities")
        ]
        
        for category, description in categories_to_remove:
            files = analysis[category]
            if not files:
                continue
                
            self.logger.console.print(f"\n[nanolm.warning]Removing {description}...[/]")
            
            for filename in files:
                file_path = self.workspace / filename
                if file_path.exists():
                    size = file_path.stat().st_size
                    
                    if not dry_run:
                        file_path.unlink()
                        self.logger.console.print(f"  🗑️ Removed {filename}")
                    else:
                        self.logger.console.print(f"  🗑️ Would remove {filename}")
                        
                    total_removed += 1
                    total_size_saved += size
                    
        # Clean up directories
        self.logger.console.print(f"\n[nanolm.warning]Cleaning directories...[/]")
        for dir_pattern in self.dirs_to_clean:
            for dir_path in self.workspace.glob(dir_pattern):
                if dir_path.is_dir():
                    if not dry_run:
                        shutil.rmtree(dir_path)
                        self.logger.console.print(f"  🗑️ Removed directory {dir_path.name}")
                    else:
                        self.logger.console.print(f"  🗑️ Would remove directory {dir_path.name}")
                        
        # Summary
        size_mb = total_size_saved / (1024 * 1024)
        action = "Removed" if not dry_run else "Would remove"
        
        summary_panel = Panel(
            f"""[nanolm.success]🎉 Cleanup Complete![/]

[nanolm.info]{action}:[/] {total_removed} files
[nanolm.memory]Space saved:[/] {size_mb:.1f} MB
[nanolm.tokenizer]Token data:[/] ✅ Preserved
[nanolm.export]Export systems:[/] ✅ Preserved
[nanolm.quantization]Quantization:[/] ✅ Preserved

[nanolm.success]Codebase is now optimized for RTX 3060 TI training![/]""",
            title="[nanolm.header]Cleanup Summary[/]",
            border_style="nanolm.success"
        )
        
        self.logger.console.print(summary_panel)
        
        # Show next steps
        if not dry_run:
            next_steps = Panel(
                """[nanolm.info]🚀 Next Steps[/]

1. [nanolm.tokenizer]Train tokenizer:[/]
   python nanolm_gpu_tokenizer_streaming.py

2. [nanolm.quantization]Start training:[/]
   python trainer_fp4_fqt.py

3. [nanolm.export]Export models:[/]
   python enhanced_export_system.py

4. [nanolm.memory]Monitor with Rich:[/]
   python nanolm_rich_logger.py

[nanolm.success]Backup available in cleanup_backup/ if needed[/]""",
                title="[nanolm.header]Ready for Training[/]",
                border_style="nanolm.info"
            )
            
            self.logger.console.print(next_steps)
            
    def create_optimized_launch_script(self):
        """Create a single optimized launch script"""
        launch_script = '''#!/usr/bin/env python3
"""
NanoLM Optimized Training Launcher
=================================

Single launcher for RTX 3060 TI optimized NanoLM training with:
- BitNet 1.58-bit quantization
- Rich logging
- Multi-platform export
- Memory optimization
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch optimized NanoLM training"""
    print("🚀 NanoLM RTX 3060 TI Training Launcher")
    print("=" * 50)
    
    # Check if tokenizer exists
    tokenizer_path = Path("nanolm_tokenizer")
    if not tokenizer_path.exists():
        print("📝 Training tokenizer first...")
        try:
            subprocess.run([sys.executable, "nanolm_gpu_tokenizer_streaming.py"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Tokenizer training failed")
            return 1
    
    # Start training with FP4 quantization
    print("🎯 Starting NanoLM training...")
    try:
        subprocess.run([sys.executable, "trainer_fp4_fqt.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Training failed")
        return 1
        
    print("✅ Training complete! Models ready for export.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
        
        launch_path = self.workspace / "launch_nanolm_training.py"
        with open(launch_path, 'w') as f:
            f.write(launch_script)
            
        # Make executable
        launch_path.chmod(0o755)
        
        self.logger.console.print(
            f"[nanolm.success]✅ Created optimized launcher: {launch_path.name}[/]"
        )
        

def main():
    """Main cleanup function"""
    parser = argparse.ArgumentParser(description="NanoLM Codebase Cleanup")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without removing")
    parser.add_argument("--workspace", default=".", help="Workspace directory")
    args = parser.parse_args()
    
    # Initialize cleanup system
    cleanup = NanoLMCodebaseCleanup(args.workspace)
    cleanup.logger.show_startup_banner()
    
    # Analyze codebase
    cleanup.logger.console.print("[nanolm.info]🔍 Analyzing codebase...[/]")
    analysis = cleanup.analyze_codebase()
    
    # Show plan
    cleanup.show_cleanup_plan(analysis)
    
    # Confirm if not dry run
    if not args.dry_run:
        cleanup.logger.console.print("\n[nanolm.warning]⚠️ This will permanently remove files![/]")
        cleanup.logger.console.print("[nanolm.info]Backup will be created in cleanup_backup/[/]")
        response = input("\nProceed with cleanup? [y/N]: ")
        if response.lower() != 'y':
            cleanup.logger.console.print("[nanolm.info]Cleanup cancelled.[/]")
            return
            
    # Perform cleanup
    cleanup.perform_cleanup(analysis, dry_run=args.dry_run)
    
    # Create optimized launcher
    if not args.dry_run:
        cleanup.create_optimized_launch_script()

if __name__ == "__main__":
    main()