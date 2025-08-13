import os, glob, torch, gzip, pickle
from typing import Dict, Optional, Tuple
import shutil

def save_checkpoint(state: Dict, ckpt_dir: str, step: int, keep_last_k: int = 3, compress: bool = False):
    """Enhanced checkpoint saving with compression option and metadata"""
    os.makedirs(ckpt_dir, exist_ok=True)

    # Add metadata
    state['save_time'] = torch.tensor(0)  # Placeholder for timestamp
    state['pytorch_version'] = torch.__version__

    if compress:
        path = os.path.join(ckpt_dir, f"step_{step:08d}.pt.gz")
        with gzip.open(path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        path = os.path.join(ckpt_dir, f"step_{step:08d}.pt")
        torch.save(state, path)

    # Create a symlink to latest
    latest_path = os.path.join(ckpt_dir, 'latest.pt')
    if os.path.exists(latest_path):
        os.remove(latest_path)
    try:
        os.symlink(os.path.basename(path), latest_path)
    except:
        # Fallback: copy instead of symlink (Windows compatibility)
        shutil.copy2(path, latest_path)

    # Cleanup old checkpoints
    if compress:
        pattern = os.path.join(ckpt_dir, 'step_*.pt.gz')
    else:
        pattern = os.path.join(ckpt_dir, 'step_*.pt')

    ckpts = sorted(glob.glob(pattern))
    if len(ckpts) > keep_last_k:
        for f in ckpts[:-keep_last_k]:
            try:
                os.remove(f)
            except:
                pass

def load_latest(ckpt_dir: str) -> Tuple[Optional[Dict], int]:
    """Enhanced checkpoint loading with compression detection and error handling"""
    if not os.path.exists(ckpt_dir):
        return None, 0

    # Try latest symlink first
    latest_path = os.path.join(ckpt_dir, 'latest.pt')
    if os.path.exists(latest_path):
        try:
            state = torch.load(latest_path, map_location='cpu')
            step = state.get('step', 0)
            return state, step
        except:
            pass  # Fall back to finding latest manually

    # Find latest checkpoint manually
    patterns = [
        os.path.join(ckpt_dir, 'step_*.pt'),
        os.path.join(ckpt_dir, 'step_*.pt.gz')
    ]

    all_ckpts = []
    for pattern in patterns:
        all_ckpts.extend(glob.glob(pattern))

    if not all_ckpts:
        return None, 0

    # Sort by step number
    def extract_step(path):
        basename = os.path.basename(path)
        try:
            return int(basename.split('_')[1].split('.')[0])
        except:
            return 0

    latest = max(all_ckpts, key=extract_step)

    try:
        if latest.endswith('.gz'):
            # Load compressed checkpoint
            with gzip.open(latest, 'rb') as f:
                state = pickle.load(f)
        else:
            state = torch.load(latest, map_location='cpu')

        step = state.get('step', 0)
        print(f"📂 Loaded checkpoint: {latest} (step {step})")
        return state, step

    except Exception as e:
        print(f"❌ Failed to load checkpoint {latest}: {e}")
        return None, 0

def save_model_for_inference(model, tokenizer, save_dir: str, config=None):
    """Save model in inference-ready format"""
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    model_path = os.path.join(save_dir, 'pytorch_model.bin')
    torch.save(model.state_dict(), model_path)

    # Save tokenizer
    tokenizer.save_pretrained(save_dir)

    # Save config
    if config:
        import json
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config.__dict__ if hasattr(config, '__dict__') else config, f, indent=2)

    print(f"💾 Model saved for inference: {save_dir}")

def cleanup_old_checkpoints(ckpt_dir: str, keep_best: bool = True):
    """Cleanup old checkpoints, optionally keeping best model"""
    if not os.path.exists(ckpt_dir):
        return

    patterns = ['step_*.pt', 'step_*.pt.gz']
    files_to_keep = ['latest.pt', 'final_model.pt']

    if keep_best:
        files_to_keep.append('best_model.pt')

    removed_count = 0
    for pattern in patterns:
        for ckpt_path in glob.glob(os.path.join(ckpt_dir, pattern)):
            basename = os.path.basename(ckpt_path)
            if basename not in files_to_keep:
                try:
                    os.remove(ckpt_path)
                    removed_count += 1
                except:
                    pass

    print(f"🧹 Cleaned up {removed_count} old checkpoints")

def get_checkpoint_info(ckpt_dir: str) -> Dict:
    """Get information about available checkpoints"""
    if not os.path.exists(ckpt_dir):
        return {}

    info = {
        'checkpoint_dir': ckpt_dir,
        'regular_checkpoints': [],
        'special_checkpoints': {}
    }

    # Regular step checkpoints
    for pattern in ['step_*.pt', 'step_*.pt.gz']:
        for path in glob.glob(os.path.join(ckpt_dir, pattern)):
            size_mb = os.path.getsize(path) / 1e6
            info['regular_checkpoints'].append({
                'path': path,
                'size_mb': size_mb,
                'compressed': path.endswith('.gz')
            })

    # Special checkpoints
    special_files = ['best_model.pt', 'final_model.pt', 'latest.pt']
    for special in special_files:
        path = os.path.join(ckpt_dir, special)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            info['special_checkpoints'][special] = {
                'path': path,
                'size_mb': size_mb
            }

    return info
