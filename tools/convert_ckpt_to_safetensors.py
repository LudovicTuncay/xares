import argparse
import sys
from pathlib import Path
from typing import Dict, List
import torch
from loguru import logger

try:
    from safetensors.torch import save_file
except ImportError:
    logger.error("safetensors is not installed. Please install it with 'pip install safetensors'.")
    sys.exit(1)

def convert_file(ckpt_path: Path) -> None:
    """Converts a single .ckpt file to .safetensors."""
    if ckpt_path.suffix != ".ckpt":
        logger.warning(f"Skipping {ckpt_path}: Not a .ckpt file")
        return
        
    safetensors_path = ckpt_path.with_suffix(".safetensors")
    if safetensors_path.exists():
        logger.info(f"Skipping {ckpt_path}: {safetensors_path.name} already exists")
        return

    logger.info(f"Converting {ckpt_path} to .safetensors...")
    
    try:
        # Load on CPU to avoid CUDA requirements
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        state_dict: Dict[str, torch.Tensor] = {}
        
        # Handle PyTorch Lightning checkpoints which wrap weights in "state_dict"
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            logger.debug(f"Detected PyTorch Lightning checkpoint structure for {ckpt_path.name}")
            raw_state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
             # Assume it's a direct state dict
            raw_state_dict = checkpoint
        else:
            # Fallback for other formats (unlikely for .ckpt)
             logger.warning(f"Unknown checkpoint format for {ckpt_path}, attempting to use as state dict.")
             raw_state_dict = checkpoint

        # Filter for only tensors and ensure keys are strings
        for k, v in raw_state_dict.items():
            if isinstance(v, torch.Tensor):
                state_dict[str(k)] = v
            else:
                 logger.debug(f"Skipping non-tensor key: {k} (type: {type(v)})")

        if not state_dict:
             logger.error(f"No tensor data found in {ckpt_path}. Skipping.")
             return

        # Handle shared tensors (e.g. tied weights)
        # Safetensors does not support shared memory. We must clone shared tensors.
        data_ptrs = {}
        for k, v in state_dict.items():
            ptr = v.data_ptr()
            if ptr in data_ptrs:
                # This tensor shares memory with a previous one. Clone it to detach.
                logger.debug(f"Cloning shared tensor: {k} (shares with {data_ptrs[ptr]})")
                state_dict[k] = v.clone()
            else:
                data_ptrs[ptr] = k

        save_file(state_dict, safetensors_path)
        logger.success(f"Saved {safetensors_path}")
        
    except Exception as e:
        logger.error(f"Failed to convert {ckpt_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert .ckpt files to .safetensors recursively.")
    parser.add_argument("path", nargs="?", default=".", help="File or directory to scan for .ckpt files (default: current directory)")
    
    args = parser.parse_args()
    
    root_path = Path(args.path).resolve()
    
    if not root_path.exists():
        logger.error(f"Path not found: {root_path}")
        sys.exit(1)
        
    if root_path.is_file():
        convert_file(root_path)
    else:
        logger.info(f"Scanning {root_path} for .ckpt files...")
        ckpt_files = list(root_path.rglob("*.ckpt"))
        
        if not ckpt_files:
            logger.warning(f"No .ckpt files found in {root_path}")
            sys.exit(0)
            
        logger.info(f"Found {len(ckpt_files)} .ckpt files.")
        for ckpt in ckpt_files:
            convert_file(ckpt)

if __name__ == "__main__":
    main()
