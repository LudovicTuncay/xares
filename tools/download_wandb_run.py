import argparse
import wandb
from pathlib import Path
from loguru import logger

def normalize_run_path(run_path: str) -> str:
    """
    Normalizes a wandb run path to the format 'entity/project/run_id'.
    Handles paths containing '/runs/' and leading slashes.
    """
    # Remove leading slash if present
    if run_path.startswith("/"):
        run_path = run_path[1:]
    
    parts = run_path.split("/")
    
    # Filter out 'runs' if it appears (common in URLs)
    parts = [p for p in parts if p != "runs"]
    
    if len(parts) != 3:
        # It might be that the project name has slashes or spaces, but usually project names are single segments or handled carefully.
        # However, wandb paths are strictly entity/project/run_id.
        # If the user provides something else, we might just try to join them or warn.
        # For 'audio embeddings', it might be passed as one string if quoted, or separate args if not.
        # But here run_path is a single string argument.
        # Let's assume standard 3 parts after cleaning.
        logger.warning(f"Parsed path components {parts} does not look like standard entity/project/run_id. Trying anyway.")

    return "/".join(parts)

def download_run_files(run_path: str, output_root: str = "jepa_checkpoints"):
    api = wandb.Api()
    
    normalized_path = normalize_run_path(run_path)
    logger.info(f"Run path provided: {run_path}")
    logger.info(f"Normalized run path: {normalized_path}")
    
    try:
        run = api.run(normalized_path)
    except Exception as e:
        logger.error(f"Failed to access run '{normalized_path}': {e}")
        logger.error("Please ensure the path is correct and you have access.")
        return

    run_name = run.name if run.name else run.id
    # Clean run name for filesystem
    safe_run_name = "".join([c for c in run_name if c.isalnum() or c in (' ', '.', '_', '-')]).strip()
    
    output_dir = Path(output_root) / safe_run_name
    
    logger.info(f"Run Name: {run_name}")
    logger.info(f"Output Directory: {output_dir}")
    
    if output_dir.exists():
        logger.warning(f"Output directory already exists: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = run.files()
    logger.info(f"Found {len(files)} files. Starting download...")
    
    count = 0
    for file in files:
        try:
            # Download file, preserving structure relative to output_dir
            file.download(root=output_dir, replace=True)
            logger.info(f"Downloaded: {file.name}")
            count += 1
        except Exception as e:
            logger.error(f"Failed to download {file.name}: {e}")
            
    logger.info(f"Finished. Downloaded {count} files to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all files from a WandB run.")
    parser.add_argument("run_path", help="WandB run path (e.g., entity/project/runs/run_id or /entity/project/runs/run_id)")
    parser.add_argument("--output_root", default="jepa_checkpoints", help="Root folder to save files (default: jepa_checkpoints)")
    
    args = parser.parse_args()
    
    download_run_files(args.run_path, args.output_root)
