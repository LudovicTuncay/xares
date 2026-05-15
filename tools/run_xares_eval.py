import argparse
import os
import sys
import glob
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger

# Add project root to sys.path to allow importing modules from src
root_dir = Path(__file__).parent.parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from xares.utils import attr_from_py_path
    from xares.common import XaresSettings
except ImportError:
    # Fallback if xares is not installed in the environment running this script
    # We try to add 'src' to path to find xares package if it's there
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from xares.utils import attr_from_py_path
    from xares.common import XaresSettings


def find_file(root_dir: Path, patterns: List[str]) -> Optional[Path]:
    for pattern in patterns:
        files = list(root_dir.glob(pattern))
        if files:
            return files[0]
    return None

def find_all_files(root_dir: Path, pattern: str) -> List[Path]:
    return sorted(list(root_dir.glob(pattern)))

def select_checkpoint(checkpoints: List[Path], use_default: bool = False) -> Path:
    if len(checkpoints) == 1:
        return checkpoints[0]
    
    # Identify default (last.ckpt or last.safetensors)
    default_idx = -1
    for idx, cp in enumerate(checkpoints):
        if cp.name in ["last.ckpt", "last.safetensors"]:
            default_idx = idx
            break
            
    if use_default:
        if default_idx != -1:
            logger.info(f"Auto-selected default checkpoint: {checkpoints[default_idx]}")
            return checkpoints[default_idx]
        else:
            logger.error("Multiple checkpoints found but no 'last.ckpt' or 'last.safetensors' to use as default.")
            # List them for clarity
            for idx, cp in enumerate(checkpoints):
                print(f"[{idx}] {cp}")
            sys.exit(1)
    
    print(f"\nMultiple checkpoints found:")
    for idx, cp in enumerate(checkpoints):
        if idx == default_idx:
            print(f"[{idx}] {cp} (default)")
        else:
            print(f"[{idx}] {cp}")
            
    try:
        selection = input(f"Select checkpoint index [default: {default_idx if default_idx != -1 else 'None'}]: ").strip()
        if not selection:
            if default_idx != -1:
                return checkpoints[default_idx]
            else:
                pass
        
        idx = int(selection)
        if 0 <= idx < len(checkpoints):
            return checkpoints[idx]
        else:
            logger.error(f"Index {idx} out of range.")
            sys.exit(1)
    except (ValueError, KeyboardInterrupt):
        logger.error("Invalid selection or cancelled.")
        sys.exit(1)

def get_available_tasks() -> dict[str, str]:
    """
    Scans src/tasks for task files and returns a mapping of task name to file path.
    Example: 'esc50' -> 'src/tasks/esc50_task.py'
    """
    tasks_dir = Path("src/tasks")
    if not tasks_dir.exists():
        return {}
        
    task_map = {}
    for task_file in tasks_dir.glob("*_task.py"):
        # name is filename without _task.py
        name = task_file.name.replace("_task.py", "")
        task_map[name] = str(task_file)
        
    return task_map

def resolve_tasks(task_args: List[str]) -> List[str]:
    """
    Resolves task arguments to file paths.
    Supports:
    - "all": returns all tasks in src/tasks
    - "name": looks up name in available tasks
    - "path/to/file.py": keeps as is
    """
    available_tasks = get_available_tasks()
    resolved_paths = []
    
    if not task_args:
        # Default behavior: run all tasks
        return [str(p) for p in Path("src/tasks").glob("*_task.py")]

    for arg in task_args:
        if arg == "all":
            return [str(p) for p in Path("src/tasks").glob("*_task.py")]
        elif arg in available_tasks:
            resolved_paths.append(available_tasks[arg])
        elif os.path.exists(arg):
             # It's a direct file path
            resolved_paths.append(arg)
        elif "*" in arg:
            # It's a glob pattern
            resolved_paths.append(arg)
        else:
            logger.error(f"Task '{arg}' not found.")
            print("\nAvailable tasks:")
            for name in sorted(available_tasks.keys()):
                print(f"  {name}")
            sys.exit(1)
            
    return resolved_paths

def get_task_size(task_path: str) -> int:
    """
    Calculates the size of the audio tar files for a given task.
    Returns size in bytes.
    """
    try:
        # Resolve path relative to CWD if possible, or leave as is
        # attr_from_py_path expects a string path like 'src/tasks/esc50_task.py'
        # which maps to module src.tasks.esc50_task
        
        # We assume task_path is relative to project root or absolute.
        # If absolute, we might need to adjust for import.
        # But resolve_tasks returns paths like 'src/tasks/...' (relative) usually.
        
        config_fn = attr_from_py_path(str(task_path), endswith="_config")
        config = config_fn(None) # encoder=None
        
        env_root = Path(config.env_root) if config.env_root else Path(XaresSettings().env_root)
        task_env_dir = env_root / config.name
        
        total_size = 0
        if task_env_dir.exists():
            # config.audio_tar_name_of_split is a dict
            # We want unique patterns
            patterns = set(config.audio_tar_name_of_split.values())
            for pat in patterns:
                for f in task_env_dir.glob(pat):
                    if f.is_file():
                        total_size += f.stat().st_size
        return total_size
    except Exception as e:
        logger.debug(f"Could not calculate size for {task_path}: {e}")
        return 0

def run_eval(
    checkpoint_folder: str,
    tasks: List[str],
    extra_args: List[str],
    dry_run: bool = False,
    use_default_ckpt: bool = False
):
    root_path = Path(checkpoint_folder).resolve()
    if not root_path.exists():
        logger.error(f"Folder not found: {root_path}")
        sys.exit(1)

    # Resolve task names to paths
    tasks = resolve_tasks(tasks)

    # 1. Expand Glob Patterns in Tasks
    expanded_tasks = []
    for t in tasks:
        if "*" in t:
            matched = glob.glob(t)
            if not matched:
                logger.warning(f"No files matched pattern: {t}")
            expanded_tasks.extend(matched)
        else:
            expanded_tasks.append(t)
            
    if not expanded_tasks:
        logger.error("No task files found to run.")
        sys.exit(1)
        
    # 2. Sort tasks by size (Biggest first)
    logger.info("Calculating task dataset sizes to optimize execution order...")
    tasks_with_size: List[Tuple[str, int]] = []
    for t in expanded_tasks:
        size = get_task_size(t)
        tasks_with_size.append((t, size))
        
    # Sort descending
    tasks_with_size.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("Execution Order (Biggest to Smallest):")
    for t, s in tasks_with_size:
        size_gb = s / (1024 * 1024 * 1024)
        logger.info(f"  {Path(t).stem:<25} {size_gb:6.2f} GB")
        
    final_task_list = [t for t, _ in tasks_with_size]

    # 3. Find Config
    # Check common locations
    config_path = find_file(root_path, ["**/.hydra/config.yaml", "**/config.yaml"])
    
    if not config_path:
        logger.error(f"No config.yaml or .hydra/config.yaml found in {root_path}")
        sys.exit(1)
    
    logger.info(f"Using config: {config_path}")

    # 4. Find Checkpoint
    checkpoints = find_all_files(root_path, "**/*.ckpt") + find_all_files(root_path, "**/*.safetensors")
    checkpoints = sorted(checkpoints)

    if not checkpoints:
        logger.error(f"No .ckpt or .safetensors files found in {root_path}")
        sys.exit(1)
        
    checkpoint_path = select_checkpoint(checkpoints, use_default=use_default_ckpt)
    logger.info(f"Using checkpoint: {checkpoint_path}")

    # 5. Prepare Environment
    env = os.environ.copy()
    env["AUDIO_JEPA_CONFIG"] = str(config_path)
    env["AUDIO_JEPA_CHECKPOINT"] = str(checkpoint_path)
    
    # 6. Construct Command
    # Use the sorted final_task_list
        
    cmd = [
        "uv", "run", "python", "-u", "-m", "xares.run",
        "src/xares/encoders/audio_jepa.py"
    ] + final_task_list + extra_args
    
    cmd_str = " ".join(cmd)
    logger.info(f"Running command: {cmd_str}")
    
    if not dry_run:
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation failed with exit code {e.returncode}")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            logger.info("Evaluation cancelled.")
            sys.exit(130)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AudioJEPA evaluation from a checkpoint folder.")
    parser.add_argument("checkpoint_folder", help="Path to the folder containing checkpoint and config")
    parser.add_argument("tasks", nargs="*", help="Task names (e.g. 'esc50', 'all') or file paths (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    parser.add_argument("--max-jobs", default="1", help="Max parallel jobs (default: 1)")
    parser.add_argument("-y", "--default-ckpt", action="store_true", help="Automatically select 'last.ckpt' or 'last.safetensors' if multiple checkpoints exist")
    
    # Capture unknown args to pass to xares.run
    args, unknown_args = parser.parse_known_args()
    
    # Add max-jobs to unknown_args if not present (handled by argparse but we want to pass it through)
    pass_through_args = []
    if args.max_jobs:
        pass_through_args.extend(["--max-jobs", args.max_jobs])
        
    pass_through_args.extend(unknown_args)
    
    run_eval(args.checkpoint_folder, args.tasks, pass_through_args, args.dry_run, args.default_ckpt)
