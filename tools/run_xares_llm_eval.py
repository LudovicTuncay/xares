
import argparse
import os
import sys
import glob
import subprocess
import csv
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from loguru import logger

# Add project root to sys.path
root_dir = Path(__file__).parent.parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

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
    
    # Identify default (last.ckpt)
    default_idx = -1
    for idx, cp in enumerate(checkpoints):
        if cp.name == "last.ckpt":
            default_idx = idx
            break
            
    if use_default:
        if default_idx != -1:
            logger.info(f"Auto-selected default checkpoint: {checkpoints[default_idx]}")
            return checkpoints[default_idx]
        else:
            logger.error("Multiple checkpoints found but no 'last.ckpt' to use as default.")
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

def get_available_tasks() -> List[str]:
    """
    Scans src/xares_llm/tasks/single/train for task configs.
    Returns a list of task names (e.g. 'esc-50').
    """
    tasks_dir = Path("src/xares_llm/tasks/single/train")
    if not tasks_dir.exists():
        return []
        
    tasks = []
    for task_file in tasks_dir.glob("*_config.yaml"):
        name = task_file.name.replace("_config.yaml", "")
        tasks.append(name)
    return sorted(tasks)

def resolve_tasks(task_args: List[str]) -> List[str]:
    """
    Resolves task arguments to a list of task names.
    Supports:
    - "all": returns "all" (let xares_llm handle it, or expand if we want loop)
             Actually, to support aggregation, running tasks individually is better if 'all' config produces one big output.
             But 'all' config in xares might run all datasets sequentially in one process.
             If we use 'all', the output will be in experiments/all_config/audio_jepa/scores.tsv
             which should contain all scores.
             
             If user specifies multiple individual tasks "esc-50 clotho", we loop.
             If user specifies "all", we pass "all".
    """
    available_tasks = get_available_tasks()
    resolved_tasks = []
    
    if not task_args:
        # Default: all
        return ["all"]

    for arg in task_args:
        if arg == "all":
            return ["all"]
        elif arg in available_tasks:
            resolved_tasks.append(arg)
        elif arg in ["task1", "task2"]:
             resolved_tasks.append(arg)
        elif "*" in arg:
            # Glob pattern matching against available tasks
            matched = [t for t in available_tasks if glob.fnmatch.fnmatch(t, arg)]
            if not matched:
                logger.warning(f"No tasks matched pattern: {arg}")
            resolved_tasks.extend(matched)
        else:
            if arg not in available_tasks:
                logger.warning(f"Task '{arg}' not found in single tasks. Assuming it works for xares_llm.run anyway.")
            resolved_tasks.append(arg)
            
    return sorted(list(set(resolved_tasks)))

def export_results(tasks: List[str], output_filename: str = "results.csv"):
    """
    Aggregates scores.tsv from experiment directories and saves to a single CSV.
    """
    # Assuming output dir is 'experiments/' (default in xares_llm)
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        logger.warning(f"Experiments directory not found at {experiments_dir}. Cannot export results.")
        return

    aggregated_results = []
    
    # We need to find where results are.
    # Structure: experiments/{config_name}/audio_jepa/scores.tsv
    # config_name usually matches task name + "_config" for single tasks.
    # For "all", "task1", "task2", it matches the config file stem.
    
    # Heuristic: search all scores.tsv in experiments/
    score_files = list(experiments_dir.glob("**/scores.tsv"))
    
    if not score_files:
        logger.warning("No scores.tsv files found.")
        return
        
    logger.info(f"Found {len(score_files)} score files. Aggregating...")
    
    all_scores = []
    
    for sf in score_files:
        # parent is audio_jepa (model name), parent.parent is config_name
        config_name = sf.parent.parent.name
        model_name = sf.parent.name
        
        try:
            with open(sf, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    row['config'] = config_name
                    row['model'] = model_name
                    all_scores.append(row)
        except Exception as e:
            logger.error(f"Error reading {sf}: {e}")

    if not all_scores:
        logger.warning("No scores extracted.")
        return
        
    # Write to CSV
    keys = ['config', 'model'] + [k for k in all_scores[0].keys() if k not in ['config', 'model']]
    
    try:
        with open(output_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_scores)
        logger.info(f"Results exported to {output_filename}")
    except Exception as e:
        logger.error(f"Error writing results to {output_filename}: {e}")

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

    # 1. Find Config
    config_path = find_file(root_path, ["**/.hydra/config.yaml", "**/config.yaml"])
    if not config_path:
        logger.error(f"No config.yaml or .hydra/config.yaml found in {root_path}")
        sys.exit(1)
    
    logger.info(f"Using config: {config_path}")

    # 2. Find Checkpoint
    checkpoints = find_all_files(root_path, "**/*.ckpt")
    if not checkpoints:
        logger.error(f"No .ckpt files found in {root_path}")
        sys.exit(1)
        
    checkpoint_path = select_checkpoint(checkpoints, use_default=use_default_ckpt)
    logger.info(f"Using checkpoint: {checkpoint_path}")

    # 3. Prepare Environment
    env = os.environ.copy()
    env["AUDIO_JEPA_CONFIG"] = str(config_path)
    env["AUDIO_JEPA_CHECKPOINT"] = str(checkpoint_path)
    
    # 4. Resolve Tasks
    final_tasks = resolve_tasks(tasks)
    
    if not final_tasks:
        logger.error("No tasks to run.")
        sys.exit(1)
        
    logger.info(f"Tasks to run: {final_tasks}")
    
    encoder_path = "src/xares_llm/encoders/audio_jepa.py"
    if not Path(encoder_path).exists():
        logger.error(f"Encoder file not found at {encoder_path}")
        sys.exit(1)

    # 5. Run Tasks
    task_failures = 0
    for task in final_tasks:
        logger.info(f"--- Running Task: {task} ---")
        
        cmd = [
            "uv", "run", "-m", "xares_llm.run",
            encoder_path,
            task, # train_config
            task, # eval_config
        ] + extra_args
        
        cmd_str = " ".join(cmd)
        logger.info(f"Running command: {cmd_str}")
        
        if not dry_run:
            try:
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Task {task} failed with exit code {e.returncode}")
                task_failures += 1
            except KeyboardInterrupt:
                logger.info("Evaluation cancelled.")
                sys.exit(130)
    
    if not dry_run:
        export_results(final_tasks, "evaluation_results.csv")
        
    if task_failures > 0:
        logger.error(f"{task_failures} tasks failed.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XARES-LLM evaluation from a checkpoint folder.")
    parser.add_argument("checkpoint_folder", help="Path to the folder containing checkpoint and config")
    parser.add_argument("tasks", nargs="*", help="Task names (e.g. 'esc-50', 'all') (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    parser.add_argument("-y", "--default-ckpt", action="store_true", help="Automatically select 'last.ckpt' if multiple checkpoints exist")
    
    # Capture unknown args to pass to xares_llm.run
    args, unknown_args = parser.parse_known_args()
    
    run_eval(args.checkpoint_folder, args.tasks, unknown_args, args.dry_run, args.default_ckpt)
