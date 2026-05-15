from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from loguru import logger

root_dir = Path(__file__).parent.parent.resolve()
tools_dir = Path(__file__).parent.resolve()
for path in (root_dir, root_dir / "src", tools_dir):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_best_rq_eval import (  # noqa: E402
    find_all_files,
    find_file,
    get_task_size,
    resolve_tasks,
    select_checkpoint,
)


def run_eval(
    checkpoint_folder: str,
    tasks: List[str],
    extra_args: List[str],
    dry_run: bool = False,
    use_default_ckpt: bool = False,
) -> None:
    root_path = Path(checkpoint_folder).resolve()
    if not root_path.exists():
        logger.error(f"Folder not found: {root_path}")
        sys.exit(1)

    tasks = resolve_tasks(tasks)
    expanded_tasks = []
    for task in tasks:
        if "*" in task:
            matched = sorted(glob.glob(task))
            if not matched:
                logger.warning(f"No files matched pattern: {task}")
            expanded_tasks.extend(matched)
        else:
            expanded_tasks.append(task)

    if not expanded_tasks:
        logger.error("No task files found to run.")
        sys.exit(1)

    logger.info("Calculating task dataset sizes to optimize execution order...")
    tasks_with_size = [(task, get_task_size(task)) for task in expanded_tasks]
    tasks_with_size.sort(key=lambda item: item[1], reverse=True)

    logger.info("Execution Order (Biggest to Smallest):")
    for task, size in tasks_with_size:
        size_gb = size / (1024 * 1024 * 1024)
        logger.info(f"  {Path(task).stem:<25} {size_gb:6.2f} GB")
    final_task_list = [task for task, _ in tasks_with_size]

    config_path = find_file(root_path, ["**/.hydra/config.yaml", "**/config.yaml"])
    if not config_path:
        logger.error(f"No config.yaml or .hydra/config.yaml found in {root_path}")
        sys.exit(1)
    logger.info(f"Using config: {config_path}")

    checkpoints = sorted(
        find_all_files(root_path, "**/*.ckpt")
        + find_all_files(root_path, "**/*.safetensors")
    )
    if not checkpoints:
        logger.error(f"No .ckpt or .safetensors files found in {root_path}")
        sys.exit(1)

    checkpoint_path = select_checkpoint(checkpoints, use_default=use_default_ckpt)
    logger.info(f"Using checkpoint: {checkpoint_path}")

    env = os.environ.copy()
    env["BEST_RQ3_CONFIG"] = str(config_path)
    env["BEST_RQ3_CHECKPOINT"] = str(checkpoint_path)

    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "-m",
        "xares.run",
        "src/xares/encoders/best_rq3.py",
    ] + final_task_list + extra_args

    logger.info(f"Running command: {' '.join(cmd)}")

    if dry_run:
        return

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error(f"Evaluation failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)
    except KeyboardInterrupt:
        logger.info("Evaluation cancelled.")
        sys.exit(130)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BEST-RQ-3 waveform encoder evaluation from a checkpoint folder."
    )
    parser.add_argument(
        "checkpoint_folder",
        help="Path to the folder containing checkpoint and config",
    )
    parser.add_argument(
        "tasks",
        nargs="*",
        help="Task names (e.g. 'esc50', 'all') or file paths (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without running",
    )
    parser.add_argument(
        "--max-jobs",
        default="1",
        help="Max parallel jobs (default: 1)",
    )
    parser.add_argument(
        "-y",
        "--default-ckpt",
        action="store_true",
        help=(
            "Automatically select 'last.ckpt' or 'last.safetensors' "
            "if multiple checkpoints exist"
        ),
    )

    args, unknown_args = parser.parse_known_args()

    pass_through_args = []
    if args.max_jobs:
        pass_through_args.extend(["--max-jobs", args.max_jobs])
    pass_through_args.extend(unknown_args)

    run_eval(
        args.checkpoint_folder,
        args.tasks,
        pass_through_args,
        dry_run=args.dry_run,
        use_default_ckpt=args.default_ckpt,
    )
