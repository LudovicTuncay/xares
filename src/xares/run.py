import warnings
import sys
# # Suppress TorchScript deprecation warning from ignite
# # This is an internal warning in ignite 0.5.x that we cannot fix directly
# warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*TorchScript.*")

import argparse
from functools import partial

# # Suppress TorchScript deprecation warning from ignite
# # This is an internal warning in ignite 0.5.x that we cannot fix directly
# warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*TorchScript.*")

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger

from xares.audio_encoder_checker import check_audio_encoder
from xares.common import setup_global_logger
from xares.metrics import weighted_average
from xares.task import XaresTask
from xares.utils import attr_from_py_path, get_encoder_run_name


def worker(
    encoder_py: None | str,
    task_py: str,
    do_download: bool = False,
    do_encode: bool = False,
    do_mlp: bool = False,
    do_knn: bool = False,
):
    # Encoder setup
    encoder = attr_from_py_path(encoder_py, endswith="Encoder")() if encoder_py else None

    # Task setup
    config = attr_from_py_path(task_py, endswith="_config")(encoder)
    
    # NEW: Determine run name and inject into config
    # We calculate the run name here so we can use it to distinguish checkpoints
    run_name = get_encoder_run_name(encoder=encoder, encoder_py_path=encoder_py)
    # If run_name is available (and not "unknown_checkpoint" or class name fallback), use it
    # get_encoder_run_name returns class name as fallback if no checkpoint path
    # But user wants to use run name (e.g. "2025-12-19_14-52-27")
    
    # We trust get_encoder_run_name to return something reasonable.
    # If it is "unknown_checkpoint", maybe we shouldn't use it if we want to default to class name?
    # But current behavior is using class name. If get_encoder_run_name returns class name, it's same.
    # If it returns "unknown_checkpoint", that's also distinctive.
    
    if run_name:
         config.encoder_name = run_name

    if config.disabled:
        logger.warning(f"Task {config.name} is disabled, skipping")
        return config.formal_name, (0, 0), (0, 0), True
    task = XaresTask(config=config)

    # Run the task
    if do_download:
        logger.info(f"Downloading data for task {config.name} ...")
        task.download_audio_tar()
        logger.info(f"Task {config.name} data ready.")

    if do_encode:
        logger.info(f"Running make_encoded_tar for task {config.name} ...")
        task.make_encoded_tar()
        logger.info(f"Task {config.name} encoded.")

    if config.private and not (task.encoded_tar_dir / task.config.xares_settings.encoded_ready_filename).exists():
        logger.warning(f"Task {config.name} is private and not ready, skipping.")
        do_mlp = do_knn = False

    mlp_score = (0, 0)
    if do_mlp:
        logger.info(f"Running run_mlp for task {config.name} ...")
        mlp_score = task.run_mlp()
        logger.info(f"MLP score of {config.name}: {mlp_score}")

    knn_score = (0, 0)
    if do_knn and task.config.do_knn:
        logger.info(f"Running KNN for task {config.name} ...")
        knn_score = task.run_knn()
        logger.info(f"KNN score of {config.name}: {knn_score}")

    torch.cuda.empty_cache()
    return task.config.formal_name, mlp_score, knn_score, task.config.private


def stage_1(encoder_py, task_py, gpu_id):
    if gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    return worker(encoder_py, task_py, do_encode=True)


def stage_2(encoder_py, task_py, result: dict):
    result.update({task_py: worker(encoder_py, task_py, do_mlp=True, do_knn=True)})


def main(args):
    setup_global_logger()
    enable_multiprocessing = args.max_jobs > 0
    torch.multiprocessing.set_start_method("spawn")

    # Stage 0: Download all datasets
    stage_0 = partial(worker, do_download=True)
    if args.from_stage <= 0:
        try:
            if enable_multiprocessing:
                with mp.Pool(processes=args.max_jobs) as pool:
                    pool.starmap(stage_0, [(None, task_py) for task_py in args.tasks_py])
            else:
                for task_py in args.tasks_py:
                    stage_0(None, task_py)
            logger.info("Stage 0 completed: All data downloaded.")
        except Exception as e:
            if "Max retries exceeded with url" in str(e):
                logger.error(e)
                logger.error("This may be caused by Zenodo temporarily banning your connection.")
                logger.error("You may need to wait for a few hours and retry.")
                logger.error("Alternatively, you can download manually using `tools/download_manually.sh`.")
                return
            else:
                logger.error(f"Error in stage 0 (download): {e} Must fix it before proceeding.")
                return
    else:
        # Ensure pretrained model has been saved at local if stage 0 is skipped
        for task_py in args.tasks_py:
            worker(None, task_py)

    if args.to_stage == 0:
        return

    # Check if the encoder supports the multiprocessing
    if enable_multiprocessing:
        try:
            with mp.Pool(processes=1) as pool:
                pool.starmap(worker, [(args.encoder_py, args.tasks_py[0])])
        except Exception:
            logger.warning("Multiprocessing is not supported for the encoder. Falling back to a single process.")
            logger.warning("If single processing is too slow, you can manually parallelize tasks with a shell script.")
            logger.warning("For models from Hugging Face, try save locally, which might fix for multiprocessing.")
            enable_multiprocessing = False

    # Double check the encoder and download the pretrained weights
    encoder = attr_from_py_path(args.encoder_py, endswith="Encoder")()
    if not check_audio_encoder(encoder):
        raise ValueError("Invalid encoder")
    encoder_checkpoint_path = getattr(encoder, "checkpoint_path", None)
    del encoder

    ckpt_name = get_encoder_run_name(encoder_checkpoint_path=encoder_checkpoint_path, encoder_py_path=args.encoder_py)
    
    # Stage 1: Execute make_encoded_tar
    if args.from_stage <= 1:
        try:
            if enable_multiprocessing:
                num_gpus = torch.cuda.device_count()
                tasks_args = []
                for i, task_py in enumerate(args.tasks_py):
                    gpu_id = (i % num_gpus) if num_gpus > 0 else -1
                    tasks_args.append((args.encoder_py, task_py, gpu_id))
                
                with mp.Pool(processes=args.max_jobs) as pool:
                    pool.starmap(stage_1, tasks_args)
            else:
                for task_py in args.tasks_py:
                    worker(args.encoder_py, task_py, do_encode=True)

            logger.info("Stage 1 completed: All tasks encoded.")
        except FileNotFoundError as e:
            logger.error(f"Task filename pattern error: {e}")
            return
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA out of memory. Try reducing `config.batch_size_encode` of tasks.")
            else:
                logger.error(f"Error in stage 1 (encode): {e} Must fix it before proceeding.")
                return
        logger.info("Stage 1 completed: All tasks encoded.")
    if args.to_stage == 1:
        return

    # Stage 2: Execute mlp and knn scoring
    if args.from_stage <= 2 and args.to_stage >= 2:
        if enable_multiprocessing:
            manager = mp.Manager()
            return_dict = manager.dict()
            with mp.Pool(processes=args.max_jobs) as pool:
                pool.starmap(
                    partial(stage_2, result=return_dict),
                    [(args.encoder_py, task_py) for task_py in args.tasks_py],
                )
        else:
            return_dict = {}
            for task_py in args.tasks_py:
                return_dict[task_py] = worker(args.encoder_py, task_py, do_mlp=True, do_knn=True)
        logger.info("Scoring completed: All tasks scored.")

        # Print results
        df = pd.DataFrame(return_dict.items(), columns=["py", "Scores"]).drop(columns=["py"])
        df["Task"] = df["Scores"].apply(lambda x: x[0])
        df["MLP_Score"] = df["Scores"].apply(lambda x: x[1][0])
        df["KNN_Score"] = df["Scores"].apply(lambda x: x[2][0])
        df["Private"] = df["Scores"].apply(lambda x: x[3] if len(x) > 3 else True)
        df.drop(columns=["Scores"], inplace=True)
        df.sort_values(by="Task", inplace=True)

        print(f"\nResults:\n{df.to_string(index=False)}")

        avg_mlp_all, avg_knn_all = weighted_average({k: v[1:-1] for k, v in return_dict.items()})
        print("\nWeighted Average MLP Score for All Datasets:", avg_mlp_all)
        print("Weighted Average KNN Score for All Datasets:", avg_knn_all)
        if any([v[-1] == True for v in return_dict.values()]):
            avg_mlp_public, avg_knn_public = weighted_average(
                {k: v[1:-1] for k, v in return_dict.items() if v[-1] == True}
            )

            print("\nWeighted Average MLP Score for Public Datasets:", avg_mlp_public)
            print("Weighted Average KNN Score for Public Datasets:", avg_knn_public)

        # Save results to CSV
        try:
            from pathlib import Path
            import os
            
            # Extract checkpoint name
            
            # Create results directory
            results_dir = Path("csv_results")
            results_dir.mkdir(exist_ok=True)
            
            mlp_dir = results_dir / "mlp"
            mlp_dir.mkdir(exist_ok=True)
            
            knn_dir = results_dir / "knn"
            knn_dir.mkdir(exist_ok=True)
            
            # Save MLP results
            df_mlp = df[["Task", "MLP_Score", "Private"]].copy()
            mlp_csv_path = mlp_dir / f"{ckpt_name}_MLP.csv"
            df_mlp.to_csv(mlp_csv_path, index=False)
            logger.info(f"Saved MLP results to {mlp_csv_path}")
            
            # Save KNN results
            df_knn = df[["Task", "KNN_Score", "Private"]].copy()
            knn_csv_path = knn_dir / f"{ckpt_name}_KNN.csv"
            df_knn.to_csv(knn_csv_path, index=False)
            logger.info(f"Saved KNN results to {knn_csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results to CSV: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a task")
    parser.add_argument("encoder_py", type=str, help="Encoder path. eg: example/dasheng/dasheng_encoder.py")
    parser.add_argument(
        "tasks_py",
        type=str,
        help="Tasks path. eg: src/tasks/*.py",
        nargs="+",
    )
    parser.add_argument("--max-jobs", type=int, default=1, help="Maximum number of concurrent tasks.")
    parser.add_argument("--from-stage", default=0, type=int, help="First stage to run.")
    parser.add_argument("--to-stage", default=2, type=int, help="Last stage to run.")
    args = parser.parse_args()
    main(args)
