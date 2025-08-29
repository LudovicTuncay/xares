import argparse
from functools import partial
from os.path import basename

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger

from xares.audio_encoder_checker import check_audio_encoder
from xares.common import setup_global_logger
from xares.metrics import weighted_average
from xares.task import XaresTask
from xares.utils import attr_from_py_path


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
    if config.disabled:
        logger.warning(f"Task {config.name} is disabled, skipping")
        crit = config.criterion
        crit_name = crit if isinstance(crit, str) else (getattr(crit, "__name__", None) or type(crit).__name__)
        return (
            config.formal_name,
            (0, 0),
            (0, 0),
            config.private,
            config.domain,
            config.task_type,
            crit_name,
            str(config.metric),
        )
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
    # Ensure results are CPU-serializable and avoid sending GPU-backed objects
    def _to_float_int_pair(pair):
        try:
            s, w = pair
        except Exception:
            return pair
        # Convert possible numpy/tensor scalars to python types
        try:
            s = float(s) if s is not None else None
        except Exception:
            pass
        try:
            w = int(w)
        except Exception:
            pass
        return (s, w)

    def _criterion_name(c):
        if isinstance(c, str):
            return c
        name = getattr(c, "__name__", None)
        if name is not None:
            return name
        return type(c).__name__

    return (
        task.config.formal_name,
        _to_float_int_pair(mlp_score),
        _to_float_int_pair(knn_score),
        task.config.private,
        task.config.domain,
        task.config.task_type,
        _criterion_name(task.config.criterion),
        str(task.config.metric),
    )


def stage_1(encoder_py, task_py, gpu_id):
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

    # Prepare a friendly column title for encoder
    ckpt_name = None
    ckpt_path = getattr(encoder, "checkpoint_path", None)
    if ckpt_path:
        ckpt_name = basename(ckpt_path)[:-5]  # [:-5] to remove .ckpt
        ckpt_name = ckpt_name.split("-")[2:]  # [2:] to remove Audio-JEPA-
        ckpt_name = " / ".join(ckpt_name)
    
    del encoder

    # Stage 1: Execute make_encoded_tar
    if args.from_stage <= 1:
        try:
            if enable_multiprocessing:
                num_gpus = torch.cuda.device_count()
                with mp.Pool(processes=args.max_jobs) as pool:
                    pool.starmap(
                        stage_1,
                        [(args.encoder_py, task_py, i % num_gpus) for i, task_py in enumerate(args.tasks_py)],
                    )
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
        try:
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
        except RuntimeError as e:
            logger.error(f"Error in stage 2 (scoring): {e} Must fix it before proceeding.")
            return
        logger.info("Scoring completed: All tasks scored.")

        # Return dict is {task_py: (task_name, (mlp_score, mlp_eval_size), (knn_score, knn_eval_size), private, domain, task_type, criterion, metric)}
        # print(return_dict)

        # Print results

        # Keep only public tasks and sort
        public_items = [(k, v) for k, v in return_dict.items() if v[3] is False]
        public_items.sort(key=lambda kv: (kv[1][4], kv[1][0]))  # by domain then task name

        # Helper: safe 3-decimal string or blank
        def fmt3(x):
            return f"{x:.3f}" if x is not None else ""
        
        # Helper: stringify criterion which can be a name or a callable
        def str_criterion(c):
            if isinstance(c, str):
                return c
            name = getattr(c, "__name__", None)
            if name is not None:
                return name
            return type(c).__name__

        # Build quick access structures
        domains = ["Environment", "Music", "Speech"]
        encoder_col = "**" + ckpt_name + "**" if ckpt_name else args.encoder_py

        # ---- Compute weighted averages per domain and overall 
        def dict_for_domain(domain):
            return {k: (v[1], v[2]) for k, v in return_dict.items() if (v[4] == domain and v[3] is False)}

        # weighted averages return (mlp_avg, knn_avg) when given mapping {k: ((mlp_score, mlp_size), (knn_score, knn_size))}
        mlp_env, knn_env = weighted_average({k: v[1:3] for k, v in return_dict.items() if (v[4] == "Environment" and v[3] is False)})
        mlp_mus, knn_mus = weighted_average({k: v[1:3] for k, v in return_dict.items() if (v[4] == "Music" and v[3] is False)})
        mlp_spe, knn_spe = weighted_average({k: v[1:3] for k, v in return_dict.items() if (v[4] == "Speech" and v[3] is False)})

        mlp_all, knn_all = weighted_average({k: v[1:3] for k, v in return_dict.items() if v[3] is False})

        domain_avg = {
            "Environment": (mlp_env, knn_env),
            "Music":       (mlp_mus, knn_mus),
            "Speech":      (mlp_spe, knn_spe),
        }

        # ---- Compute domain weights (sum of eval weights) and overall weights for MLP and KNN
        domain_weights_mlp = {
            dom: sum(v[1][1] for _, v in public_items if v[4] == dom)
            for dom in domains
        }
        domain_weights_knn = {
            dom: sum(v[2][1] for _, v in public_items if v[4] == dom)
            for dom in domains
        }
        overall_weight_mlp = sum(v[1][1] for _, v in public_items)
        overall_weight_knn = sum(v[2][1] for _, v in public_items)

        # ---- Collect rows for MLP and KNN markdown tables
        mlp_rows = []
        knn_rows = []

        for dom in domains:
            # Domain average row (Task and Type blank)
            mlp_dom_avg, knn_dom_avg = domain_avg.get(dom, (None, None))
            mlp_rows.append(("**"+dom+"**", "", "", "", "", domain_weights_mlp.get(dom, 0), fmt3(mlp_dom_avg) if mlp_dom_avg is not None else ""))
            knn_rows.append(("**"+dom+"**", "", "", "", "", domain_weights_knn.get(dom, 0), fmt3(knn_dom_avg) if knn_dom_avg is not None else ""))

            # Task rows
            for k, v in public_items:
                task_name, (mlp_score, mlp_w), (knn_score, knn_w), _, v_dom, task_type, criterion, metric_name = v
                if v_dom != dom:
                    continue
                if mlp_w != 0:
                    mlp_rows.append(("", task_name, task_type, metric_name, str_criterion(criterion), mlp_w, fmt3(mlp_score)))
                if knn_w != 0:
                    knn_rows.append(("", task_name, task_type, metric_name, str_criterion(criterion), knn_w, fmt3(knn_score)))

        # Overall average rows (Domain "Overall", Task and Type blank)
        mlp_rows.append(("Overall", "", "", "", "", overall_weight_mlp, fmt3(mlp_all)))
        knn_rows.append(("Overall", "", "", "", "", overall_weight_knn, fmt3(knn_all)))

        # ---- Render Markdown
        def render_markdown(rows, header_title):
            lines = []
            # column headers
            header = ["**Domain**", "**Task**", "**Type**", "**Metric**", "**Criterion**", "**Weight**", "**"+encoder_col+"**"]
            widths = [max(len(str(r[i])) for r in ([header] + rows)) for i in range(7)]

            def fmt_row(row):
                return "| " + " | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row)) + " |"

            # build table
            lines.append(f"\n### {header_title}\n")
            lines.append(fmt_row(header))
            lines.append("| :" + ": | :".join("-" * (widths[i]-2) for i in range(7)) + ": |")
            for dom, task, typ, metric_name, crit, weight, val in rows:
                display_val = val if val != "0.000" else ""
                lines.append(fmt_row([dom or "", task or "", typ or "", metric_name or "", crit or "", weight, display_val]))
            return "\n".join(lines)

        print("\n" + "-" * 100)
        print(f"\nEncoder: {args.encoder_py}")
        print("\n" + "-" * 100)

        print(render_markdown(mlp_rows, "MLP Results"))
        print()
        print(render_markdown(knn_rows, "KNN Results"))
        print("\n" + "-" * 100)


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
