from pathlib import Path
from typing import Any, List

from loguru import logger
from tqdm import tqdm


def mkdir_if_not_exists(dir: Path, main_process: bool = True):
    dir = Path(dir)
    if not dir.exists() and main_process:
        logger.info(f"Creating directory {dir} ...")
        dir.mkdir(parents=True, exist_ok=True)


def download_file(url, target_path, chunk_size=8192):
    import requests

    target_path = Path(target_path) if isinstance(target_path, str) else target_path
    existing_size = target_path.stat().st_size if target_path.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"}

    with requests.get(url, headers=headers, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0)) + existing_size
        with (
            open(target_path, "ab") as f,
            tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=f"Downloading {target_path.name}",
                leave=True,
                initial=existing_size,
            ) as pbar,
        ):
            for chunk in r.iter_content(chunk_size):
                size = f.write(chunk)
                pbar.update(size)

    pbar.set_description(f"Downloaded {target_path.name} ({total_size} bytes)")
    print()  # Add a newline after tqdm


def unzip_file(zip_file, dest_dir):
    import zipfile

    mkdir_if_not_exists(dest_dir)
    logger.info(f"Unzipping {zip_file} to {dest_dir} ...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    logger.info(f"Unzipping completed: {zip_file} extracted to {dest_dir}.")


def untar_file(tar_file, dest_dir):
    import tarfile

    mkdir_if_not_exists(dest_dir)
    logger.info(f"Extracting  {tar_file} to {dest_dir} ...")
    with tarfile.open(tar_file, "r|gz") as tar_ref:
        tar_ref.extractall(dest_dir)
    logger.info(f"Extracting completed: {tar_file} extracted to {dest_dir}.")


def download_zenodo_record(zenodo_id: str, target_dir: str, force_download: bool = False, temp_dir: None | str = None):
    import shutil
    import tempfile
    import os

    target_zip_path = Path(target_dir) / f"{zenodo_id}.zip"
    if not force_download and target_zip_path.exists():
        logger.info(f"{target_zip_path} already exists, skipping download.")
    else:
        temp_zip_path = Path(tempfile.gettempdir()) / f"{zenodo_id}.zip"
        zenodo_archive_url = f"https://zenodo.org/api/records/{zenodo_id}/files-archive"
        download_file(zenodo_archive_url, temp_zip_path)
        shutil.move(temp_zip_path, target_zip_path)
        logger.info(f"Downloading completed: {zenodo_id} saved to {target_zip_path}.")

    try:
        unzipped_flag = Path(target_dir) / ".unzipped"
        if not unzipped_flag.exists():
            unzip_file(target_zip_path, target_dir)
            unzipped_flag.touch()
            # Remove zip file to save space
            if target_zip_path.exists():
                logger.info(f"Removing {target_zip_path} to save space.")
                os.remove(target_zip_path)
        else:
            logger.info(f"{target_zip_path} already unzipped, skipping unzip.")
    except Exception as e:
        logger.error(f"Failed to unzip {target_zip_path} to {target_dir}: {e}.")
        logger.error(f"Remove {target_zip_path} and retry, or download manually using `download_data.sh`.")
        raise e


def attr_from_py_path(path: str, endswith: str | None = None) -> type:
    from importlib import import_module

    module_name = path.replace("/", ".")
    # Strip ending
    if module_name.endswith(".py"):
        module_name = module_name[:-3]  # Remove last 3 characters (".py")

    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(f"Module not found: {module_name}")

    attr_list = [m for m in dir(module) if not endswith or m.endswith(endswith)]
    if len(attr_list) != 1:
        raise ValueError(f"Expected 1 class with endswith={endswith}, got {len(attr_list)}")

    return getattr(module, attr_list[0])


def download_hf_model_to_local(model_names: str | List[str], output_root: str = "."):
    # Download the model to the local directory to avoid issues under multi-processes
    if isinstance(model_names, str):
        model_names = [model_names]

    if "bert-base-uncased:tokenizer" in model_names:
        model_name = "google-bert/bert-base-uncased"
        output_path = Path(output_root) / model_name
        if not output_path.exists():
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(output_path)
        model_names.remove("bert-base-uncased:tokenizer")

    if "qwen2" in model_names:
        model_name = "Qwen/Qwen2.5-0.5B"
        output_path = Path(output_root) / model_name
        if not output_path.exists():
            from transformers import AutoModelForCausalLM, AutoTokenizer

            for m in [AutoModelForCausalLM.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)]:
                m.save_pretrained(output_path)
        model_names.remove("qwen2")

    if len(model_names) > 0:
        logger.warning(f"Models {model_names} are not supported for download")


def get_encoder_run_name(
    encoder: Any | None = None, 
    encoder_py_path: str | Path | None = None, 
    encoder_checkpoint_path: str | Path | None = None
) -> str:
    from pathlib import Path
    import os
    from loguru import logger
    
    ckpt_name = None
    
    if encoder_checkpoint_path is None and encoder is not None:
        encoder_checkpoint_path = getattr(encoder, "checkpoint_path", None)

    if encoder_checkpoint_path:
        try:
            ckpt_path = Path(encoder_checkpoint_path).resolve()
            potential_config_dirs = [ckpt_path.parent, ckpt_path.parent.parent]
            
            for config_dir in potential_config_dirs:
                hydra_config_path = config_dir / ".hydra" / "config.yaml"
                if hydra_config_path.exists():
                    try:
                        from omegaconf import OmegaConf
                        conf = OmegaConf.load(hydra_config_path)
                        if conf and "logger" in conf and "wandb" in conf.logger and "name" in conf.logger.wandb:
                            run_name = conf.logger.wandb.name
                            if run_name:
                                ckpt_name = run_name
                                logger.info(f"Using wandb run name '{ckpt_name}' from {hydra_config_path}")
                                break
                    except ImportError:
                        logger.warning("omegaconf not installed, skipping hydra config check")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to read wandb name from {hydra_config_path}: {e}")
        except Exception as e:
            logger.debug(f"Error checking for hydra config: {e}")

        if not ckpt_name:
            ckpt_name = Path(encoder_checkpoint_path).stem
    else:
        ckpt_path_env = os.environ.get("AUDIO_JEPA_CHECKPOINT")
        if ckpt_path_env:
            ckpt_name = Path(ckpt_path_env).stem
    
    if not ckpt_name:
         if encoder_py_path:
             ckpt_name = Path(encoder_py_path).stem
         elif encoder:
             ckpt_name = encoder.__class__.__name__
    
    if not ckpt_name:
         ckpt_name = "unknown_checkpoint"
         
    return ckpt_name
